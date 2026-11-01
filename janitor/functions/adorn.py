"""Implementation of `adorn_*` functions for tabyl formatting.

These functions are inspired by the R janitor package's adorn_* family of
functions, which transform raw frequency counts into publication-ready tables.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


def numeric_column_positions(frame: pd.DataFrame) -> list[int]:
    """Return positions of columns that `select_dtypes` treats as numeric.

    Positions keep duplicate column labels distinct. An empty frame returns
    an empty list. The dtype test matches `DataFrame.select_dtypes` with
    `include=[np.number]`, including pandas nullable integer and floating
    dtypes and excluding booleans.

    Args:
        frame: A pandas DataFrame.

    Returns:
        Integer column positions, in ascending order.
    """
    if frame.shape[1] == 0:
        return []
    # Relabel a zero-row view by position so select_dtypes reports positions
    # instead of labels, while keeping its dtype rules.
    numbered = frame.iloc[:0].set_axis(range(frame.shape[1]), axis="columns")
    selected = numbered.select_dtypes(include=[np.number]).columns
    return [int(position) for position in selected]


def adorn_data_positions(frame: pd.DataFrame) -> list[int]:
    """Return numeric column positions other than the row identifier.

    Column position 0 is the row identifier for the `adorn_*` functions and
    is never modified, even when it is numeric. This follows R janitor, which
    drops column index 1 from the numeric column set before adorning. Only
    that position is exempt: a later column that repeats the first column's
    label is still adorned.

    Args:
        frame: A pandas DataFrame.

    Returns:
        Integer column positions, in ascending order.
    """
    return [position for position in numeric_column_positions(frame) if position != 0]


def adorn_count_pairs(df: pd.DataFrame, ns: pd.DataFrame) -> list[tuple[int, int]]:
    """Pair columns of `df` with the numeric count columns of `ns`.

    Column position 0 of `df` is the row identifier and is never paired.
    Every numeric column of `ns` is a count, because a counts frame passed
    through `ns` does not have to repeat the identifier column.

    When `ns` has the same columns as `df`, pairs are positional. That is
    how counts stored by `adorn_percentages` line up, including when a label
    is repeated. Otherwise pairs are by label, and a repeated label is
    consumed once per occurrence so each column reads its own count.

    Args:
        df: The frame being adorned.
        ns: The frame of original counts.

    Returns:
        `(df_position, ns_position)` pairs in `df` column order.
    """
    ns_numeric = numeric_column_positions(ns)
    if df.columns.equals(ns.columns):
        return [
            (position, position)
            for position in ns_numeric
            if position != 0 and position < df.shape[1]
        ]

    counts_by_label: dict[object, list[int]] = {}
    for ns_position in ns_numeric:
        label = ns.columns[ns_position]
        counts_by_label.setdefault(label, []).append(ns_position)

    seen: dict[object, int] = {}
    pairs: list[tuple[int, int]] = []
    for position in range(1, df.shape[1]):
        label = df.columns[position]
        candidates = counts_by_label.get(label)
        if not candidates:
            continue
        occurrence = seen.get(label, 0)
        seen[label] = occurrence + 1
        if occurrence < len(candidates):
            pairs.append((position, candidates[occurrence]))
    return pairs


@pf.register_dataframe_method
def adorn_totals(
    df: pd.DataFrame,
    where: Literal["row", "col", "both"] = "row",
    fill: str = "-",
    na_rm: bool = True,
    name: str = "Total",
) -> pd.DataFrame:
    """Add totals row and/or column to a DataFrame.

    This function adds a row and/or column with sum totals for numeric
    columns. It is particularly useful when working with frequency tables
    (tabyls) but works on any DataFrame with numeric columns.

    Examples:
        Add a totals row to a DataFrame.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {"category": ["A", "B"], "count1": [10, 20], "count2": [5, 15]}
        ... )
        >>> df.adorn_totals("row")
          category  count1  count2
        0        A      10       5
        1        B      20      15
        2    Total      30      20

        Add totals to both rows and columns.

        >>> df.adorn_totals("both")
          category  count1  count2  Total
        0        A      10       5     15
        1        B      20      15     35
        2    Total      30      20     50

    Args:
        df: A pandas DataFrame.
        where: Where to add totals. One of "row" (add a totals row),
            "col" (add a totals column), or "both" (add both).
        fill: Value to use for non-numeric columns in the totals row.
        na_rm: If True, remove NA values before summing.
        name: Name for the totals row/column.

    Raises:
        ValueError: If `where` is not one of "row", "col", or "both".

    Returns:
        A pandas DataFrame with totals added.
    """
    check("where", where, [str])
    check("fill", fill, [str])
    check("name", name, [str])

    valid_where = {"row", "col", "both"}
    if where not in valid_where:
        raise ValueError(f"`where` must be one of {valid_where}, got '{where}'.")

    df = df.copy()

    # Store original counts in attrs for adorn_ns to use later
    if "_original_counts" not in df.attrs:
        df.attrs["_original_counts"] = df.copy()

    # Identify numeric columns, excluding the first column which is treated
    # as a row identifier (consistent with R janitor's tabyl behavior)
    first_col = df.columns[0]
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns.tolist()
        if col != first_col
    ]

    if where in ("col", "both"):
        # Add totals column
        df[name] = df[numeric_cols].sum(axis=1, skipna=na_rm)

    if where in ("row", "both"):
        # Create totals row
        totals_row = {}
        for col in df.columns:
            if col in numeric_cols or col == name:
                totals_row[col] = df[col].sum(skipna=na_rm)
            elif col == first_col:
                # First column gets the totals row name (e.g., "Total")
                totals_row[col] = name
            else:
                # All other non-numeric columns get the fill value
                totals_row[col] = fill

        totals_df = pd.DataFrame([totals_row])
        df = pd.concat([df, totals_df], ignore_index=True)

    return df


@pf.register_dataframe_method
def adorn_percentages(
    df: pd.DataFrame,
    denominator: Literal["row", "col", "all"] = "row",
    na_rm: bool = True,
) -> pd.DataFrame:
    """Convert counts to percentages (row-wise, column-wise, or overall).

    This function converts numeric columns to percentages based on the
    specified denominator. It is particularly useful after creating
    frequency tables.

    Examples:
        Convert counts to row percentages.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {"category": ["A", "B"], "count1": [10, 20], "count2": [5, 15]}
        ... )
        >>> df.adorn_percentages("row")
          category    count1    count2
        0        A  0.666667  0.333333
        1        B  0.571429  0.428571

        Convert to column percentages.

        >>> df.adorn_percentages("col")
          category    count1  count2
        0        A  0.333333    0.25
        1        B  0.666667    0.75

    Args:
        df: A pandas DataFrame.
        denominator: How to calculate percentages. One of "row" (row totals),
            "col" (column totals), or "all" (grand total).
        na_rm: If True, remove NA values when calculating totals.

    Raises:
        ValueError: If `denominator` is not one of "row", "col", or "all".

    Returns:
        A pandas DataFrame with counts converted to percentages.
    """
    check("denominator", denominator, [str])

    valid_denominators = {"row", "col", "all"}
    if denominator not in valid_denominators:
        raise ValueError(
            f"`denominator` must be one of {valid_denominators}, got '{denominator}'."
        )

    df = df.copy()

    # Store original counts in attrs for adorn_ns to use later
    if "_original_counts" not in df.attrs:
        df.attrs["_original_counts"] = df.copy()

    # Identify numeric columns, excluding the first column which is treated
    # as a row identifier (consistent with R janitor's tabyl behavior)
    first_col = df.columns[0]
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns.tolist()
        if col != first_col
    ]

    if not numeric_cols:
        return df

    if denominator == "row":
        row_totals = df[numeric_cols].sum(axis=1, skipna=na_rm)
        for col in numeric_cols:
            df[col] = df[col] / row_totals
    elif denominator == "col":
        for col in numeric_cols:
            col_total = df[col].sum(skipna=na_rm)
            if col_total != 0:
                df[col] = df[col] / col_total
    else:  # "all"
        grand_total = df[numeric_cols].sum(skipna=na_rm).sum()
        if grand_total != 0:
            for col in numeric_cols:
                df[col] = df[col] / grand_total

    return df


@pf.register_dataframe_method
def adorn_pct_formatting(
    df: pd.DataFrame,
    digits: int = 1,
    rounding: Literal["half to even", "half up"] = "half to even",
    affix_sign: bool = True,
) -> pd.DataFrame:
    """Format decimal percentages as "XX.X%" strings.

    This function formats numeric columns (assumed to be proportions between
    0 and 1) as percentage strings with the specified number of decimal places.

    The column in position 0 is the row identifier and is left unchanged,
    even when it is numeric. This matches `adorn_totals` and
    `adorn_percentages`. A later column that repeats the identifier's
    label is still formatted.

    Examples:
        Format percentages with default settings.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "category": ["A", "B"],
        ...         "pct1": [0.666667, 0.571429],
        ...         "pct2": [0.333333, 0.428571],
        ...     }
        ... )
        >>> df.adorn_pct_formatting()
          category   pct1   pct2
        0        A  66.7%  33.3%
        1        B  57.1%  42.9%

        Format without percent sign.

        >>> df.adorn_pct_formatting(affix_sign=False)
          category  pct1  pct2
        0        A  66.7  33.3
        1        B  57.1  42.9

    Args:
        df: A pandas DataFrame.
        digits: Number of decimal places to show.
        rounding: Rounding method. One of "half to even" (banker's rounding)
            or "half up" (standard rounding).
        affix_sign: If True, append "%" to the formatted values.

    Raises:
        ValueError: If `rounding` is not one of the valid options.

    Returns:
        A pandas DataFrame with percentages formatted as strings.
    """
    check("digits", digits, [int])
    check("rounding", rounding, [str])

    valid_rounding = {"half to even", "half up"}
    if rounding not in valid_rounding:
        raise ValueError(
            f"`rounding` must be one of {valid_rounding}, got '{rounding}'."
        )

    df = df.copy()

    # Preserve original counts if they exist
    original_counts = df.attrs.get("_original_counts")

    rounding_mode = ROUND_HALF_EVEN if rounding == "half to even" else ROUND_HALF_UP
    quantize_str = f"0.{'0' * digits}" if digits > 0 else "0"

    def format_pct(value):
        if pd.isna(value):
            return value
        # Convert to percentage (multiply by 100)
        pct_value = value * 100
        # Round using Decimal for precision
        rounded = Decimal(str(pct_value)).quantize(
            Decimal(quantize_str), rounding=rounding_mode
        )
        result = str(rounded)
        if affix_sign:
            result += "%"
        return result

    for position in adorn_data_positions(df):
        df.isetitem(position, df.iloc[:, position].apply(format_pct))

    # Restore original counts in attrs
    if original_counts is not None:
        df.attrs["_original_counts"] = original_counts

    return df


@pf.register_dataframe_method
def adorn_ns(
    df: pd.DataFrame,
    position: Literal["front", "rear"] = "rear",
    ns: Optional[pd.DataFrame] = None,
    format_func: Optional[Callable[[int], str]] = None,
) -> pd.DataFrame:
    """Append the original N counts to formatted percentage cells.

    This function adds the original counts (N) to cells that have been
    converted to percentages. It requires either the original counts to be
    stored in the DataFrame's attrs (via prior use of adorn_percentages)
    or to be passed via the `ns` parameter.

    The column in position 0 of `df` is the row identifier and is left
    unchanged, even when it is numeric. Every numeric column of `ns` is a
    count: `ns` does not have to repeat the identifier. Columns are matched
    by position when `ns` has the same columns as `df`, and by label
    otherwise. A repeated label is matched once per occurrence. Rows of
    `df` past the end of `ns` (a totals row, for example) are left as
    they are.

    Examples:
        Add counts to formatted percentages.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {"category": ["A", "B"], "count1": [10, 20], "count2": [5, 15]}
        ... )
        >>> result = df.adorn_percentages("row").adorn_pct_formatting().adorn_ns()
        >>> result  # doctest: +SKIP
          category       count1       count2
        0        A  66.7% (10)   33.3% (5)
        1        B  57.1% (20)  42.9% (15)

    Args:
        df: A pandas DataFrame (typically after adorn_pct_formatting).
        position: Where to add the N count. "front" prepends "(N) XX%",
            "rear" appends "XX% (N)".
        ns: Optional DataFrame containing the original counts. If not
            provided, uses counts stored in df.attrs from adorn_percentages.
        format_func: Optional function to format the count values.
            Default is to format as "(N)".

    Raises:
        ValueError: If `position` is not one of "front" or "rear".
        ValueError: If no original counts are available.

    Returns:
        A pandas DataFrame with N counts appended to percentage strings.
    """
    check("position", position, [str])

    valid_positions = {"front", "rear"}
    if position not in valid_positions:
        raise ValueError(
            f"`position` must be one of {valid_positions}, got '{position}'."
        )

    # Get original counts
    if ns is None:
        ns = df.attrs.get("_original_counts")
        if ns is None:
            raise ValueError(
                "No original counts available. Either use adorn_percentages "
                "before adorn_ns, or provide counts via the `ns` parameter."
            )

    df = df.copy()

    # Default format function with thousand separator (matching R janitor behavior)
    if format_func is None:

        def _default_format_func(n):
            if pd.isna(n):
                return ""
            return f"({int(n):,})"

        format_func = _default_format_func

    # Rows are matched by position, so a frame with more rows than `ns`
    # (for example after adorn_totals) leaves the extra rows alone.
    # Columns are paired by position or label; column 0 of `df` is skipped.
    n_rows = min(len(df), len(ns))
    for df_position, ns_position in adorn_count_pairs(df, ns):
        values = df.iloc[:, df_position].tolist()
        adorned = False
        for row in range(n_rows):
            formatted_n = format_func(ns.iat[row, ns_position])
            current_value = values[row]
            if pd.notna(current_value) and formatted_n:
                if position == "rear":
                    values[row] = f"{current_value} {formatted_n}"
                else:
                    values[row] = f"{formatted_n} {current_value}"
                adorned = True
        if adorned:
            df.isetitem(df_position, values)

    return df


@pf.register_dataframe_method
def adorn_title(
    df: pd.DataFrame,
    placement: Literal["top", "combined"] = "top",
    row_name: Optional[str] = None,
    col_name: Optional[str] = None,
) -> pd.DataFrame:
    """Add variable name as title to a two-way tabyl.

    This function adds descriptive titles to the row and column dimensions
    of a cross-tabulation. It can either add a new header row with the
    column variable name, or combine row and column names in the first cell.

    Examples:
        Add title to a cross-tabulation.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"gender": ["M", "F"], "yes": [10, 15], "no": [5, 8]})
        >>> df.adorn_title(row_name="gender", col_name="response")
                  response
          gender yes  no
        0      M  10   5
        1      F  15   8

    Args:
        df: A pandas DataFrame.
        placement: Where to place the title. "top" adds a new row above
            the column headers. "combined" combines row and column names
            in the first cell as "row_name/col_name".
        row_name: Name for the row variable. If None, uses the first
            column name.
        col_name: Name for the column variable. Required for "top" placement.

    Raises:
        ValueError: If `placement` is not one of "top" or "combined".

    Returns:
        A pandas DataFrame with title added.
    """
    check("placement", placement, [str])

    valid_placements = {"top", "combined"}
    if placement not in valid_placements:
        raise ValueError(
            f"`placement` must be one of {valid_placements}, got '{placement}'."
        )

    df = df.copy()

    first_col = df.columns[0]
    if row_name is None:
        row_name = str(first_col)

    if placement == "combined":
        # Combine row and column names in the first column header
        if col_name:
            new_name = f"{row_name}/{col_name}"
        else:
            new_name = row_name
        df = df.rename(columns={first_col: new_name})
    else:  # "top"
        # Create a MultiIndex for columns with col_name as the top level
        if col_name:
            new_columns = pd.MultiIndex.from_tuples(
                [(col_name if i > 0 else "", col) for i, col in enumerate(df.columns)]
            )
            df.columns = new_columns

    return df


@pf.register_dataframe_method
def adorn_rounding(
    df: pd.DataFrame,
    digits: int = 1,
    rounding: Literal["half to even", "half up"] = "half to even",
) -> pd.DataFrame:
    """Round numeric columns with configurable rounding method.

    This function rounds numeric columns to the specified number of
    decimal places using the specified rounding method.

    The column in position 0 is the row identifier and is left unchanged,
    even when it is numeric, so an integer identifier keeps its dtype.
    This matches `adorn_totals` and `adorn_percentages`. A later column
    that repeats the identifier's label is still rounded.

    Examples:
        Round numeric columns.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "category": ["A", "B"],
        ...         "value1": [1.2345, 2.5678],
        ...         "value2": [3.4567, 4.5555],
        ...     }
        ... )
        >>> df.adorn_rounding(digits=2)
          category  value1  value2
        0        A    1.23    3.46
        1        B    2.57    4.56

        Round using "half up" method.

        >>> df.adorn_rounding(digits=1, rounding="half up")
          category  value1  value2
        0        A     1.2     3.5
        1        B     2.6     4.6

    Args:
        df: A pandas DataFrame.
        digits: Number of decimal places to round to.
        rounding: Rounding method. One of "half to even" (banker's rounding)
            or "half up" (standard rounding).

    Raises:
        ValueError: If `rounding` is not one of the valid options.

    Returns:
        A pandas DataFrame with numeric columns rounded.
    """
    check("digits", digits, [int])
    check("rounding", rounding, [str])

    valid_rounding = {"half to even", "half up"}
    if rounding not in valid_rounding:
        raise ValueError(
            f"`rounding` must be one of {valid_rounding}, got '{rounding}'."
        )

    df = df.copy()

    # Preserve original counts if they exist
    original_counts = df.attrs.get("_original_counts")

    rounding_mode = ROUND_HALF_EVEN if rounding == "half to even" else ROUND_HALF_UP
    quantize_str = f"0.{'0' * digits}" if digits > 0 else "0"

    def round_value(value):
        if pd.isna(value):
            return value
        rounded = Decimal(str(value)).quantize(
            Decimal(quantize_str), rounding=rounding_mode
        )
        return float(rounded)

    for position in adorn_data_positions(df):
        df.isetitem(position, df.iloc[:, position].apply(round_value))

    # Restore original counts in attrs
    if original_counts is not None:
        df.attrs["_original_counts"] = original_counts

    return df
