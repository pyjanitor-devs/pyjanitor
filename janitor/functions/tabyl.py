"""Implementation of the `tabyl` function for frequency tables."""

from __future__ import annotations

from typing import Hashable

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def tabyl(
    df: pd.DataFrame,
    *column_names: Hashable,
    show_na: bool = True,
    show_missing_levels: bool = True,
) -> pd.DataFrame | dict[Hashable, pd.DataFrame]:
    """Create frequency tables (1-way, 2-way, or 3-way).

    This is a Python implementation of R janitor's `tabyl()` function,
    which creates frequency tables as DataFrames. It supports:

    - **1-way tabyl**: Counts and percentages for a single variable
    - **2-way tabyl**: Crosstab (contingency table) of two variables
    - **3-way tabyl**: Dictionary of 2-way tabyls split by a third variable

    This method does not mutate the original DataFrame.

    Examples:
        1-way tabyl (frequency table of a single variable):

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "cyl": [4, 6, 8, 4, 6, 8, 4, 6],
        ...         "gear": [3, 3, 3, 4, 4, 4, 5, 5],
        ...         "am": [0, 0, 0, 1, 1, 1, 1, 1],
        ...     }
        ... )
        >>> df.tabyl("cyl")
           cyl  n  percent
        0    4  3    0.375
        1    6  3    0.375
        2    8  2    0.250

        2-way tabyl (crosstab):

        >>> df.tabyl("cyl", "gear")
           cyl  3  4  5
        0    4  1  1  1
        1    6  1  1  1
        2    8  1  1  0

        3-way tabyl (dictionary of 2-way tabyls split by third variable):

        >>> result = df.tabyl("cyl", "gear", "am")
        >>> result[0]  # Crosstab for am=0
           cyl  3
        0    4  1
        1    6  1
        2    8  1
        >>> result[1]  # Crosstab for am=1
           cyl  4  5
        0    4  1  1
        1    6  1  1
        2    8  1  0

    Args:
        df: The pandas DataFrame object.
        *column_names: One, two, or three column names.
            - 1 column: Returns counts and percentages
            - 2 columns: Returns crosstab (first as rows, second as columns)
            - 3 columns: Returns dict of crosstabs split by third variable
        show_na: Whether to include NA values in the output.
            Defaults to True.
        show_missing_levels: Whether to show zero counts for missing
            categorical levels. Defaults to True.

    Raises:
        ValueError: If fewer than 1 or more than 3 column names are provided.
        KeyError: If any column name is not found in the DataFrame.

    Returns:
        For 1-way and 2-way tabyls: A pandas DataFrame.
        For 3-way tabyls: A dictionary mapping values of the third variable
        to DataFrames (2-way tabyls).
    """
    if len(column_names) == 0:
        raise ValueError("At least one column name must be provided.")
    if len(column_names) > 3:
        raise ValueError("At most three column names can be provided.")

    # Validate that all columns exist
    for col in column_names:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    if len(column_names) == 1:
        return _tabyl_one(
            df,
            column_names[0],
            show_na=show_na,
            show_missing_levels=show_missing_levels,
        )
    elif len(column_names) == 2:
        return _tabyl_two(
            df,
            column_names[0],
            column_names[1],
            show_na=show_na,
            show_missing_levels=show_missing_levels,
        )
    else:
        return _tabyl_three(
            df,
            column_names[0],
            column_names[1],
            column_names[2],
            show_na=show_na,
            show_missing_levels=show_missing_levels,
        )


def _tabyl_one(
    df: pd.DataFrame,
    col: Hashable,
    show_na: bool,
    show_missing_levels: bool,
) -> pd.DataFrame:
    """Create a 1-way frequency table.

    Args:
        df: The pandas DataFrame.
        col: The column name to tabulate.
        show_na: Whether to include NA values.
        show_missing_levels: Whether to show missing categorical levels.

    Returns:
        DataFrame with columns: [col, 'n', 'percent'] and optionally
        'valid_percent' if there are NA values.
    """
    series = df[col]

    # Handle categorical: if not showing missing levels, convert to non-cat
    if not show_missing_levels and hasattr(series, "cat"):
        # Remove unused categories to exclude them from output
        series = series.cat.remove_unused_categories()

    # Handle categorical with missing levels
    if show_missing_levels and hasattr(series, "cat"):
        # For categorical, use all categories
        counts = series.value_counts(dropna=not show_na)
        # Add missing categories with count 0
        all_categories = series.cat.categories
        for cat in all_categories:
            if cat not in counts.index:
                counts[cat] = 0
        counts = counts.sort_index()
    else:
        counts = series.value_counts(dropna=not show_na)
        counts = counts.sort_index()

    total = counts.sum()
    valid_total = series.dropna().shape[0] if show_na else total

    result = pd.DataFrame({col: counts.index, "n": counts.values})

    # Calculate percent (including NA in denominator)
    result["percent"] = result["n"] / total if total > 0 else 0.0

    # Add valid_percent only if there are NA values and show_na is True
    has_na = bool(series.isna().any())
    if has_na and show_na:
        # valid_percent excludes NA from both numerator and denominator
        result["valid_percent"] = result.apply(
            lambda row: (
                row["n"] / valid_total
                if valid_total > 0 and pd.notna(row[col])
                else None
            ),
            axis=1,
        )

    return result.reset_index(drop=True)


def _tabyl_two(
    df: pd.DataFrame,
    row_col: Hashable,
    col_col: Hashable,
    show_na: bool,
    show_missing_levels: bool,
) -> pd.DataFrame:
    """Create a 2-way frequency table (crosstab).

    Args:
        df: The pandas DataFrame.
        row_col: The column for rows.
        col_col: The column for columns.
        show_na: Whether to include NA values.
        show_missing_levels: Whether to show missing categorical levels.

    Returns:
        DataFrame with row_col as first column and counts as remaining columns.
    """
    # Create crosstab
    crosstab = pd.crosstab(
        df[row_col],
        df[col_col],
        dropna=not show_na,
    )

    # Handle missing categorical levels for rows
    if show_missing_levels and hasattr(df[row_col], "cat"):
        all_row_cats = df[row_col].cat.categories
        for cat in all_row_cats:
            if cat not in crosstab.index:
                crosstab.loc[cat] = 0
        crosstab = crosstab.sort_index()

    # Handle missing categorical levels for columns
    if show_missing_levels and hasattr(df[col_col], "cat"):
        all_col_cats = df[col_col].cat.categories
        for cat in all_col_cats:
            if cat not in crosstab.columns:
                crosstab[cat] = 0
        crosstab = crosstab[sorted(crosstab.columns)]

    # Convert to desired format: first column is row_col values
    result = crosstab.reset_index()
    result.columns.name = None  # Remove the column axis name

    return result


def _tabyl_three(
    df: pd.DataFrame,
    row_col: Hashable,
    col_col: Hashable,
    split_col: Hashable,
    show_na: bool,
    show_missing_levels: bool,
) -> dict[Hashable, pd.DataFrame]:
    """Create a 3-way frequency table (dict of crosstabs split by third var).

    Args:
        df: The pandas DataFrame.
        row_col: The column for rows.
        col_col: The column for columns.
        split_col: The column to split by.
        show_na: Whether to include NA values.
        show_missing_levels: Whether to show missing categorical levels.

    Returns:
        Dictionary mapping values of split_col to 2-way tabyl DataFrames.
    """
    result = {}

    # Get unique values of split column
    if show_na:
        split_values = df[split_col].unique()
    else:
        split_values = df[split_col].dropna().unique()

    # Handle missing categorical levels for split column
    if show_missing_levels and hasattr(df[split_col], "cat"):
        all_split_cats = df[split_col].cat.categories
        split_values = list(set(split_values) | set(all_split_cats))

    for val in sorted(split_values, key=lambda x: (pd.isna(x), x)):
        if pd.isna(val):
            mask = df[split_col].isna()
        else:
            mask = df[split_col] == val
        subset = df.loc[mask]

        if len(subset) > 0 or show_missing_levels:
            subset_df = subset if len(subset) > 0 else df.iloc[:0]
            tabyl_result = _tabyl_two(
                subset_df,
                row_col,
                col_col,
                show_na=show_na,
                show_missing_levels=show_missing_levels,
            )
            result[val] = tabyl_result

    return result
