from functools import partial
from typing import Optional, Union

import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, check_column, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", type="cleaning_style")
def currency_column_to_numeric(
    df: pd.DataFrame,
    column_name: str,
    cleaning_style: Optional[str] = None,
    cast_non_numeric: Optional[dict] = None,
    fill_all_non_numeric: Optional[Union[float, int]] = None,
    remove_non_numeric: bool = False,
) -> pd.DataFrame:
    """Convert currency column to numeric.

    This method does not mutate the original DataFrame.

    This method allows one to take a column containing currency values,
    inadvertently imported as a string, and cast it as a float. This is
    usually the case when reading CSV files that were modified in Excel.
    Empty strings (i.e. `''`) are retained as `NaN` values.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a_col": [" 24.56", "-", "(12.12)", "1,000,000"],
        ...     "d_col": ["", "foo", "1.23 dollars", "-1,000 yen"],
        ... })
        >>> df  # doctest: +NORMALIZE_WHITESPACE
               a_col         d_col
        0      24.56
        1          -           foo
        2    (12.12)  1.23 dollars
        3  1,000,000    -1,000 yen

        The default cleaning style.

        >>> df.currency_column_to_numeric("d_col")
               a_col    d_col
        0      24.56      NaN
        1          -      NaN
        2    (12.12)     1.23
        3  1,000,000 -1000.00

        The accounting cleaning style.

        >>> df.currency_column_to_numeric("a_col", cleaning_style="accounting")  # doctest: +NORMALIZE_WHITESPACE
                a_col         d_col
        0       24.56
        1        0.00           foo
        2      -12.12  1.23 dollars
        3  1000000.00    -1,000 yen

    Valid cleaning styles are:

    - `None`: Default cleaning is applied. Empty strings are always retained as
        `NaN`. Numbers, `-`, `.` are extracted and the resulting string
        is cast to a float.
    - `'accounting'`: Replaces numbers in parentheses with negatives, removes commas.

    Args:
        df: The pandas DataFrame.
        column_name: The column containing currency values to modify.
        cleaning_style: What style of cleaning to perform.
        cast_non_numeric: A dict of how to coerce certain strings to numeric
            type. For example, if there are values of 'REORDER' in the DataFrame,
            `{'REORDER': 0}` will cast all instances of 'REORDER' to 0.
            Only takes effect in the default cleaning style.
        fill_all_non_numeric: Similar to `cast_non_numeric`, but fills all
            strings to the same value. For example, `fill_all_non_numeric=1`, will
            make everything that doesn't coerce to a currency `1`.
            Only takes effect in the default cleaning style.
        remove_non_numeric: If set to True, rows of `df` that contain
            non-numeric values in the `column_name` column will be removed.
            Only takes effect in the default cleaning style.

    Raises:
        ValueError: If `cleaning_style` is not one of the accepted styles.

    Returns:
        A pandas DataFrame.
    """  # noqa: E501

    check("column_name", column_name, [str])
    check_column(df, column_name)

    column_series = df[column_name]
    if cleaning_style == "accounting":
        outcome = (
            df[column_name]
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("(", "-", regex=False)
            .replace({"-": 0.0})
            .astype(float)
        )
        return df.assign(**{column_name: outcome})
    if cleaning_style is not None:
        raise ValueError(
            "`cleaning_style` is expected to be one of ('accounting', None). "
            f"Got {cleaning_style!r} instead."
        )

    if cast_non_numeric:
        check("cast_non_numeric", cast_non_numeric, [dict])

    _make_cc_patrial = partial(
        _currency_column_to_numeric,
        cast_non_numeric=cast_non_numeric,
    )
    column_series = column_series.apply(_make_cc_patrial)

    if remove_non_numeric:
        df = df.loc[column_series != "", :]

    # _replace_empty_string_with_none is applied here after the check on
    # remove_non_numeric since "" is our indicator that a string was coerced
    # in the original column
    column_series = _replace_empty_string_with_none(column_series)

    if fill_all_non_numeric is not None:
        check("fill_all_non_numeric", fill_all_non_numeric, [int, float])
        column_series = column_series.fillna(fill_all_non_numeric)

    column_series = _replace_original_empty_string_with_none(column_series)

    df = df.assign(**{column_name: pd.to_numeric(column_series)})

    return df


def _currency_column_to_numeric(
    x: str,
    cast_non_numeric: Optional[dict] = None,
) -> Union[int, float, str]:
    """
    Perform logic for changing cell values.

    This is a private function intended to be used only in
    `currency_column_to_numeric`.

    It is intended to be used in a pandas `apply` method, after being passed
    through `partial`.

    Args:
        x: A string representing currency.
        cast_non_numeric: A dict of how to coerce certain strings to numeric
            type. For example, if there are values of 'REORDER' in the
            DataFrame, `{'REORDER': 0}` will cast all instances of 'REORDER'
            to 0.
    """
    acceptable_currency_characters = {
        "-",
        ".",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    }
    if not x:
        return "ORIGINAL_NA"

    if cast_non_numeric:
        if x in cast_non_numeric:
            mapped_x = cast_non_numeric[x]
            check(
                "{%r: %r}" % (x, str(mapped_x)),
                mapped_x,
                [int, float],
            )
            return mapped_x
    return "".join(i for i in x if i in acceptable_currency_characters)


def _replace_empty_string_with_none(column_series):
    column_series.loc[column_series == ""] = None
    return column_series


def _replace_original_empty_string_with_none(column_series):
    """Replaces original empty string with None"""
    column_series.loc[column_series == "ORIGINAL_NA"] = None
    return column_series
