"""Function for performing coalesce."""
from typing import Optional, Union
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, deprecated_alias
from janitor.functions.utils import _select_column_names


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_column_name="target_column_name")
def coalesce(
    df: pd.DataFrame,
    *column_names,
    target_column_name: Optional[str] = None,
    default_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """Coalesce two or more columns of data in order of column names provided.

    Given the variable arguments of column names,
    `coalesce` finds and returns the first non-missing value
    from these columns, for every row in the input dataframe.
    If all the column values are null for a particular row,
    then the `default_value` will be filled in.

    If `target_column_name` is not provided,
    then the first column is coalesced.

    This method does not mutate the original DataFrame.

    Example: Use `coalesce` with 3 columns, "a", "b" and "c".

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": [np.nan, 1, np.nan],
        ...     "b": [2, 3, np.nan],
        ...     "c": [4, np.nan, np.nan],
        ... })
        >>> df.coalesce("a", "b", "c")
             a    b    c
        0  2.0  2.0  4.0
        1  1.0  3.0  NaN
        2  NaN  NaN  NaN

    Example: Provide a target_column_name.

        >>> df.coalesce("a", "b", "c", target_column_name="new_col")
             a    b    c  new_col
        0  NaN  2.0  4.0      2.0
        1  1.0  3.0  NaN      1.0
        2  NaN  NaN  NaN      NaN

    Example: Provide a default value.

        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": [1, np.nan, np.nan],
        ...     "b": [2, 3, np.nan],
        ... })
        >>> df.coalesce(
        ...     "a", "b",
        ...     target_column_name="new_col",
        ...     default_value=-1,
        ... )
             a    b  new_col
        0  1.0  2.0      1.0
        1  NaN  3.0      3.0
        2  NaN  NaN     -1.0

    This is more syntactic diabetes! For R users, this should look familiar to
    `dplyr`'s `coalesce` function; for Python users, the interface
    should be more intuitive than the `pandas.Series.combine_first`
    method.

    :param df: A pandas DataFrame.
    :param column_names: A list of column names.
    :param target_column_name: The new column name after combining.
        If `None`, then the first column in `column_names` is updated,
        with the Null values replaced.
    :param default_value: A scalar to replace any remaining nulls
        after coalescing.
    :returns: A pandas DataFrame with coalesced columns.
    :raises ValueError: if length of `column_names` is less than 2.
    """

    if not column_names:
        return df

    if len(column_names) < 2:
        raise ValueError(
            "The number of columns to coalesce should be a minimum of 2."
        )

    column_names = _select_column_names([*column_names], df)

    if target_column_name:
        check("target_column_name", target_column_name, [str])

    if default_value:
        check("default_value", default_value, [int, float, str])

    if target_column_name is None:
        target_column_name = column_names[0]

    outcome = df.filter(column_names).bfill(axis="columns").iloc[:, 0]
    if outcome.hasnans and (default_value is not None):
        outcome = outcome.fillna(default_value)

    return df.assign(**{target_column_name: outcome})
