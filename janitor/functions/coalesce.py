"""Function for performing coalesce."""

from typing import Any, Optional, Union

import pandas as pd
import pandas_flavor as pf

from janitor.functions.select import _select_index
from janitor.utils import check, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_column_name="target_column_name")
def coalesce(
    df: pd.DataFrame,
    *column_names: Any,
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

    The [`select`][janitor.functions.select.select] syntax
    can be used in `column_names`.

    Examples:
        Use `coalesce` with 3 columns, "a", "b" and "c".

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

        Provide a target_column_name.

        >>> df.coalesce("a", "b", "c", target_column_name="new_col")
             a    b    c  new_col
        0  NaN  2.0  4.0      2.0
        1  1.0  3.0  NaN      1.0
        2  NaN  NaN  NaN      NaN

        Provide a default value.

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

    Args:
        df: A pandas DataFrame.
        column_names: A list of column names.
        target_column_name: The new column name after combining.
            If `None`, then the first column in `column_names` is updated,
            with the Null values replaced.
        default_value: A scalar to replace any remaining nulls
            after coalescing.

    Raises:
        ValueError: If length of `column_names` is less than 2.

    Returns:
        A pandas DataFrame with coalesced columns.
    """

    if not column_names:
        return df

    indexers = _select_index([*column_names], df, axis="columns")

    if len(indexers) < 2:
        raise ValueError(
            "The number of columns to coalesce should be a minimum of 2."
        )

    if target_column_name:
        check("target_column_name", target_column_name, [str])

    if default_value:
        check("default_value", default_value, [int, float, str])

    df = df.copy()

    outcome = df.iloc[:, indexers[0]]

    for num in range(1, len(indexers)):
        position = indexers[num]
        replacement = df.iloc[:, position]
        outcome = outcome.fillna(replacement)

    if outcome.hasnans and (default_value is not None):
        outcome = outcome.fillna(default_value)

    if target_column_name is None:
        df.iloc[:, indexers[0]] = outcome
    else:
        df[target_column_name] = outcome

    return df
