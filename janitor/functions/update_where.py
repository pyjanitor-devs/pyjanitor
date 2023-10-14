"""Function for updating values based on other column values."""

from typing import Any, Hashable

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_bool_dtype

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(target_col="target_column_name")
def update_where(
    df: pd.DataFrame,
    conditions: Any,
    target_column_name: Hashable,
    target_val: Any,
) -> pd.DataFrame:
    """Add multiple conditions to update a column in the dataframe.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import janitor
        >>> data = {
        ...    "a": [1, 2, 3, 4],
        ...    "b": [5, 6, 7, 8],
        ...    "c": [0, 0, 0, 0],
        ... }
        >>> df = pd.DataFrame(data)
        >>> df
           a  b  c
        0  1  5  0
        1  2  6  0
        2  3  7  0
        3  4  8  0
        >>> df.update_where(
        ...    conditions = (df.a > 2) & (df.b < 8),
        ...    target_column_name = 'c',
        ...    target_val = 10
        ... )
           a  b   c
        0  1  5   0
        1  2  6   0
        2  3  7  10
        3  4  8   0
        >>> df.update_where( # supports pandas *query* style string expressions
        ...    conditions = "a > 2 and b < 8",
        ...    target_column_name = 'c',
        ...    target_val = 10
        ... )
           a  b   c
        0  1  5   0
        1  2  6   0
        2  3  7  10
        3  4  8   0

    Args:
        df: The pandas DataFrame object.
        conditions: Conditions used to update a target column
            and target value.
        target_column_name: Column to be updated. If column does not exist
            in DataFrame, a new column will be created; note that entries
            that do not get set in the new column will be null.
        target_val: Value to be updated.

    Raises:
        ValueError: If `conditions` does not return a boolean array-like
            data structure.

    Returns:
        A pandas DataFrame.
    """

    df = df.copy()

    # use query mode if a string expression is passed
    if isinstance(conditions, str):
        conditions = df.eval(conditions)

    if not is_bool_dtype(conditions):
        raise ValueError(
            """
            Kindly ensure that `conditions` passed
            evaluates to a Boolean dtype.
            """
        )

    df.loc[conditions, target_column_name] = target_val

    return df
