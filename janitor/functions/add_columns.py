import pandas_flavor as pf

from janitor.utils import check, deprecated_alias
import pandas as pd
from typing import Union, List, Any, Tuple
import numpy as np


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name")
def add_column(
    df: pd.DataFrame,
    column_name: str,
    value: Union[List[Any], Tuple[Any], Any],
    fill_remaining: bool = False,
) -> pd.DataFrame:
    """Add a column to the dataframe.

    Intended to be the method-chaining alternative to:

    ```python
    df[column_name] = value
    ```

    Example: Add a column of constant values to the dataframe.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("abc")})
        >>> df.add_column(column_name="c", value=1)
           a  b  c
        0  0  a  1
        1  1  b  1
        2  2  c  1

    Example: Add a column of different values to the dataframe.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("abc")})
        >>> df.add_column(column_name="c", value=list("efg"))
           a  b  c
        0  0  a  e
        1  1  b  f
        2  2  c  g

    Example: Add a column using an iterator.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("abc")})
        >>> df.add_column(column_name="c", value=range(4, 7))
           a  b  c
        0  0  a  4
        1  1  b  5
        2  2  c  6

    :param df: A pandas DataFrame.
    :param column_name: Name of the new column. Should be a string, in order
        for the column name to be compatible with the Feather binary
        format (this is a useful thing to have).
    :param value: Either a single value, or a list/tuple of values.
    :param fill_remaining: If value is a tuple or list that is smaller than
        the number of rows in the DataFrame, repeat the list or tuple
        (R-style) to the end of the DataFrame.
    :returns: A pandas DataFrame with an added column.
    :raises ValueError: If attempting to add a column that already exists.
    :raises ValueError: If `value` has more elements that number of
        rows in the DataFrame.
    :raises ValueError: If attempting to add an iterable of values with
        a length not equal to the number of DataFrame rows.
    :raises ValueError: If `value` has length of `0`.
    """
    check("column_name", column_name, [str])

    if column_name in df.columns:
        raise ValueError(
            f"Attempted to add column that already exists: " f"{column_name}."
        )

    nrows = len(df)

    if hasattr(value, "__len__") and not isinstance(
        value, (str, bytes, bytearray)
    ):
        len_value = len(value)

        # if `value` is a list, ndarray, etc.
        if len_value > nrows:
            raise ValueError(
                "`value` has more elements than number of rows "
                f"in your `DataFrame`. vals: {len_value}, "
                f"df: {nrows}"
            )
        if len_value != nrows and not fill_remaining:
            raise ValueError(
                "Attempted to add iterable of values with length"
                " not equal to number of DataFrame rows"
            )
        if not len_value:
            raise ValueError(
                "`value` has to be an iterable of minimum length 1"
            )

    elif fill_remaining:
        # relevant if a scalar val was passed, yet fill_remaining == True
        len_value = 1
        value = [value]

    df = df.copy()
    if fill_remaining:
        times_to_loop = int(np.ceil(nrows / len_value))
        fill_values = list(value) * times_to_loop
        df[column_name] = fill_values[:nrows]
    else:
        df[column_name] = value

    return df


@pf.register_dataframe_method
def add_columns(
    df: pd.DataFrame,
    fill_remaining: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Add multiple columns to the dataframe.

    This method does not mutate the original DataFrame.

    Method to augment
    [`add_column`][janitor.functions.add_columns.add_column]
    with ability to add multiple columns in
    one go. This replaces the need for multiple
    [`add_column`][janitor.functions.add_columns.add_column] calls.

    Usage is through supplying kwargs where the key is the col name and the
    values correspond to the values of the new DataFrame column.

    Values passed can be scalar or iterable (list, ndarray, etc.)

    Example: Inserting two more columns into a dataframe.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": list(range(3)), "b": list("abc")})
        >>> df.add_columns(x=4, y=list("def"))
           a  b  x  y
        0  0  a  4  d
        1  1  b  4  e
        2  2  c  4  f

    :param df: A pandas dataframe.
    :param fill_remaining: If value is a tuple or list that is smaller than
        the number of rows in the DataFrame, repeat the list or tuple
        (R-style) to the end of the DataFrame. (Passed to `add_column`)
    :param **kwargs: Column, value pairs which are looped through in
        `add_column` calls.
    :returns: A pandas DataFrame with added columns.
    """
    # Note: error checking can pretty much be handled in `add_column`

    for col_name, values in kwargs.items():
        df = df.add_column(col_name, values, fill_remaining=fill_remaining)

    return df
