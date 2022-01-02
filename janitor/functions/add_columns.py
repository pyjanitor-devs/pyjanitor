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
    :raises ValueError: if attempting to add a column that already exists.
    :raises ValueError: if `value` has more elements that number of
        rows in the DataFrame.
    :raises ValueError: if attempting to add an iterable of values with
        a length not equal to the number of DataFrame rows.
    :raises ValueError: if `value` has length of `0``.
    """
    # TODO: Convert examples to notebook.
    # :Setup:

    # ```python

    #     import pandas as pd
    #     import janitor
    #     data = {
    #         "a": [1, 2, 3] * 3,
    #         "Bell__Chart": [1, 2, 3] * 3,
    #         "decorated-elephant": [1, 2, 3] * 3,
    #         "animals": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    #     }
    #     df = pd.DataFrame(data)

    # :Example 1: Create a new column with a single value:

    # ```python

    #     df.add_column("city_pop", 100000)

    # :Output:

    # ```python

    #        a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    #     0  1            1                   1   rabbit  Cambridge    100000
    #     1  2            2                   2  leopard   Shanghai    100000
    #     2  3            3                   3     lion      Basel    100000
    #     3  1            1                   1   rabbit  Cambridge    100000
    #     4  2            2                   2  leopard   Shanghai    100000
    #     5  3            3                   3     lion      Basel    100000
    #     6  1            1                   1   rabbit  Cambridge    100000
    #     7  2            2                   2  leopard   Shanghai    100000
    #     8  3            3                   3     lion      Basel    100000

    # :Example 2: Create a new column with an iterator which fills to the
    # column
    # size:

    # ```python

    #     df.add_column("city_pop", range(3), fill_remaining=True)

    # :Output:

    # ```python

    #        a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    #     0  1            1                   1   rabbit  Cambridge         0
    #     1  2            2                   2  leopard   Shanghai         1
    #     2  3            3                   3     lion      Basel         2
    #     3  1            1                   1   rabbit  Cambridge         0
    #     4  2            2                   2  leopard   Shanghai         1
    #     5  3            3                   3     lion      Basel         2
    #     6  1            1                   1   rabbit  Cambridge         0
    #     7  2            2                   2  leopard   Shanghai         1
    #     8  3            3                   3     lion      Basel         2

    # :Example 3: Add new column based on mutation of other columns:

    # ```python

    #     df.add_column("city_pop", df.Bell__Chart - 2 * df.a)

    # :Output:

    # ```python

    #        a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    #     0  1            1                   1   rabbit  Cambridge        -1
    #     1  2            2                   2  leopard   Shanghai        -2
    #     2  3            3                   3     lion      Basel        -3
    #     3  1            1                   1   rabbit  Cambridge        -1
    #     4  2            2                   2  leopard   Shanghai        -2
    #     5  3            3                   3     lion      Basel        -3
    #     6  1            1                   1   rabbit  Cambridge        -1
    #     7  2            2                   2  leopard   Shanghai        -2
    #     8  3            3                   3     lion      Basel        -3

    df = df.copy()
    check("column_name", column_name, [str])

    if column_name in df.columns:
        raise ValueError(
            f"Attempted to add column that already exists: " f"{column_name}."
        )

    nrows = df.shape[0]

    if hasattr(value, "__len__") and not isinstance(
        value, (str, bytes, bytearray)
    ):
        # if `value` is a list, ndarray, etc.
        if len(value) > nrows:
            raise ValueError(
                "`value` has more elements than number of rows "
                f"in your `DataFrame`. vals: {len(value)}, "
                f"df: {nrows}"
            )
        if len(value) != nrows and not fill_remaining:
            raise ValueError(
                "Attempted to add iterable of values with length"
                " not equal to number of DataFrame rows"
            )

        if len(value) == 0:
            raise ValueError(
                "`value` has to be an iterable of minimum length 1"
            )
        len_value = len(value)
    elif fill_remaining:
        # relevant if a scalar val was passed, yet fill_remaining == True
        len_value = 1
        value = [value]

    nrows = df.shape[0]

    if fill_remaining:
        times_to_loop = int(np.ceil(nrows / len_value))

        fill_values = list(value) * times_to_loop

        df[column_name] = fill_values[:nrows]
    else:
        df[column_name] = value

    return df


@pf.register_dataframe_method
def add_columns(
    df: pd.DataFrame, fill_remaining: bool = False, **kwargs
) -> pd.DataFrame:
    """Add multiple columns to the dataframe.

    This method does not mutate the original DataFrame.

    Method to augment `add_column` with ability to add multiple columns in
    one go. This replaces the need for multiple `add_column` calls.

    Usage is through supplying kwargs where the key is the col name and the
    values correspond to the values of the new DataFrame column.

    Values passed can be scalar or iterable (list, ndarray, etc.)

    Usage example:



        x = 3
        y = np.arange(0, 10)
        df = pd.DataFrame(...).add_columns(x=x, y=y)

    :param df: A pandas dataframe.
    :param fill_remaining: If value is a tuple or list that is smaller than
        the number of rows in the DataFrame, repeat the list or tuple
        (R-style) to the end of the DataFrame. (Passed to `add_column`)
    :param kwargs: column, value pairs which are looped through in
        `add_column` calls.
    :returns: A pandas DataFrame with added columns.
    """
    # Note: error checking can pretty much be handled in `add_column`

    for col_name, values in kwargs.items():
        df = df.add_column(col_name, values, fill_remaining=fill_remaining)

    return df
