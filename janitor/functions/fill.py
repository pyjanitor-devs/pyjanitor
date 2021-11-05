from enum import Enum
from operator import methodcaller
from typing import Hashable, Iterable, Union

import pandas as pd
import pandas_flavor as pf
from janitor.utils import check, check_column, deprecated_alias
from multipledispatch import dispatch


@pf.register_dataframe_method
def fill_direction(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Provide a method-chainable function for filling missing values
    in selected columns.

    It is a wrapper for `pd.Series.ffill` and `pd.Series.bfill`,
    and pairs the column name with one of `up`, `down`, `updown`,
    and `downup`.


    Functional usage syntax:

    ```python
    import pandas as pd
    import janitor as jn

    df = pd.DataFrame(...)
    df = jn.fill_direction(
                df = df,
                column_1 = direction_1,
                column_2 = direction_2,
            )
    ```

    Method-chaining usage syntax:

    ```python
    import pandas as pd
    import janitor as jn

    df = pd.DataFrame(...)
            .fill_direction(
                column_1 = direction_1,
                column_2 = direction_2,
            )
    ```

    :param df: A pandas DataFrame.
    :param kwargs: Key - value pairs of columns and directions.
        Directions can be either `down`, `up`, `updown`
        (fill up then down) and `downup` (fill down then up).
    :returns: A pandas DataFrame with modified column(s).
    :raises ValueError: if direction supplied is not one of `down`, `up`,
        `updown`, or `downup`.
    """

    if not kwargs:
        return df

    fill_types = {fill.name for fill in _FILLTYPE}
    for column_name, fill_type in kwargs.items():
        check("column_name", column_name, [str])
        check("fill_type", fill_type, [str])
        if fill_type.upper() not in fill_types:
            raise ValueError(
                """
                fill_type should be one of
                up, down, updown, or downup.
                """
            )

    check_column(df, kwargs)

    new_values = {}
    for column_name, fill_type in kwargs.items():
        direction = _FILLTYPE[f"{fill_type.upper()}"].value
        if len(direction) == 1:
            direction = methodcaller(direction[0])
            output = direction(df[column_name])
        else:
            direction = [methodcaller(entry) for entry in direction]
            output = _chain_func(df[column_name], *direction)
        new_values[column_name] = output

    return df.assign(**new_values)


def _chain_func(column: pd.Series, *funcs):
    """
    Apply series of functions consecutively
    to a Series.
    https://blog.finxter.com/how-to-chain-multiple-function-calls-in-python/
    """
    new_value = column.copy()
    for func in funcs:
        new_value = func(new_value)
    return new_value


class _FILLTYPE(Enum):
    """List of fill types for fill_direction."""

    UP = ("bfill",)
    DOWN = ("ffill",)
    UPDOWN = "bfill", "ffill"
    DOWNUP = "ffill", "bfill"


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def fill_empty(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable], value
) -> pd.DataFrame:
    """
    Fill `NaN` values in specified columns with a given value.

    Super sugary syntax that wraps `pandas.DataFrame.fillna`.

    This method mutates the original DataFrame.

    Functional usage syntax:

    ```python
        df = fill_empty(df, column_names=[col1, col2], value=0)
    ```

    Method chaining syntax:

    ```python
        import pandas as pd
        import janitor
        df = pd.DataFrame(...).fill_empty(column_names=col1, value=0)
    ```

    :param df: A pandas DataFrame.
    :param column_names: column_names: A column name or an iterable (list
        or tuple) of column names. If a single column name is passed in, then
        only that column will be filled; if a list or tuple is passed in, then
        those columns will all be filled with the same value.
    :param value: The value that replaces the `NaN` values.
    :returns: A pandas DataFrame with `NaN` values filled.
    """
    check_column(df, column_names)
    return _fill_empty(df, column_names, value=value)


@dispatch(pd.DataFrame, (list, tuple))
def _fill_empty(df, column_names, value=None):
    """Fill empty function for the case that column_names is list or tuple."""
    fill_mapping = {c: value for c in column_names}
    return df.fillna(value=fill_mapping)


@dispatch(pd.DataFrame, str)  # noqa: F811
def _fill_empty(df, column_names, value=None):  # noqa: F811
    """Fill empty function for the case that column_names is a string."""
    fill_mapping = {column_names: value}
    return df.fillna(value=fill_mapping)
