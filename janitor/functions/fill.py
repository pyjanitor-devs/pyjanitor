from collections.abc import Iterable as abcIterable
from enum import Enum
from operator import methodcaller
from typing import Any, Hashable, Iterable, Union

import pandas as pd
import pandas_flavor as pf
from multipledispatch import dispatch

from janitor.utils import (
    check,
    check_column,
    deprecated_alias,
    refactored_function,
)


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.assign` instead."
    )
)
def fill_direction(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Provide a method-chainable function for filling missing values
    in selected columns.

    It is a wrapper for `pd.Series.ffill` and `pd.Series.bfill`,
    and pairs the column name with one of `up`, `down`, `updown`,
    and `downup`.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.assign` instead.

    Examples:
        >>> import pandas as pd
        >>> import janitor as jn
        >>> df = pd.DataFrame(
        ...    {
        ...        'col1': [1, 2, 3, 4],
        ...        'col2': [None, 5, 6, 7],
        ...        'col3': [8, 9, 10, None],
        ...        'col4': [None, None, 11, None],
        ...        'col5': [None, 12, 13, None]
        ...    }
        ... )
        >>> df
           col1  col2  col3  col4  col5
        0     1   NaN   8.0   NaN   NaN
        1     2   5.0   9.0   NaN  12.0
        2     3   6.0  10.0  11.0  13.0
        3     4   7.0   NaN   NaN   NaN
        >>> df.fill_direction(
        ... col2 = 'up',
        ... col3 = 'down',
        ... col4 = 'downup',
        ... col5 = 'updown'
        ... )
           col1  col2  col3  col4  col5
        0     1   5.0   8.0  11.0  12.0
        1     2   5.0   9.0  11.0  12.0
        2     3   6.0  10.0  11.0  13.0
        3     4   7.0  10.0  11.0  13.0

    Args:
        df: A pandas DataFrame.
        **kwargs: Key - value pairs of columns and directions.
            Directions can be either `down`, `up`, `updown`
            (fill up then down) and `downup` (fill down then up).

    Raises:
        ValueError: If direction supplied is not one of `down`, `up`,
            `updown`, or `downup`.

    Returns:
        A pandas DataFrame with modified column(s).
    """  # noqa: E501

    if not kwargs:
        return df

    fill_types = {fill.name for fill in _FILLTYPE}
    for column_name, fill_type in kwargs.items():
        check("column_name", column_name, [str])
        check("fill_type", fill_type, [str])
        if fill_type.upper() not in fill_types:
            raise ValueError(
                "fill_type should be one of up, down, updown, or downup."
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
@refactored_function(
    message="This function will be deprecated in a 1.x release. "
    "Kindly use `jn.impute` instead."
)
@deprecated_alias(columns="column_names")
def fill_empty(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable],
    value: Any,
) -> pd.DataFrame:
    """Fill `NaN` values in specified columns with a given value.

    Super sugary syntax that wraps `pandas.DataFrame.fillna`.

    This method mutates the original DataFrame.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use [`jn.impute`][janitor.functions.impute.impute] instead.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...        {
        ...            'col1': [1, 2, 3],
        ...            'col2': [None, 4, None ],
        ...            'col3': [None, 5, 6]
        ...        }
        ...    )
        >>> df
           col1  col2  col3
        0     1   NaN   NaN
        1     2   4.0   5.0
        2     3   NaN   6.0
        >>> df.fill_empty(column_names = 'col2', value = 0)
           col1  col2  col3
        0     1   0.0   NaN
        1     2   4.0   5.0
        2     3   0.0   6.0
        >>> df.fill_empty(column_names = ['col2', 'col3'], value = 0)
           col1  col2  col3
        0     1   0.0   0.0
        1     2   4.0   5.0
        2     3   0.0   6.0

    Args:
        df: A pandas DataFrame.
        column_names: A column name or an iterable (list
            or tuple) of column names. If a single column name is passed in,
            then only that column will be filled; if a list or tuple is passed
            in, then those columns will all be filled with the same value.
        value: The value that replaces the `NaN` values.

    Returns:
        A pandas DataFrame with `NaN` values filled.
    """

    check_column(df, column_names)
    return _fill_empty(df, column_names, value=value)


@dispatch(pd.DataFrame, abcIterable)
def _fill_empty(df, column_names, value=None):
    """Fill empty function for the case that column_names is list or tuple."""
    fill_mapping = {c: value for c in column_names}
    return df.fillna(value=fill_mapping)


@dispatch(pd.DataFrame, str)  # noqa: F811
def _fill_empty(df, column_names, value=None):  # noqa: F811
    """Fill empty function for the case that column_names is a string."""
    fill_mapping = {column_names: value}
    return df.fillna(value=fill_mapping)
