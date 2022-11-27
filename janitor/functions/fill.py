from collections.abc import Iterable as abcIterable
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


    Example:

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

    check_column(df, kwargs)

    fill_types = "up", "down", "updown", "downup"
    new_values = {}
    for column_name, fill_type in kwargs.items():
        check("fill_type", fill_type, [str])
        if fill_type not in fill_types:
            raise ValueError(
                f"The direction for {column_name} "
                "should be one of "
                "up, down, updown, or downup."
            )
        if fill_type == "up":
            output = df[column_name].bfill()
        elif fill_type == "down":
            output = df[column_name].ffill()
        elif fill_type == "updown":
            output = df[column_name].bfill().ffill()
        else:
            output = df[column_name].ffill().bfill()
        new_values[column_name] = output

    return df.assign(**new_values)


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def fill_empty(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable], value
) -> pd.DataFrame:
    """
    Fill `NaN` values in specified columns with a given value.

    Super sugary syntax that wraps `pandas.DataFrame.fillna`.

    This method mutates the original DataFrame.

    Example:

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
