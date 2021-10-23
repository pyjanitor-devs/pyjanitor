from typing import Hashable, Iterable, Union
from multipledispatch import dispatch
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check_column, deprecated_alias


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
