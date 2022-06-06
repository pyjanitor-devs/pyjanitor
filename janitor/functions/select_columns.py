"""Implementation of select_columns"""
from typing import Optional, Union
import pandas_flavor as pf
import pandas as pd
from pandas.api.types import is_list_like
from janitor.utils import deprecated_alias, check

from janitor.functions.utils import _select_column_names


@pf.register_dataframe_method
@deprecated_alias(search_cols="search_column_names")
def select_columns(
    df: pd.DataFrame,
    *args,
    level: Optional[Union[int, str]] = None,
    invert: bool = False,
) -> pd.DataFrame:
    """
    Method-chainable selection of columns.

    Not applicable to MultiIndex columns.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of columns available as well.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"col1": [1, 2], "foo": [3, 4], "col2": [5, 6]})
        >>> df
           col1  foo  col2
        0     1    3     5
        1     2    4     6
        >>> df.select_columns("col*")
           col1  col2
        0     1     5
        1     2     6

    :param df: A pandas DataFrame.
    :param args: Valid inputs include: an exact column name to look for,
        a shell-style glob string (e.g., `*_thing_*`),
        a regular expression,
        a callable which is applicable to each Series in the DataFrame,
        or variable arguments of all the aforementioned.
        A sequence of booleans is also acceptable.
    :param level: Determines which level in the columns should be used for the
        column selection.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the columns
        provided.
    :returns: A pandas DataFrame with the specified columns selected.
    """  # noqa: E501

    # applicable for any
    # list-like object (ndarray, Series, pd.Index, ...)
    search_column_names = []
    for arg in args:
        if is_list_like(arg) and (not isinstance(arg, tuple)):
            search_column_names.extend(arg)
        else:
            search_column_names.append(arg)
    if level is not None:
        # goal here is to capture the original columns
        # trim the df.columns to the specified level only,
        # and apply the selection (_select_column_names)
        # to get the relevant column labels.
        # note that no level is dropped; if there are three levels,
        # then three levels are returned, with the specified labels
        # selected/deselected.
        # A copy of the dataframe is made via set_axis
        df_columns = df.columns
        check("level", level, [int, str])
        full_column_list = df_columns.get_level_values(level)
        full_column_list = _select_column_names(
            search_column_names, df.set_axis(full_column_list, axis=1)
        )
        full_column_list = df_columns.isin(full_column_list, level=level)
        full_column_list = df_columns[full_column_list]
    else:
        full_column_list = _select_column_names(search_column_names, df)
    if invert:
        return df.drop(columns=full_column_list)
    return df.loc[:, full_column_list]
