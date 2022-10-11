import pandas_flavor as pf
import pandas as pd
from pandas.api.types import is_list_like
from janitor.utils import deprecated_alias
from janitor.functions.utils import (
    IndexLabel,
    _select_columns,
    _select_rows,
    _select_index_labels,
)


@pf.register_dataframe_method
@deprecated_alias(search_cols="search_column_names")
def select_columns(
    df: pd.DataFrame,
    *args,
    invert: bool = False,
) -> pd.DataFrame:
    """
    Method-chainable selection of columns.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    Selection on a MultiIndex is possible with the
    [`IndexLabel`][janitor.functions.select.IndexLabel] class.

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
        For selection on a MultiIndex,
        [`IndexLabel`][janitor.functions.select.IndexLabel] can be handy.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the columns
        provided.
    :returns: A pandas DataFrame with the specified columns selected.
    """  # noqa: E501

    # applicable to any list-like object (ndarray, Series, pd.Index, ...)
    search_result = []
    for arg in args:
        if is_list_like(arg) and (not isinstance(arg, tuple)):
            search_result.extend(arg)
        else:
            search_result.append(arg)
    index_labels = (isinstance(arg, IndexLabel) for arg in search_result)
    if all(index_labels):
        return _select_index_labels(df, search_result, "columns", invert)
    search_result = _select_columns(search_result, df)
    if invert:
        return df.drop(columns=search_result)
    return df.loc[:, search_result]


@pf.register_dataframe_method
def select_rows(
    df: pd.DataFrame,
    *args,
    invert: bool = False,
) -> pd.DataFrame:
    """
    Method-chainable selection of rows.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    Selection on a MultiIndex is possible with the
    [`IndexLabel`][janitor.functions.select.IndexLabel] class.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of rows available as well.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = {"col1": [1, 2], "foo": [3, 4], "col2": [5, 6]}
        >>> df = pd.DataFrame.from_dict(df, orient='index')
        >>> df
              0  1
        col1  1  2
        foo   3  4
        col2  5  6
        >>> df.select_rows("col*")
              0  1
        col1  1  2
        col2  5  6

    :param df: A pandas DataFrame.
    :param args: Valid inputs include: an exact index name to look for,
        a shell-style glob string (e.g., `*_thing_*`),
        a regular expression,
        a callable which is applicable to the DataFrame,
        or variable arguments of all the aforementioned.
        A sequence of booleans is also acceptable.
        For selection on a MultiIndex,
        [`IndexLabel`][janitor.functions.select.IndexLabel] can be handy.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the rows
        provided.
    :returns: A pandas DataFrame with the specified rows selected.
    """  # noqa: E501

    # applicable to any list-like object (ndarray, Series, pd.Index, ...)
    search_result = []
    for arg in args:
        if is_list_like(arg) and (not isinstance(arg, tuple)):
            search_result.extend(arg)
        else:
            search_result.append(arg)
    index_labels = (isinstance(arg, IndexLabel) for arg in search_result)
    if all(index_labels):
        return _select_index_labels(df, search_result, "index", invert)

    search_result = _select_rows(search_result, df)
    if invert:
        return df.drop(index=search_result)
    return df.loc[search_result]