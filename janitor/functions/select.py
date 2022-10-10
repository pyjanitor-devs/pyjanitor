import pandas_flavor as pf
import pandas as pd
import numpy as np
from pandas.api.types import is_list_like
from janitor.utils import deprecated_alias
from janitor.functions.utils import (
    _select_columns,
    _select_rows,
    _level_labels,
    level_labels,
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
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the columns
        provided.
    :raises ValueError: if `levels_label` is combined with other selection
        options.
    :returns: A pandas DataFrame with the specified columns selected.
    """  # noqa: E501

    all_levels = all((isinstance(arg, level_labels) for arg in args))
    if all_levels:
        contents = [
            _level_labels(df.columns, arg.label, arg.level) for arg in args
        ]
        if len(contents) > 1:
            contents = np.concatenate(contents)
            # remove possible duplicates
            contents = pd.unique(contents)
        else:
            contents = contents[0]
        if invert:
            arr = np.ones(df.columns.size, dtype=np.bool8)
            arr[contents] = False
            return df.iloc[arr]
        return df.iloc[contents]
    any_levels = any((isinstance(arg, level_labels) for arg in args))
    if any_levels:
        raise ValueError(
            "`level_labels` cannot be combined with other selection options."
        )
    # applicable to any list-like object (ndarray, Series, pd.Index, ...)
    search_result = []
    for arg in args:
        if is_list_like(arg) and (not isinstance(arg, tuple)):
            search_result.extend(arg)
        else:
            search_result.append(arg)
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

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of rows available as well.

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
    :param args: Valid inputs include: an exact index name to look for,
        a shell-style glob string (e.g., `*_thing_*`),
        a regular expression,
        a callable which is applicable to each Series in the DataFrame,
        or variable arguments of all the aforementioned.
        A sequence of booleans is also acceptable.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the rows
        provided.
    :raises ValueError: if `levels_label` is combined with other selection
        options.
    :returns: A pandas DataFrame with the specified rows selected.
    """  # noqa: E501

    all_levels = all((isinstance(arg, level_labels) for arg in args))
    if all_levels:
        contents = [
            _level_labels(df.index, arg.label, arg.level) for arg in args
        ]
        if len(contents) > 1:
            contents = np.concatenate(contents)
            # remove possible duplicates
            contents = pd.unique(contents)
        else:
            contents = contents[0]
        if invert:
            arr = np.ones(df.index.size, dtype=np.bool8)
            arr[contents] = False
            return df.iloc[arr]
        return df.iloc[contents]
    any_levels = any((isinstance(arg, level_labels) for arg in args))
    if any_levels:
        raise ValueError(
            "`level_labels` cannot be combined with other selection options."
        )
    # applicable to any list-like object (ndarray, Series, pd.Index, ...)
    search_result = []
    for arg in args:
        if is_list_like(arg) and (not isinstance(arg, tuple)):
            search_result.extend(arg)
        else:
            search_result.append(arg)
    search_result = _select_rows(search_result, df)
    if invert:
        return df.drop(index=search_result)
    return df.loc[search_result]
