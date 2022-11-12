import pandas_flavor as pf
import pandas as pd
from janitor.utils import deprecated_alias
from janitor.functions.utils import _select, DropLabel  # noqa: F401


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

    Selection on a MultiIndex on a level, or multiple levels,
    is possible with a dictionary.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of columns available as well.

    !!!note

        The preferred option when selecting columns or rows in a Pandas DataFrame
        is with `.loc` or `.iloc` methods, as they are generally performant.
        `select_columns` is primarily for convenience.

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
        a shell-style glob string (e.g. `*_thing_*`),
        a regular expression,
        a callable,
        or variable arguments of all the aforementioned.
        A sequence of booleans is also acceptable.
        A dictionary can be used for selection on a MultiIndex on different levels.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the columns
        provided.
    :returns: A pandas DataFrame with the specified columns selected.
    """  # noqa: E501

    return _select(df, args=args, invert=invert, axis="columns")


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

    Selection on a MultiIndex on a level, or multiple levels,
    is possible with a dictionary.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of rows available as well.


    !!! info "New in version 0.24.0"


    !!!note

        The preferred option when selecting columns or rows in a Pandas DataFrame
        is with `.loc` or `.iloc` methods, as they are generally performant.
        `select_rows` is primarily for convenience.


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
        a shell-style glob string (e.g. `*_thing_*`),
        a regular expression,
        a callable,
        or variable arguments of all the aforementioned.
        A sequence of booleans is also acceptable.
        A dictionary can be used for selection on a MultiIndex on different levels.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the rows
        provided.
    :returns: A pandas DataFrame with the specified rows selected.
    """  # noqa: E501
    return _select(df, args=args, invert=invert, axis="index")


@pf.register_dataframe_method
def select(df: pd.DataFrame, *, rows=None, columns=None) -> pd.DataFrame:
    """
    Method-chainable selection of rows and columns.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    Selection on a MultiIndex on a level, or multiple levels,
    is possible with a dictionary.

    This method does not mutate the original DataFrame.

    Selection can be inverted with the `DropLabel` class.


    !!! info "New in version 0.24.0"


    !!!note

        The preferred option when selecting columns or rows in a Pandas DataFrame
        is with `.loc` or `.iloc` methods, as they are generally performant.
        `select` is primarily for convenience.


    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=['cobra', 'viper', 'sidewinder'],
        ...      columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8
        >>> df.select(rows='cobra', columns='shield')
               shield
        cobra       2

    Labels can be dropped with the `DropLabel` class:

        >>> df.select(rows=DropLabel('cobra'))
                    max_speed  shield
        viper               4       5
        sidewinder          7       8

    :param df: A pandas DataFrame.
    :param rows: Valid inputs include: an exact label to look for,
        a shell-style glob string (e.g. `*_thing_*`),
        a regular expression,
        a callable,
        or variable arguments of all the aforementioned.
        A sequence of booleans is also acceptable.
        A dictionary can be used for selection on a MultiIndex on different levels.
    :param columns: Valid inputs include: an exact label to look for,
        a shell-style glob string (e.g. `*_thing_*`),
        a regular expression,
        a callable,
        or variable arguments of all the aforementioned.
        A sequence of booleans is also acceptable.
        A dictionary can be used for selection on a MultiIndex on different levels.
    :returns: A pandas DataFrame with the specified rows and/or columns selected.
    """  # noqa: E501

    return _select(df, args=None, rows=rows, columns=columns, axis="both")
