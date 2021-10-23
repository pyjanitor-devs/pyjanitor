from typing import Callable, List, Optional, Pattern, Tuple, Union
import pandas_flavor as pf
import pandas as pd

from pandas.api.types import is_list_like, is_string_dtype

from janitor.utils import check
from janitor.functions.utils import _select_column_names
import re
import numpy as np
from collections import defaultdict


@pf.register_dataframe_method
def pivot_longer(
    df: pd.DataFrame,
    index: Optional[Union[List, Tuple, str, Pattern]] = None,
    column_names: Optional[Union[List, Tuple, str, Pattern]] = None,
    names_to: Optional[Union[List, Tuple, str]] = None,
    values_to: Optional[str] = "value",
    column_level: Optional[Union[int, str]] = None,
    names_sep: Optional[Union[str, Pattern]] = None,
    names_pattern: Optional[Union[List, Tuple, str, Pattern]] = None,
    sort_by_appearance: Optional[bool] = False,
    ignore_index: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Unpivots a DataFrame from *wide* to *long* format.

    This method does not mutate the original DataFrame.

    It is a wrapper around `pd.melt` and is meant to serve as a single point
    for transformations that require `pd.melt` or `pd.wide_to_long`.

    It is modeled after the `pivot_longer` function in R's tidyr package, and
    offers more functionality and flexibility than `pd.wide_to_long`.

    This function is useful to massage a DataFrame into a format where
    one or more columns are considered measured variables, and all other
    columns are considered as identifier variables.

    All measured variables are *unpivoted* (and typically duplicated) along the
    row axis.


    Example 1: The following DataFrame contains heartrate data for patients
    treated with two different drugs, `a` and `b`.

    ```python
              name   a   b
        0   Wilbur  67  56
        1  Petunia  80  90
        2  Gregory  64  50
    ```

    The column names `a` and `b` are actually the names of a measured variable
    (i.e. the name of a drug), but the values are a different measured variable
    (heartrate). We would like to unpivot these `a` and `b` columns into a
    `drug` column and a `heartrate` column.

    ```python
        df.pivot_longer(
            column_names = ['a', 'b'],
            names_to = 'drug',
            values_to = 'heartrate',
            sort_by_appearance = True
        )


              name drug  heartrate
        0   Wilbur    a         67
        1   Wilbur    b         56
        2  Petunia    a         80
        3  Petunia    b         90
        4  Gregory    a         64
        5  Gregory    b         50
    ```

    Note how the data is stacked in order of first appearance. If, however,
    you do not care for order of appearance, and want to wring out some
    more performance, you can set `sort_by_appearance` to `False` (the
    default is `False`).

    ```python
        df.pivot_longer(
            column_names = ['a', 'b'],
            names_to = 'drug',
            values_to = 'heartrate',
            sort_by_appearance = False
        )


                name     drug  heartrate
        0	Wilbur	   a	67
        1	Petunia	   a	80
        2	Gregory    a	64
        3	Wilbur	   b	56
        4	Petunia	   b	90
        5	Gregory	   b	50
    ```

    You can set `ignore_index` to `False`, if you wish to reuse the index
    from the source DataFrame (the index will be repeated as many times as
    necessary):


    ```python
        df.pivot_longer(
            column_names = ['a', 'b'],
            names_to = 'drug',
            values_to = 'heartrate',
            sort_by_appearance = False,
            ignore_index = False
        )


                name     drug  heartrate
        0	Wilbur	   a	67
        1	Petunia	   a	80
        2	Gregory    a	64
        0	Wilbur	   b	56
        1	Petunia	   b	90
        2	Gregory	   b	50
    ```

    MultiIndex DataFrames are unpivoted in the same form that you would
    expect from pandas' `melt`:

    ```python
            A  B  C
            D  E  F
        0   a  1  2
        1   b  3  4
        2   c  5  6

        df.pivot_longer(
            index = [("A", "D")],
            names_to = ["first", "second"]
        )


             (A, D)  first   second   value
        0	a	B	E	1
        1	b	B	E	3
        2	c	B	E	5
        3	a	C	F	2
        4	b	C	F	4
        5	c	C	F	6
    ```

    You can also unpivot on a specific level:


    ```python
        df.pivot_longer(
            index = "A",
            names_to = "first",
            column_level = 0
        )


           A      first  value
        0  a        B      1
        1  b        B      3
        2  c        B      5
    ```

    Example 2: The DataFrame below has year and month variables embedded within
    the column names.


    ```python
              col1	    2019-12	 2020-01	 2020-02
        0	a	   -1.085631	-1.506295	-2.426679
        1	b	    0.997345	-0.578600	-0.428913
        2	c	    0.282978	 1.651437	 1.265936
    ```

    `pivot_longer` can conveniently reshape the DataFrame into long format,
    with new columns for the year and month. You simply pass in the new
    column names to `names_to`, and pass the hyphen `-` to the `names_sep`
    argument.


    ```python
        df.pivot_longer(
            index = 'col1',
            names_to = ('year', 'month'),
            names_sep = '-',
            sort_by_appearance = True
        )

           col1 year   month      value
        0    a  2019     12     -1.085631
        1    a  2020     01     -1.506295
        2    a  2020     02     -2.426679
        3    b  2019     12      0.997345
        4    b  2020     01     -0.578600
        5    b  2020     02     -0.428913
        6    c  2019     12      0.282978
        7    c  2020     01      1.651437
        8    c  2020     02      1.265936
    ```

    Example 3: The DataFrame below has names embedded in it
    `(measure1, measure2)` that we would love to reuse as column names.


    ```python
            treat1-measure1     treat1-measure2 treat2-measure1 treat2-measure2
        0                1              4                   2               5
        1                2              5                   3               4
    ```

    For this, we use the `.value` variable, which signals to `pivot_longer`
    to treat the part of the column names corresponding to `.value` as new
    column names. The `.value` variable is similar to `stubnames` in pandas'
    `wide_to_long` function, but with more flexibility.


    ```python
        df.pivot_longer(
            names_to = ("group", '.value'),
            names_sep = '-',
            sort_by_appearance = True
        )


            group  measure1  measure2
        0  treat1         1         4
        1  treat2         2         5
        2  treat1         2         5
        3  treat2         3         4
    ```

    Let's break down the `.value` idea. When `.value` is used, `pivot_longer`
    creates a pairing. In the example above, we get a pairing
    `{"group":["treat1", "treat2"], ".value":["measure1", "measure2"]}`. All
    the values associated with `.value` become new column names, while those
    not associated with `.value`(`treat1` and `treat2`) become values in a
    new column `group`. `values_to` is overridden during this process.

    !!! note
        The values not associated with `.value` (in the example above,
        this is the `group` column) are returned as object dtypes. You can
        change it to your preferred dtype using pandas' `astype` method.

    Example 4: You can also unpivot from wide to long using regular expressions


    ```python
            n_1  n_2  n_3  pct_1  pct_2  pct_3
        0   10   20   30   0.1    0.2    0.3

        df.pivot_longer(
            names_to = (".value", "name"),
            names_pattern = "(.*)_(.)"
         )


            name    n  pct
        0     1  10.0  0.1
        1     2  20.0  0.2
        2     3  30.0  0.3
    ```

    The same idea of `.value` works here as well. Based on the capturing groups
    in the regex in `names_pattern`, we have two pairings -->
    `{".value":["n", "pct"], "name":[1,2,3]}`. Just like in the previous
    example, the values associated with `.value` become new column names,
    while those not associated with `.value` become values in the new column
    `name`.

    !!!note
        There are no limits to the pairing; however, you can only have
        one `.value` in `names_to`.

    Example 5: You can also pass a list/tuple of regular expressions that match
    specific patterns to `names_pattern`, along with a list/tuple of new
    names to `names_to`; this can come in handy if `.value` falls short:

    ```python
          GameID   Date	        Visitor	     Score_V	   Home	        Score_H
        0  1     9/10/2020   Houston Texans     20    Kansas City Chiefs   34
        1  2     9/13/2020   Seattle Seahawks   38    Atlanta Falcons      25



        df.pivot_longer(
            index = ['GameID','Date'],
            names_to = ("Team","Score"),
            names_pattern = ("^Visitor|Home", "^Score")
            )

               GameID       Date              Team       Score
        0       1       9/10/2020      Houston Texans     20
        1       2       9/13/2020    Seattle Seahawks     38
        2       1       9/10/2020  Kansas City Chiefs     34
        3       2       9/13/2020     Atlanta Falcons     25
    ```

    Note that in the code above, the number of entries in both `names_to` and
    `names_pattern` must match. Essentially, what the code does is look for
    columns that start with `Visitor` or `Home` (using the regex supplied) and
    puts all the values associated with these columns under a new column name
    `Team`. It then looks for columns that start with `Score` and collate all
    the values associated with these columns to a single column named `Score`.

    You can also take advantage of `janitor.patterns` function,
    or the `select_columns` syntax, which allows selection of columns via a
    regular expression; this can come in handy if you have a lot of
    column names to pass to the `index` or `column_names`  parameters,
    and you do not wish to manually type them all.

    ```python
             name    wk1   wk2   wk3   wk4
        0    Alice     5     9    20    22
        1    Bob       7    11    17    33
        2    Carla     6    13    39    40

        df.pivot_longer(index = janitor.patterns("^(?!wk)"))


             name   variable  value
        0   Alice      wk1      5
        1     Bob      wk1      7
        2   Carla      wk1      6
        3   Alice      wk2      9
        4     Bob      wk2     11
        5   Carla      wk2     13
        6   Alice      wk3     20
        7     Bob      wk3     17
        8   Carla      wk3     39
        9   Alice      wk4     22
        10    Bob      wk4     33
        11  Carla      wk4     40
    ```

    !!!note
        Unpivoting a DataFrame with MultiIndex columns, when
        either `names_sep` or `names_pattern` is provided is not
        supported.


    Functional usage syntax:

    ```python
    import pandas as pd
    import janitor as jn

    df = pd.DataFrame(...)
    df = jn.pivot_longer(
        df = df,
        index = [column1, column2, ...],
        column_names = [column3, column4, ...],
        names_to = new_column_name,
        names_sep = string/regular expression,
        names_pattern = string/regular expression,
        values_to= new_column_name,
        column_level = None/int/str,
        sort_by_appearance = True/False,
        ignore_index = True/False,
    )
    ```

    Method chaining syntax:

    ```python
    df = (
        pd.DataFrame(...)
        .pivot_longer(
            index = [column1, column2, ...],
            column_names = [column3, column4, ...],
            names_to = new_column_name,
            names_sep = string/regular expression,
            names_pattern = string/regular expression,
            values_to = new_column_name,
            column_level = None/int/str,
            sort_by_appearance = True/False,
            ignore_index = True/False,
        )
    )
    ```

    :param df: A pandas DataFrame.
    :param index: Name(s) of columns to use as identifier variables.
        Should be either a single column name, or a list/tuple of
        column names. The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        Index should be a list of tuples if the columns are a MultiIndex.
    :param column_names: Name(s) of columns to unpivot. Should be either
        a single column name or a list/tuple of column names.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        Column_names should be a list of tuples
        if the columns are a MultiIndex.
    :param names_to: Name of new column as a string that will contain
        what were previously the column names in `column_names`.
        The default is `variable` if no value is provided. It can
        also be a list/tuple of strings that will serve as new column
        names, if `name_sep` or `names_pattern` is provided.
        If `.value` is in `names_to`, new column names will be extracted
        from part of the existing column names and overrides`values_to`.
    :param names_sep: Determines how the column name is broken up, if
        `names_to` contains multiple values. It takes the same
        specification as pandas' `str.split` method, and can be a string
        or regular expression. `names_sep` does not work with MultiIndex
        columns.
    :param names_pattern: Determines how the column name is broken up.
        It can be a regular expression containing matching groups (it takes
        the same specification as pandas' `str.extract` method), or a
        list/tuple of regular expressions. If it is a single regex, the
        number of groups must match the length of `names_to` ( if the
        length of `names_to` is 3, then the number of groups must be 3.
        If `names_to` is a string, then there should be only one group
        in `names_pattern``). For a list/tuple of regular expressions,
        `names_to` must also be a list/tuple and the lengths of both
        arguments must match(if the length of `names_to` is 4, then the
        length of `names_pattern` must also be 4). The entries in both
        arguments must also match positionally, i.e  if
        `names_to = ("name1", "name2", "name3")``, then `names_pattern``
        should be ("regex1", "regex2", "regex3"), with "name1" pairing
        "regex1", "name2" pairing "regex2", and "name3" pairing "regex3".
        `names_pattern` does not work with MultiIndex columns.
    :param values_to: Name of new column as a string that will contain what
        were previously the values of the columns in `column_names`.
    :param column_level: If columns are a MultiIndex, then use this level to
        unpivot the DataFrame. Provided for compatibility with pandas' melt,
        and applies only if neither `names_sep` nor `names_pattern` is
        provided.
    :param sort_by_appearance: Default `False`. Boolean value that determines
        the final look of the DataFrame. If `True`, the unpivoted DataFrame
        will be stacked in order of first appearance. See examples for more
        details. `pivot_longer` is usually more performant if
        `sort_by_appearance` is `False``.
    :param ignore_index: Default `True`. If True, original index is ignored.
        If False, the original index is retained and the Index labels will be
        repeated as necessary.
    :returns: A pandas DataFrame that has been unpivoted from wide to long
        format.
    """

    # this code builds on the wonderful work of @benjaminjack’s PR
    # https://github.com/benjaminjack/pyjanitor/commit/e3df817903c20dd21634461c8a92aec137963ed0

    df = df.copy()

    (
        df,
        index,
        column_names,
        names_to,
        values_to,
        column_level,
        names_sep,
        names_pattern,
        sort_by_appearance,
        ignore_index,
    ) = _data_checks_pivot_longer(
        df,
        index,
        column_names,
        names_to,
        values_to,
        column_level,
        names_sep,
        names_pattern,
        sort_by_appearance,
        ignore_index,
    )

    return _computations_pivot_longer(
        df,
        index,
        column_names,
        names_to,
        values_to,
        column_level,
        names_sep,
        names_pattern,
        sort_by_appearance,
        ignore_index,
    )


@pf.register_dataframe_method
def pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[List, str]] = None,
    names_from: Optional[Union[List, str]] = None,
    values_from: Optional[Union[List, str]] = None,
    levels_order: Optional[list] = None,
    flatten_levels: Optional[bool] = True,
    names_sep="_",
    names_glue: Callable = None,
) -> pd.DataFrame:
    """
    Reshapes data from 'long' to 'wide' form.

    The number of columns are increased, while decreasing
    the number of rows. It is the inverse of the `pivot_longer`
    method, and is a wrapper around `pd.DataFrame.pivot` method.

    This method does not mutate the original DataFrame.

    .. note:: Column selection in `index`, `names_from`
        and `values_from` is possible using the
        `janitor.select_columns` syntax.

    .. note:: A ValueError is raised if the combination
        of the `index` and `names_from` is not unique.

    .. note:: By default, values from `values_from` are always
        at the top level if the columns are not flattened.
        If flattened, the values from `values_from` are usually
        at the start of each label in the columns.

    Functional usage syntax:

    ```python
        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.pivot_wider(
            df = df,
            index = [column1, column2, ...],
            names_from = [column3, column4, ...],
            value_from = [column5, column6, ...],
            levels_order = None/list,
            flatten_levels = True/False,
            names_sep='_',
            names_glue= Callable

        )
    ```

    Method chaining syntax:

    ```python
        df = (
            pd.DataFrame(...)
            .pivot_wider(
                index = [column1, column2, ...],
                names_from = [column3, column4, ...],
                value_from = [column5, column6, ...],
                levels_order = None/list,
                flatten_levels = True/False,
                names_sep='_',
                names_glue= Callable
                )
        )
    ```

    :param df: A pandas dataframe.
    :param index: Name(s) of columns to use as identifier variables.
        Should be either a single column name, or a list of column names.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        If `index` is not provided, the dataframe's index is used.
    :param names_from: Name(s) of column(s) to use to make the new
        dataframe's columns. Should be either a single column name, or a
        list of column names.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        A label or labels must be provided for `names_from`.
    :param values_from: Name(s) of column(s) that will be used for populating
        the new dataframe's values.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        If ``values_from`` is not specified,  all remaining columns
        will be used. Note that values from `values_from` are usually at
        the top level, the dataframe's columns is not flattened, or the
        start of each label in the columns, if flattened.
    :param levels_order: Applicable if there are multiple `names_from`
        and/or `values_from`. Reorders the levels. Accepts a list of strings.
        If there are multiple `values_from`, pass a None to represent that
        level.
    :param flatten_levels: Default is `True`. If `False`, the dataframe stays
        as a MultiIndex.
    :param names_sep: If `names_from` or `values_from` contain multiple
        variables, this will be used to join their values into a single string
        to use as a column name. Default is `_`.
        Applicable only if `flatten_levels` is `True`.
    :param names_glue: A callable to control
        the output of the flattened columns.
        Applicable only if `flatten_levels` is True.
        Function should be acceptable to pandas’ `map` function.
    :returns: A pandas DataFrame that has been unpivoted from long to wide
        form.
    """

    df = df.copy()

    return _computations_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        levels_order,
        flatten_levels,
        names_sep,
        names_glue,
    )


def _data_checks_pivot_longer(
    df,
    index,
    column_names,
    names_to,
    values_to,
    column_level,
    names_sep,
    names_pattern,
    sort_by_appearance,
    ignore_index,
):

    """
    This function raises errors if the arguments have the wrong python type,
    or if an unneeded argument is provided. It also raises errors for some
    other scenarios(e.g if there are no matches returned for the regular
    expression in `names_pattern`, or if the dataframe has MultiIndex
    columns and `names_sep` or `names_pattern` is provided).

    This function is executed before proceeding to the computation phase.

    Type annotations are not provided because this function is where type
    checking happens.
    """

    if column_level is not None:
        check("column_level", column_level, [int, str])
        df.columns = df.columns.get_level_values(column_level)

    if index is not None:
        if is_list_like(index) and (not isinstance(index, tuple)):
            index = list(index)
        index = _select_column_names(index, df)

    if column_names is not None:
        if is_list_like(column_names) and (
            not isinstance(column_names, tuple)
        ):
            column_names = list(column_names)
        column_names = _select_column_names(column_names, df)

    len_names_to = 0
    if names_to is not None:
        if isinstance(names_to, str):
            names_to = [names_to]
        elif isinstance(names_to, tuple):
            names_to = list(names_to)
        check("names_to", names_to, [list])

        unique_names_to = set()
        for word in names_to:
            if not isinstance(word, str):
                raise TypeError(
                    f"""
                    All entries in the names_to
                    argument must be strings.
                    {word} is of type {type(word).__name__}
                    """
                )

            if word in unique_names_to:
                raise ValueError(
                    f"""
                    {word} already exists in names_to.
                    Duplicates are not allowed.
                    """
                )
            unique_names_to.add(word)  # noqa: PD005
        unique_names_to = None

        len_names_to = len(names_to)

    if names_sep and names_pattern:
        raise ValueError(
            """
                Only one of names_pattern or names_sep
                should be provided.
                """
        )

    if names_pattern is not None:
        check("names_pattern", names_pattern, [str, Pattern, list, tuple])
        if names_to is None:
            raise ValueError(
                """
                Kindly provide values for names_to.
                """
            )
        if isinstance(names_pattern, (str, Pattern)):
            num_regex_grps = re.compile(names_pattern).groups

            if len_names_to != num_regex_grps:
                raise ValueError(
                    f"""
                    The length of names_to does not match
                    the number of groups in names_pattern.
                    The length of names_to is {len_names_to}
                    while the number of groups in the regex
                    is {num_regex_grps}
                    """
                )

        if isinstance(names_pattern, (list, tuple)):
            for word in names_pattern:
                if not isinstance(word, (str, Pattern)):
                    raise TypeError(
                        f"""
                        All entries in the names_pattern argument
                        must be regular expressions.
                        `{word}` is of type {type(word).__name__}
                        """
                    )

            if len(names_pattern) != len_names_to:
                raise ValueError(
                    f"""
                    The length of names_to does not match
                    the number of regexes in names_pattern.
                    The length of names_to is {len_names_to}
                    while the number of regexes
                    is {len(names_pattern)}
                    """
                )

            if names_to and (".value" in names_to):
                raise ValueError(
                    """
                    `.value` is not accepted in names_to
                    if names_pattern is a list/tuple.
                    """
                )

    if names_sep is not None:
        check("names_sep", names_sep, [str, Pattern])
        if names_to is None:
            raise ValueError(
                """
                Kindly provide values for names_to.
                """
            )

    check("values_to", values_to, [str])
    df_columns = df.columns

    dot_value = (names_to is not None) and (
        (".value" in names_to) or (isinstance(names_pattern, (list, tuple)))
    )
    if (values_to in df_columns) and (not dot_value):
        # copied from pandas' melt source code
        # with a minor tweak
        raise ValueError(
            """
            This dataframe has a column name that matches the
            values_to argument.
            Kindly set the values_to parameter to a unique name.
            """
        )

    # avoid duplicate columns in the final output
    if (names_to is not None) and (not dot_value) and (values_to in names_to):
        raise ValueError(
            """
            `values_to` is present in names_to;
            this is not allowed. Kindly use a unique
            name.
            """
        )

    if any((names_sep, names_pattern)) and (
        isinstance(df_columns, pd.MultiIndex)
    ):
        raise ValueError(
            """
            Unpivoting a MultiIndex column dataframe
            when names_sep or names_pattern is supplied
            is not supported.
            """
        )

    if all((names_sep is None, names_pattern is None)):
        # adapted from pandas' melt source code
        if (
            (index is not None)
            and isinstance(df_columns, pd.MultiIndex)
            and (not isinstance(index, list))
        ):
            raise ValueError(
                """
                index must be a list of tuples
                when columns are a MultiIndex.
                """
            )

        if (
            (column_names is not None)
            and isinstance(df_columns, pd.MultiIndex)
            and (not isinstance(column_names, list))
        ):
            raise ValueError(
                """
                column_names must be a list of tuples
                when columns are a MultiIndex.
                """
            )

    check("sort_by_appearance", sort_by_appearance, [bool])

    check("ignore_index", ignore_index, [bool])

    return (
        df,
        index,
        column_names,
        names_to,
        values_to,
        column_level,
        names_sep,
        names_pattern,
        sort_by_appearance,
        ignore_index,
    )


def _computations_pivot_longer(
    df: pd.DataFrame,
    index: list = None,
    column_names: list = None,
    names_to: list = None,
    values_to: str = "value",
    column_level: Union[int, str] = None,
    names_sep: Union[str, Pattern] = None,
    names_pattern: Union[list, tuple, str, Pattern] = None,
    sort_by_appearance: bool = False,
    ignore_index: bool = True,
) -> pd.DataFrame:
    """
    This is where the final dataframe in long form is created.
    """

    if (
        (index is None)
        and column_names
        and (df.columns.size > len(column_names))
    ):
        index = [
            column_name
            for column_name in df
            if column_name not in column_names
        ]

    # scenario 1
    if all((names_pattern is None, names_sep is None)):
        if names_to:
            for word in names_to:
                if word in index:
                    raise ValueError(
                        f"""
                        `{word}` in names_to already exists
                        in column labels assigned
                        to the dataframe's index parameter.
                        Kindly use a unique name.
                        """
                    )

        len_index = len(df)

        df = pd.melt(
            df,
            id_vars=index,
            value_vars=column_names,
            var_name=names_to,
            value_name=values_to,
            col_level=column_level,
            ignore_index=ignore_index,
        )

        if sort_by_appearance:
            df = _sort_by_appearance_for_melt(df=df, len_index=len_index)

        if ignore_index:
            df.index = np.arange(len(df))

        return df

    # names_sep or names_pattern
    if index:
        df = df.set_index(index, append=True)

    if column_names:
        df = df.loc[:, column_names]

    df_index_names = df.index.names

    # checks to avoid duplicate columns
    # idea is that if there is no `.value`
    # then the word should not exist in the index
    # if, however there is `.value`
    # then the word should not be found in
    # neither the index or column names

    # idea from pd.wide_to_long
    for word in names_to:
        if (".value" not in names_to) and (word in df_index_names):
            raise ValueError(
                f"""
                `{word}` in names_to already exists
                in column labels assigned
                to the dataframe's index.
                Kindly use a unique name.
                """
            )

        if (
            (".value" in names_to)
            and (word != ".value")
            and (word in df_index_names)
        ):
            raise ValueError(
                f"""
                `{word}` in names_to already exists
                in column labels assigned
                to the dataframe's index.
                Kindly use a unique name.
                """
            )

    if names_sep:
        return _pivot_longer_names_sep(
            df,
            index,
            names_to,
            names_sep,
            values_to,
            sort_by_appearance,
            ignore_index,
        )

    if isinstance(names_pattern, (str, Pattern)):
        return _pivot_longer_names_pattern_str(
            df,
            index,
            names_to,
            names_pattern,
            values_to,
            sort_by_appearance,
            ignore_index,
        )

    return _pivot_longer_names_pattern_sequence(
        df, index, names_to, names_pattern, sort_by_appearance, ignore_index
    )


def _computations_pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[List, str]] = None,
    names_from: Optional[Union[List, str]] = None,
    values_from: Optional[Union[List, str]] = None,
    levels_order: Optional[list] = None,
    flatten_levels: Optional[bool] = True,
    names_sep="_",
    names_glue: Callable = None,
) -> pd.DataFrame:
    """
    This is the main workhorse of the `pivot_wider` function.

    It is a wrapper around `pd.pivot`. For a MultiIndex, the
    order of the levels can be changed with `levels_order`.
    The output for multiple `names_from` and/or `values_from`
    can be controlled with `names_glue` and/or `names_sep`.

    A dataframe pivoted from long to wide form is returned.
    """

    (
        df,
        index,
        names_from,
        values_from,
        levels_order,
        flatten_levels,
        names_sep,
        names_glue,
    ) = _data_checks_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        levels_order,
        flatten_levels,
        names_sep,
        names_glue,
    )
    # check dtype of `names_from` is string
    names_from_all_strings = (
        df.filter(names_from).agg(is_string_dtype).all().item()
    )

    # check dtype of columns
    column_dtype = is_string_dtype(df.columns)

    df = df.pivot(  # noqa: PD010
        index=index, columns=names_from, values=values_from
    )

    if levels_order and (isinstance(df.columns, pd.MultiIndex)):
        df = df.reorder_levels(order=levels_order, axis="columns")

    # an empty df is likely because
    # there is no `values_from`
    if any((df.empty, flatten_levels is False)):
        return df

    # ensure all entries in names_from are strings
    if (names_from_all_strings is False) or (column_dtype is False):
        if isinstance(df.columns, pd.MultiIndex):
            new_columns = [tuple(map(str, ent)) for ent in df]
            df.columns = pd.MultiIndex.from_tuples(new_columns)
        else:
            df.columns = df.columns.astype(str)

    if (names_sep is not None) and (isinstance(df.columns, pd.MultiIndex)):
        df.columns = df.columns.map(names_sep.join)

    if names_glue:
        df.columns = df.columns.map(names_glue)

    # if columns are of category type
    # this returns columns to object dtype
    # also, resetting index with category columns is not possible
    df.columns = [*df.columns]

    if index:
        df = df.reset_index()

    if df.columns.names:
        df = df.rename_axis(columns=None)

    return df


def _sort_by_appearance_for_melt(
    df: pd.DataFrame, len_index: int
) -> pd.DataFrame:
    """
    This function sorts the resulting dataframe by appearance,
    via the `sort_by_appearance` parameter in `computations_pivot_longer`.

    A dataframe that is sorted by appearance is returned.
    """

    index_sorter = None

    # explanation here to help future me :)

    # if the height of the new dataframe
    # is the same as the height of the original dataframe,
    # then there is no need to sort by appearance
    length_check = any((len_index == 1, len_index == len(df)))

    # pd.melt flips the columns into vertical positions
    # it `tiles` the index during the flipping
    # example:

    #          first last  height  weight
    # person A  John  Doe     5.5     130
    #        B  Mary   Bo     6.0     150

    # melting the dataframe above yields:
    # df.melt(['first', 'last'])

    #   first last variable  value
    # 0  John  Doe   height    5.5
    # 1  Mary   Bo   height    6.0
    # 2  John  Doe   weight  130.0
    # 3  Mary   Bo   weight  150.0

    # sort_by_appearance `untiles` the index
    # and keeps all `John` before all `Mary`
    # since `John` appears first in the original dataframe:

    #   first last variable  value
    # 0  John  Doe   height    5.5
    # 1  John  Doe   weight  130.0
    # 2  Mary   Bo   height    6.0
    # 3  Mary   Bo   weight  150.0

    # to get to this second form, which is sorted by appearance,
    # get the lengths of the dataframe
    # before and after it is melted
    # for the example above, the length before melting is 2
    # and after - 4.
    # reshaping allows us to track the original positions
    # in the previous dataframe ->
    # np.reshape([0,1,2,3], (-1, 2))
    # array([[0, 1],
    #        [2, 3]])
    # ravel, with the Fortran order (`F`) ensures the John's are aligned
    # before the Mary's -> [0, 2, 1, 3]
    # the raveled array is then passed to `take`
    if not length_check:
        index_sorter = np.arange(len(df))
        index_sorter = np.reshape(index_sorter, (-1, len_index))
        index_sorter = index_sorter.ravel(order="F")
        df = df.take(index_sorter)

    return df


def _pivot_longer_frame_MultiIndex(
    df: pd.DataFrame,
    index,
    sort_by_appearance: bool,
    ignore_index: bool,
    values_to: str,
) -> pd.DataFrame:
    """
    This creates the final dataframe,
    where names_sep/names_pattern is provided,
    and the extraction/split of the columns
    result in a MultiIndex. This applies only
    to names_sep or names_pattern as a string,
    where more than one group is present in the
    regex.
    """

    len_index = len(df)
    mapping = df.columns
    if ".value" not in mapping.names:
        df = df.melt(ignore_index=False, value_name=values_to)

        if sort_by_appearance:
            df = _sort_by_appearance_for_melt(df=df, len_index=len_index)

        if index:
            df = df.reset_index(index)

        if ignore_index:
            df.index = range(len(df))

        return df

    # labels that are not `.value`
    # required when recombining list of individual dataframes
    # as they become the keys in the concatenation
    others = mapping.droplevel(".value").unique()
    if isinstance(others, pd.MultiIndex):
        levels = others.names
    else:
        levels = others.name
    # here, we get the dataframes containing the `.value` labels
    # as columns
    # and then concatenate vertically, using the other variables
    # in `names_to`, which in this is case, is captured in `others`
    # as keys. This forms a MultiIndex; reset_index puts it back
    # as columns into the dataframe.
    df = [df.xs(key=key, axis="columns", level=levels) for key in others]
    df = pd.concat(df, keys=others, axis="index", copy=False, sort=False)
    if isinstance(levels, str):
        levels = [levels]
    # represents the cumcount,
    # used in making the columns unique (if they werent originally)
    null_in_levels = None in levels
    # gets rid of None, for scenarios where we
    # generated cumcount to make the columns unique
    levels = [level for level in levels if level]
    # need to order the dataframe's index
    # so that when resetting,
    # the index appears before the other columns
    # this is relevant only if `index` is True
    # using numbers here, in case there are multiple Nones
    # in the index names
    if index:
        new_order = np.roll(np.arange(len(df.index.names)), len(index) + 1)
        df = df.reorder_levels(new_order, axis="index")
        df = df.reset_index(level=index + levels)
    else:
        df = df.reset_index(levels)

    if null_in_levels:
        df = df.droplevel(level=-1, axis="index")

    if df.columns.names:
        df = df.rename_axis(columns=None)

    if sort_by_appearance:
        df = _sort_by_appearance_for_melt(df=df, len_index=len_index)

    if ignore_index:
        df.index = range(len(df))

    return df


def _pivot_longer_frame_single_Index(
    df: pd.DataFrame,
    index,
    sort_by_appearance: bool,
    ignore_index: bool,
    values_to: str = None,
) -> pd.DataFrame:
    """
    This creates the final dataframe,
    where names_pattern is provided,
    and the extraction/split of the columns
    result in a single Index.
    This covers scenarios where names_pattern
    is a list/tuple, or where a single group
    is present in the regex string.
    """

    if df.columns.name != ".value":
        len_index = len(df)
        df = df.melt(ignore_index=False, value_name=values_to)

        if sort_by_appearance:
            df = _sort_by_appearance_for_melt(df=df, len_index=len_index)

        if index:
            df = df.reset_index(index)

        if ignore_index:
            df.index = range(len(df))

        return df

    mapping = df.columns
    len_df_columns = mapping.size
    mapping = mapping.unique()
    len_mapping = mapping.size

    len_index = len(df)

    if len_df_columns > 1:
        container = defaultdict(list)
        for name, series in df.items():
            container[name].append(series)
        if len_mapping == 1:  # single unique column
            container = container[mapping[0]]
            df = pd.concat(
                container, axis="index", join="outer", sort=False, copy=False
            )
            df = df.to_frame()
        else:
            # concat works fine here and efficient too,
            # since we are combining Series
            # a Series is returned for each concatenation
            # the outer keys serve as a pairing mechanism
            # for recombining the dataframe
            # so if we have a dataframe like below:
            #        id  x1  x2  y1  y2
            #    0   1   4   5   7  10
            #    1   2   5   6   8  11
            #    2   3   6   7   9  12
            # then x1 will pair with y1, and x2 will pair with y2
            # if the dataframe column positions were alternated, like below:
            #        id  x2  x1  y1  y2
            #    0   1   5   4   7  10
            #    1   2   6   5   8  11
            #    2   3   7   6   9  12
            # then x2 will pair with y1 and x1 will pair with y2
            # it is simply a first come first serve approach
            df = [
                pd.concat(value, copy=False, keys=np.arange(len(value)))
                for _, value in container.items()
            ]
            first, *rest = df
            first = first.to_frame()
            df = first.join(rest, how="outer", sort=False)
            # drop outermost keys (used in the concatenation)
            df = df.droplevel(level=0, axis="index")

    if df.columns.names:
        df = df.rename_axis(columns=None)

    if sort_by_appearance:
        df = _sort_by_appearance_for_melt(df=df, len_index=len_index)

    if index:
        df = df.reset_index(index)

    if ignore_index:
        df.index = range(len(df))

    return df


def _data_checks_pivot_wider(
    df,
    index,
    names_from,
    values_from,
    levels_order,
    flatten_levels,
    names_sep,
    names_glue,
):

    """
    This function raises errors if the arguments have the wrong
    python type, or if the column does not exist in the dataframe.
    This function is executed before proceeding to the computation phase.
    Type annotations are not provided because this function is where type
    checking happens.
    """

    if index is not None:
        if is_list_like(index):
            index = [*index]
        index = _select_column_names(index, df)

    if names_from is None:
        raise ValueError(
            "pivot_wider() is missing 1 required argument: 'names_from'"
        )

    if is_list_like(names_from):
        names_from = [*names_from]
    names_from = _select_column_names(names_from, df)

    if values_from is not None:
        if is_list_like(values_from):
            values_from = [*values_from]
        values_from = _select_column_names(values_from, df)
        if len(values_from) == 1:
            values_from = values_from[0]

    if levels_order is not None:
        check("levesl_order", levels_order, [list])

    check("flatten_levels", flatten_levels, [bool])

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    if names_glue is not None:
        check("names_glue", names_glue, [callable])

    return (
        df,
        index,
        names_from,
        values_from,
        levels_order,
        flatten_levels,
        names_sep,
        names_glue,
    )


def _pivot_longer_names_pattern_sequence(
    df: pd.DataFrame,
    index,
    names_to: list,
    names_pattern: Union[list, tuple],
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_pattern is provided, and is a list/tuple.
    """

    df_columns = df.columns
    mapping = [
        df_columns.str.contains(regex, na=False, regex=True)
        for regex in names_pattern
    ]

    matches = [arr.any() for arr in mapping]
    if np.any(matches).item() is False:
        raise ValueError(
            """
            No label in the columns
            matched the regexes
            in names_pattern.
            Kindly provide regexes
            that match all labels
            in the columns.
            """
        )
    for position, boolean in enumerate(matches):
        if boolean.item() is False:
            raise ValueError(
                f"""
                No match was returned for
                regex `{names_pattern[position]}`
                """
            )

    mapping = np.select(mapping, names_to, None)
    # guard .. for scenarios where not all labels
    # in the columns are matched to the regex(es)
    # the any_nulls takes care of that,
    # via boolean indexing
    any_nulls = pd.notna(mapping)
    mapping = pd.MultiIndex.from_arrays([mapping, df_columns])
    mapping.names = [".value", None]
    df.columns = mapping
    if any_nulls.any():
        df = df.loc[:, any_nulls]
    df = df.droplevel(level=-1, axis="columns")

    return _pivot_longer_frame_single_Index(
        df, index, sort_by_appearance, ignore_index, values_to=None
    )


def _pivot_longer_names_pattern_str(
    df: pd.DataFrame,
    index,
    names_to: list,
    names_pattern: Union[str, Pattern],
    values_to: str,
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_pattern is provided, and is a string.
    """

    mapping = df.columns.str.extract(names_pattern, expand=True)

    nulls_found = mapping.isna()

    if nulls_found.all(axis=None):
        raise ValueError(
            """
            No labels in the columns
            matched the regular expression
            in names_pattern.
            Kindly provide a regular expression
            that matches all labels in the columns.
            """
        )

    if nulls_found.any(axis=None):
        raise ValueError(
            f"""
            Not all labels in the columns
            matched the regular expression
            in names_pattern.Column Labels
            {*df.columns[nulls_found.any(axis='columns')],}
            could not be matched with the regex.
            Kindly provide a regular expression
            (with the correct groups) that matches all labels
            in the columns.
            """
        )

    mapping.columns = names_to

    if len(names_to) == 1:
        mapping = mapping.squeeze()
        df.columns = mapping
        return _pivot_longer_frame_single_Index(
            df, index, sort_by_appearance, ignore_index, values_to
        )

    if ".value" in names_to:
        exclude = mapping[".value"].array
        for word in names_to:
            if (word != ".value") and (word in exclude):
                raise ValueError(
                    f"""
                    `{word}` in names_to already exists
                    in the new dataframe's columns.
                    Kindly use a unique name.
                    """
                )

    mapping_is_unique = not mapping.duplicated().any(axis=None).item()

    if mapping_is_unique or (".value" not in names_to):
        mapping = pd.MultiIndex.from_frame(mapping)
    else:
        cumcount = mapping.groupby(names_to).cumcount()
        mapping = [series for _, series in mapping.items()]
        mapping.append(cumcount)
        mapping = pd.MultiIndex.from_arrays(mapping)
    df.columns = mapping

    return _pivot_longer_frame_MultiIndex(
        df, index, sort_by_appearance, ignore_index, values_to
    )


def _pivot_longer_names_sep(
    df: pd.DataFrame,
    index,
    names_to: list,
    names_sep: Union[str, Pattern],
    values_to: str,
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_sep is provided.
    """

    mapping = pd.Series(df.columns).str.split(names_sep, expand=True)
    len_mapping_columns = len(mapping.columns)
    len_names_to = len(names_to)

    if len_names_to != len_mapping_columns:
        raise ValueError(
            f"""
            The length of names_to does not match
            the number of levels extracted.
            The length of names_to is {len_names_to}
            while the number of levels extracted is
            {len_mapping_columns}.
            """
        )

    mapping.columns = names_to

    if ".value" in names_to:
        exclude = mapping[".value"].array
        for word in names_to:
            if (word != ".value") and (word in exclude):
                raise ValueError(
                    f"""
                    `{word}` in names_to already exists
                    in the new dataframe's columns.
                    Kindly use a unique name.
                    """
                )

    # having unique columns ensure the data can be recombined
    # successfully via pd.concat; if the columns are not unique,
    # a counter is created with cumcount to ensure uniqueness.
    # This is dropped later on, and is not part of the final
    # dataframe.
    # This is relevant only for scenarios where `.value` is
    # in names_to.
    mapping_is_unique = not mapping.duplicated().any(axis=None).item()

    if mapping_is_unique or (".value" not in names_to):
        mapping = pd.MultiIndex.from_frame(mapping)
    else:
        cumcount = mapping.groupby(names_to).cumcount()
        mapping = [series for _, series in mapping.items()]
        mapping.append(cumcount)
        mapping = pd.MultiIndex.from_arrays(mapping)
    df.columns = mapping

    return _pivot_longer_frame_MultiIndex(
        df, index, sort_by_appearance, ignore_index, values_to
    )
