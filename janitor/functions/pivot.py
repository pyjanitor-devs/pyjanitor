from collections import defaultdict
from itertools import zip_longest, chain
import re
import warnings
from typing import Optional, Pattern, Union, Callable, Generator

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import (
    is_list_like,
    is_string_dtype,
)
from pandas.core.construction import extract_array
from pandas.core.dtypes.concat import concat_compat

from janitor.functions.utils import (
    _select_column_names,
    _computations_expand_grid,
)
from janitor.utils import check


@pf.register_dataframe_method
def pivot_longer(
    df: pd.DataFrame,
    index: Optional[Union[list, tuple, str, Pattern]] = None,
    column_names: Optional[Union[list, tuple, str, Pattern]] = None,
    names_to: Optional[Union[list, tuple, str]] = None,
    values_to: Optional[str] = "value",
    column_level: Optional[Union[int, str]] = None,
    names_sep: Optional[Union[str, Pattern]] = None,
    names_pattern: Optional[Union[list, tuple, str, Pattern]] = None,
    names_transform: Optional[Union[str, Callable, dict]] = None,
    sort_by_appearance: Optional[bool] = False,
    ignore_index: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Unpivots a DataFrame from *wide* to *long* format.

    This method does not mutate the original DataFrame.

    It is modeled after the `pivot_longer` function in R's tidyr package,
    and also takes inspiration from R's data.table package.

    This function is useful to massage a DataFrame into a format where
    one or more columns are considered measured variables, and all other
    columns are considered as identifier variables.

    All measured variables are *unpivoted* (and typically duplicated) along the
    row axis.

    Column selection in `index` and `column_names` is possible using the
    [`select_columns`][janitor.functions.select_columns.select_columns] syntax.

    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "Sepal.Length": [5.1, 5.9],
        ...         "Sepal.Width": [3.5, 3.0],
        ...         "Petal.Length": [1.4, 5.1],
        ...         "Petal.Width": [0.2, 1.8],
        ...         "Species": ["setosa", "virginica"],
        ...     }
        ... )
        >>> df
           Sepal.Length  Sepal.Width  Petal.Length  Petal.Width    Species
        0           5.1          3.5           1.4          0.2     setosa
        1           5.9          3.0           5.1          1.8  virginica

    Replicate pandas' melt:

        >>> df.pivot_longer(index = 'Species')
             Species      variable  value
        0     setosa  Sepal.Length    5.1
        1  virginica  Sepal.Length    5.9
        2     setosa   Sepal.Width    3.5
        3  virginica   Sepal.Width    3.0
        4     setosa  Petal.Length    1.4
        5  virginica  Petal.Length    5.1
        6     setosa   Petal.Width    0.2
        7  virginica   Petal.Width    1.8

    Split the column labels into parts:

        >>> df.pivot_longer(
        ...     index = 'Species',
        ...     names_to = ('part', 'dimension'),
        ...     names_sep = '.',
        ...     sort_by_appearance = True,
        ... )
             Species   part dimension  value
        0     setosa  Sepal    Length    5.1
        1     setosa  Sepal     Width    3.5
        2     setosa  Petal    Length    1.4
        3     setosa  Petal     Width    0.2
        4  virginica  Sepal    Length    5.9
        5  virginica  Sepal     Width    3.0
        6  virginica  Petal    Length    5.1
        7  virginica  Petal     Width    1.8

    Retain parts of the column names as headers:

        >>> df.pivot_longer(
        ...     index = 'Species',
        ...     names_to = ('part', '.value'),
        ...     names_sep = '.',
        ...     sort_by_appearance = True,
        ... )
             Species   part  Length  Width
        0     setosa  Sepal     5.1    3.5
        1     setosa  Petal     1.4    0.2
        2  virginica  Sepal     5.9    3.0
        3  virginica  Petal     5.1    1.8

    Split the column labels based on regex:

        >>> df = pd.DataFrame({"id": [1], "new_sp_m5564": [2], "newrel_f65": [3]})
        >>> df
           id  new_sp_m5564  newrel_f65
        0   1             2           3
        >>> df.pivot_longer(
        ...     index = 'id',
        ...     names_to = ('diagnosis', 'gender', 'age'),
        ...     names_pattern = r"new_?(.+)_(.)(\\d+)",
        ... )
           id diagnosis gender   age  value
        0   1        sp      m  5564      2
        1   1       rel      f    65      3

    Convert the dtypes of specific columns with `names_transform`:

        >>> result = (df
        ...          .pivot_longer(
        ...              index = 'id',
        ...              names_to = ('diagnosis', 'gender', 'age'),
        ...              names_pattern = r"new_?(.+)_(.)(\\d+)",
        ...              names_transform = {'gender': 'category', 'age':'int'})
        ... )
        >>> result.dtypes
        id           int64
        gender    category
        age          int64
        value        int64
        dtype: object

    Use multiple `.value` to reshape dataframe:

        >>> df = pd.DataFrame(
        ...     [
        ...         {
        ...             "x_1_mean": 10,
        ...             "x_2_mean": 20,
        ...             "y_1_mean": 30,
        ...             "y_2_mean": 40,
        ...             "unit": 50,
        ...         }
        ...     ]
        ... )
        >>> df
           x_1_mean  x_2_mean  y_1_mean  y_2_mean  unit
        0        10        20        30        40    50
        >>> df.pivot_longer(
        ...     index="unit",
        ...     names_to=(".value", "time", ".value"),
        ...     names_pattern=r"(x|y)_([0-9])(_mean)",
        ... )
           unit time  x_mean  y_mean
        0    50    1      10      30
        1    50    2      20      40

    Multiple values_to:

        >>> df = pd.DataFrame(
        ...         {
        ...             "City": ["Houston", "Austin", "Hoover"],
        ...             "State": ["Texas", "Texas", "Alabama"],
        ...             "Name": ["Aria", "Penelope", "Niko"],
        ...             "Mango": [4, 10, 90],
        ...             "Orange": [10, 8, 14],
        ...             "Watermelon": [40, 99, 43],
        ...             "Gin": [16, 200, 34],
        ...             "Vodka": [20, 33, 18],
        ...         },
        ...         columns=[
        ...             "City",
        ...             "State",
        ...             "Name",
        ...             "Mango",
        ...             "Orange",
        ...             "Watermelon",
        ...             "Gin",
        ...             "Vodka",
        ...         ],
        ...     )
        >>> df
              City    State      Name  Mango  Orange  Watermelon  Gin  Vodka
        0  Houston    Texas      Aria      4      10          40   16     20
        1   Austin    Texas  Penelope     10       8          99  200     33
        2   Hoover  Alabama      Niko     90      14          43   34     18
        >>> df.pivot_longer(
        ...         index=["City", "State"],
        ...         column_names=slice("Mango", "Vodka"),
        ...         names_to=("Fruit", "Drink"),
        ...         values_to=("Pounds", "Ounces"),
        ...         names_pattern=[r"M|O|W", r"G|V"],
        ...     )
              City    State       Fruit  Pounds  Drink  Ounces
        0  Houston    Texas       Mango       4    Gin    16.0
        1   Austin    Texas       Mango      10    Gin   200.0
        2   Hoover  Alabama       Mango      90    Gin    34.0
        3  Houston    Texas      Orange      10  Vodka    20.0
        4   Austin    Texas      Orange       8  Vodka    33.0
        5   Hoover  Alabama      Orange      14  Vodka    18.0
        6  Houston    Texas  Watermelon      40   None     NaN
        7   Austin    Texas  Watermelon      99   None     NaN
        8   Hoover  Alabama  Watermelon      43   None     NaN

    :param df: A pandas DataFrame.
    :param index: Name(s) of columns to use as identifier variables.
        Should be either a single column name, or a list/tuple of
        column names.
        `index` should be a list of tuples if the columns are a MultiIndex.
    :param column_names: Name(s) of columns to unpivot. Should be either
        a single column name or a list/tuple of column names.
        `column_names` should be a list of tuples
        if the columns are a MultiIndex.
    :param names_to: Name of new column as a string that will contain
        what were previously the column names in `column_names`.
        The default is `variable` if no value is provided. It can
        also be a list/tuple of strings that will serve as new column
        names, if `name_sep` or `names_pattern` is provided.
        If `.value` is in `names_to`, new column names will be extracted
        from part of the existing column names and overrides`values_to`.
    :param values_to: Name of new column as a string that will contain what
        were previously the values of the columns in `column_names`.
        values_to can also be a list/tuple
        and requires that names_pattern is also a list/tuple.
    :param column_level: If columns are a MultiIndex, then use this level to
        unpivot the DataFrame. Provided for compatibility with pandas' melt,
        and applies only if neither `names_sep` nor `names_pattern` is
        provided.
    :param names_sep: Determines how the column name is broken up, if
        `names_to` contains multiple values. It takes the same
        specification as pandas' `str.split` method, and can be a string
        or regular expression. `names_sep` does not work with MultiIndex
        columns.
    :param names_pattern: Determines how the column name is broken up.
        It can be a regular expression containing matching groups (it takes
        the same specification as pandas' `str.extract` method), or a
        list/tuple of regular expressions. If it is a single regex, the
        number of groups must match the length of `names_to`.
        For a list/tuple of regular expressions,
        `names_to` must also be a list/tuple and the lengths of both
        arguments must match.
        `names_pattern` does not work with MultiIndex columns.
    :param names_transform: Use this option to change the types of columns that
        have been transformed to rows. This does not applies to the values columns.
        Accepts a string, a callable, or a dictionary that is acceptable by `pd.astype`.
    :param sort_by_appearance: Default `False`. Boolean value that determines
        the final look of the DataFrame. If `True`, the unpivoted DataFrame
        will be stacked in order of first appearance.
    :param ignore_index: Default `True`. If `True`,
        the original index is ignored. If `False`, the original index
        is retained and the index labels will be repeated as necessary.
    :returns: A pandas DataFrame that has been unpivoted from wide to long
        format.
    """  # noqa: E501

    # this code builds on the wonderful work of @benjaminjack’s PR
    # https://github.com/benjaminjack/pyjanitor/commit/e3df817903c20dd21634461c8a92aec137963ed0

    df = df.copy()

    (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        names_transform,
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
        names_transform,
        sort_by_appearance,
        ignore_index,
    )

    return _computations_pivot_longer(
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        names_transform,
        sort_by_appearance,
        ignore_index,
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
    names_transform,
    sort_by_appearance,
    ignore_index,
) -> tuple:
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

    if any((names_sep, names_pattern)) and (
        isinstance(df.columns, pd.MultiIndex)
    ):
        raise ValueError(
            "Unpivoting a MultiIndex column dataframe "
            "when names_sep or names_pattern is supplied "
            "is not supported."
        )

    if all((names_sep is None, names_pattern is None)):
        # adapted from pandas' melt source code
        if (
            (index is not None)
            and isinstance(df.columns, pd.MultiIndex)
            and (not isinstance(index, list))
        ):
            raise ValueError(
                "index must be a list of tuples "
                "when the columns are a MultiIndex."
            )

        if (
            (column_names is not None)
            and isinstance(df.columns, pd.MultiIndex)
            and (not isinstance(column_names, list))
        ):
            raise ValueError(
                "column_names must be a list of tuples "
                "when the columns are a MultiIndex."
            )

    if column_names is not None:
        if is_list_like(column_names) and (
            not isinstance(column_names, tuple)
        ):
            column_names = [*column_names]
        column_names = _select_column_names(column_names, df)

    if index is not None:
        if is_list_like(index) and (not isinstance(index, tuple)):
            index = [*index]
        index = _select_column_names(index, df)

    if not index:
        if not column_names:
            column_names = [*df.columns]
        else:
            index = [*df.columns.difference(column_names, sort=False)]
    else:
        if not column_names:
            column_names = [*df.columns.difference(index, sort=False)]

    len_names_to = 0
    if names_to is not None:
        if isinstance(names_to, str):
            names_to = [names_to]
        elif isinstance(names_to, tuple):
            names_to = [*names_to]
        check("names_to", names_to, [list, str, tuple])

        uniques = set()
        for word in names_to:
            check(f"{word} in names_to", word, [str])
            if (word in uniques) and (word != ".value"):
                raise ValueError(f"{word} is duplicated in names_to.")
            uniques.add(word)

        len_names_to = len(names_to)
    else:
        if not any((names_sep, names_pattern)):
            names_to = ["variable"]

    check("values_to", values_to, [str, list, tuple])
    if isinstance(values_to, (list, tuple)) and (
        not isinstance(names_pattern, (list, tuple))
    ):
        raise TypeError(
            "values_to can be a list/tuple only "
            "if names_pattern is a list/tuple."
        )
    if (
        (names_sep is None)
        and (names_pattern is None)
        and index
        and (values_to in index)
    ):
        raise ValueError(
            "The argument provided to values_to "
            "already exist as a column label "
            "assigned to the dataframe's index parameter. "
            "Kindly use a unique label."
        )

    if names_sep and names_pattern:
        raise ValueError(
            "Only one of names_pattern or names_sep should be provided."
        )

    if names_pattern is not None:
        check("names_pattern", names_pattern, [str, Pattern, list, tuple])
        if names_to is None:
            raise ValueError("Kindly provide values for names_to.")
        if isinstance(names_pattern, (str, Pattern)):
            num_regex_grps = re.compile(names_pattern).groups

            if len_names_to != num_regex_grps:
                raise ValueError(
                    f"The length of names_to does not match "
                    "the number of groups in names_pattern. "
                    f"The length of names_to is {len_names_to} "
                    "while the number of groups in the regex "
                    f"is {num_regex_grps}."
                )

        elif isinstance(names_pattern, (list, tuple)):
            for word in names_pattern:
                check(f"{word} in names_pattern", word, [str, Pattern])

            if len(names_pattern) != len_names_to:
                raise ValueError(
                    f"The length of names_to does not match "
                    "the number of regexes in names_pattern. "
                    f"The length of names_to is {len_names_to} "
                    f"while the number of regexes is {len(names_pattern)}."
                )

            if names_to and (".value" in names_to):
                raise ValueError(
                    ".value is not accepted in names_to "
                    "if names_pattern is a list/tuple."
                )

            if isinstance(values_to, (list, tuple)):
                if len(values_to) != len(names_pattern):
                    raise ValueError(
                        f"The length of values_to does not match "
                        "the number of regexes in names_pattern. "
                        f"The length of values_to is {len(values_to)} "
                        f"while the number of regexes is {len(names_pattern)}."
                    )
                uniques = set()
                for word in values_to:
                    check(f"{word} in values_to", word, [str])
                    if word in names_to:
                        raise ValueError(
                            f"{word} in values_to already exists in names_to."
                        )

                    if word in uniques:
                        raise ValueError(f"{word} is duplicated in values_to.")
                    uniques.add(word)

    if names_sep is not None:
        check("names_sep", names_sep, [str, Pattern])
        if names_to is None:
            raise ValueError("Kindly provide values for names_to.")

    dot_value = (names_to is not None) and (
        (".value" in names_to) or (isinstance(names_pattern, (list, tuple)))
    )

    # avoid duplicate columns in the final output
    if (names_to is not None) and (not dot_value):
        if values_to in names_to:
            raise ValueError(
                "The argument provided for values_to "
                "already exists in names_to; "
                "Kindly use a unique name."
            )
        # idea is that if there is no `.value`
        # then the label should not exist in the index
        # there is no need to check the columns
        # since that is what we are replacing with names_to
        if index:
            exclude = set(names_to).intersection(index)
            if exclude:
                raise ValueError(
                    f"Labels {*exclude, } in names_to already exist "
                    "as column labels assigned to the dataframe's "
                    "index parameter. Kindly provide unique label(s)."
                )

    if names_transform:
        check("names_transform", names_transform, [str, Callable, dict])

    check("sort_by_appearance", sort_by_appearance, [bool])

    check("ignore_index", ignore_index, [bool])

    if isinstance(df.columns, pd.MultiIndex):
        if not any(df.columns.names):
            if len(names_to) == 1:
                df.columns.names = [
                    f"{names_to[0]}_{i}" for i in range(df.columns.nlevels)
                ]
            elif len(names_to) == df.columns.nlevels:
                df.columns.names = names_to
            else:
                raise ValueError(
                    "The length of names_to does not match "
                    "the number of levels in the columns. "
                    f"names_to has a length of {len(names_to)}, "
                    "while the number of column levels is "
                    f"{df.columns.nlevels}."
                )
        elif None in df.columns.names:
            raise ValueError(
                "Kindly ensure there is no None "
                "in the names for the column levels."
            )
    elif (
        not isinstance(df.columns, pd.MultiIndex)
        and not any((names_sep, names_pattern))
        and (not df.columns.names[0])
    ):
        df.columns.names = names_to

    return (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        names_transform,
        sort_by_appearance,
        ignore_index,
    )


def _computations_pivot_longer(
    df: pd.DataFrame,
    index: Union[list, None],
    column_names: Union[list, None],
    names_to: Union[list, None],
    values_to: Union[list, str, None],
    names_sep: Union[str, Pattern],
    names_pattern: Union[list, tuple, str, Pattern],
    names_transform: Union[str, Callable, dict],
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This is where the final dataframe in long form is created.
    """
    # the core idea for the combination/reshape
    # is that the index will be tiled, while the rest will be repeated
    # where necessary ------->
    # if index is [1,2,3] then tiling makes it [1,2,3,1,2,3,...]
    # for column names, if it is [1,2,3], then repeats [1,1,1,2,2,2,3,3,3]
    # dump down into arrays, and build a new dataframe, with copy = False
    # since we already have made a copy of the original df

    if not column_names:
        return df.rename_axis(columns=None)

    if index:
        index = {
            name: extract_array(df[name], extract_numpy=True) for name in index
        }

    df = df.loc[:, column_names]

    len_index = len(df)

    if all((names_pattern is None, names_sep is None)):
        return _base_melt(
            df=df,
            index=index,
            len_index=len_index,
            values_to=values_to,
            names_transform=names_transform,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
        )

    if names_sep is not None:
        return _pivot_longer_names_sep(
            df=df,
            index=index,
            len_index=len_index,
            names_to=names_to,
            names_sep=names_sep,
            names_transform=names_transform,
            values_to=values_to,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
        )

    if isinstance(names_pattern, (str, Pattern)):
        return _pivot_longer_names_pattern_str(
            df=df,
            index=index,
            len_index=len_index,
            names_to=names_to,
            names_pattern=names_pattern,
            names_transform=names_transform,
            values_to=values_to,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
        )

    return _pivot_longer_names_pattern_sequence(
        df=df,
        index=index,
        len_index=len_index,
        names_to=names_to,
        names_pattern=names_pattern,
        names_transform=names_transform,
        sort_by_appearance=sort_by_appearance,
        values_to=values_to,
        ignore_index=ignore_index,
    )


def _pivot_longer_names_pattern_sequence(
    df: pd.DataFrame,
    index: Union[dict, None],
    len_index: int,
    names_to: list,
    names_pattern: Union[list, tuple],
    names_transform: Union[str, Callable, dict],
    sort_by_appearance: bool,
    values_to: Union[str, list, tuple],
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_pattern is provided, and is a list/tuple.
    """
    values_to_is_a_sequence = isinstance(values_to, (list, tuple))
    if values_to_is_a_sequence and index:
        exclude = set(values_to).intersection(index)
        if exclude:
            raise ValueError(
                f"Labels {*exclude, } in values_to already exist as "
                "column labels assigned to the dataframe's index parameter. "
                "Kindly use unique labels."
            )
    outcome = df.columns

    mapping = [
        outcome.str.contains(regex, na=False, regex=True)
        for regex in names_pattern
    ]

    outcome = (arr.any() for arr in mapping)
    # within each match, check the individual matches
    # and raise an error if any is False
    for position, boolean in enumerate(outcome):
        if not boolean.item():
            raise ValueError(
                "No match was returned for the regex "
                f"at position {position} -> {names_pattern[position]}."
            )

    if values_to_is_a_sequence:
        mapping, values = np.select(mapping, values_to, None), np.select(
            mapping, names_to, None
        )
    else:
        mapping = np.select(mapping, names_to, None)

    # only matched columns are retained
    outcome = pd.notna(mapping)
    df = df.loc[:, outcome]
    mapping = mapping[outcome]
    if values_to_is_a_sequence:
        names_to = zip(names_to, values_to)
        names_to = [*chain.from_iterable(names_to)]
        if index:
            names_to = [*index] + names_to
        values = values[outcome]
        arr = defaultdict(list)
        for label, name in zip(values, df.columns):
            arr[label].append(name)
        values = arr.keys()
        arr = (entry for _, entry in arr.items())
        arr = zip(*zip_longest(*arr))
        arr = map(pd.Series, arr)
        values = zip(values, arr)
        if names_transform:
            values = _names_transform(
                names_transform, is_dataframe=False, values=values
            )
        values = {
            name: extract_array(arr, extract_numpy=True).repeat(len_index)
            for name, arr in values
        }
    else:
        values = {}

    mapping = pd.Series(mapping)
    outcome, group_max = _headers_single_series(df=df, mapping=mapping)

    df = _final_frame_longer(
        df=df,
        len_index=len_index,
        reps=group_max,
        index=index,
        outcome=outcome,
        values=values,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
    )

    if values_to_is_a_sequence:
        df = df.loc[:, names_to]

    return df


def _pivot_longer_names_pattern_str(
    df: pd.DataFrame,
    index: Union[list, None],
    len_index: int,
    names_to: list,
    names_pattern: Union[str, Pattern],
    names_transform: bool,
    values_to: str,
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_pattern is provided, and is a string/regex.
    """
    mapping = df.columns.str.extract(names_pattern, expand=True)

    nulls_found = mapping.isna()

    if nulls_found.any(axis=None):
        no_match = df.columns[nulls_found.any(axis="columns")]
        raise ValueError(
            f"Column labels {*no_match,} "
            "could not be matched with any of the groups "
            "in the provided regex. Kindly provide a regular expression "
            "(with the correct groups) that matches all labels in the columns."
        )

    mapping.columns = names_to

    if ".value" not in names_to:
        if len(names_to) == 1:
            df.columns = mapping.iloc[:, 0]
        else:
            df.columns = pd.MultiIndex.from_frame(mapping)
        return _base_melt(
            df=df,
            index=index,
            len_index=len_index,
            values_to=values_to,
            names_transform=names_transform,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
        )

    return _pivot_longer_dot_value(
        df=df,
        index=index,
        len_index=len_index,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
        names_to=names_to,
        names_transform=names_transform,
        mapping=mapping,
    )


def _pivot_longer_names_sep(
    df: pd.DataFrame,
    index: list,
    len_index: int,
    names_to: list,
    names_sep: Union[str, Pattern],
    values_to: str,
    names_transform: bool,
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
            f"The length of names_to does not match "
            "the number of levels extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of levels extracted is "
            f"{len_mapping_columns}."
        )

    mapping.columns = names_to

    if ".value" not in names_to:
        if len(names_to) == 1:
            df.columns = mapping.iloc[:, 0]
        else:
            df.columns = pd.MultiIndex.from_frame(mapping)
        return _base_melt(
            df=df,
            index=index,
            len_index=len_index,
            values_to=values_to,
            names_transform=names_transform,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
        )

    return _pivot_longer_dot_value(
        df=df,
        index=index,
        len_index=len_index,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
        names_to=names_to,
        names_transform=names_transform,
        mapping=mapping,
    )


def _base_melt(
    df: pd.DataFrame,
    index: Union[dict, None],
    len_index: int,
    values_to: str,
    names_transform: Union[str, Callable, dict, None],
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    Applicable where there is no `.value` in names_to.
    """

    columns = df.columns
    reps = len(columns)
    outcome = (
        (name, columns.get_level_values(name)) for name in columns.names
    )
    if names_transform:
        outcome = _names_transform(
            names_transform, is_dataframe=False, values=outcome
        )
    outcome = {
        name: extract_array(arr, extract_numpy=True).repeat(len_index)
        for name, arr in outcome
    }

    values = [extract_array(arr, extract_numpy=True) for _, arr in df.items()]
    values = {values_to: concat_compat(values)}

    return _final_frame_longer(
        df=df,
        len_index=len_index,
        reps=reps,
        index=index,
        outcome=outcome,
        values=values,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
    )


def _pivot_longer_dot_value(
    df: pd.DataFrame,
    index: Union[dict, None],
    len_index: int,
    sort_by_appearance: bool,
    ignore_index: bool,
    names_to: list,
    names_transform: Union[str, Callable, dict, None],
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivots the dataframe into the final form,
    for scenarios where names_pattern is a string/regex,
    or names_sep is provided, and .value is in names_to.

    Returns a DataFrame.
    """
    if np.count_nonzero(mapping.columns == ".value") > 1:
        outcome = mapping.pop(".value")
        outcome = outcome.agg("".join, axis=1)
        mapping[".value"] = outcome

    exclude = {
        word
        for word in mapping[".value"].array
        if (word in names_to) and (word != ".value")
    }
    if exclude:
        raise ValueError(
            f"Labels {*exclude, } in names_to already exist "
            "in the new dataframe's columns. "
            "Kindly provide unique label(s)."
        )

    if index:
        exclude = set(index).intersection(mapping[".value"])
        if exclude:
            raise ValueError(
                f"Labels {*exclude, } already exist "
                "as column labels assigned to the dataframe's "
                "index parameter. Kindly provide unique label(s)."
            )
    if mapping.columns[0] != ".value":
        outcome = mapping.pop(".value")
        mapping.insert(loc=0, column=".value", value=outcome)

    if len(mapping.columns) == 1:
        mapping = mapping.iloc[:, 0]
        values, group_max = _headers_single_series(df=df, mapping=mapping)
        outcome = {}
    else:
        # For multiple columns, the labels in `.value`
        # should have every value in other
        # and in the same order
        # reindex ensures that, after getting a MultiIndex.from_product
        other = [entry for entry in names_to if entry != ".value"]
        columns = [*mapping.columns]
        others = mapping.loc[:, other].drop_duplicates()
        outcome = mapping.loc[:, ".value"].unique()
        if not mapping.duplicated().any(axis=None):
            df.columns = pd.MultiIndex.from_frame(mapping)
            indexer = {".value": outcome, "other": others}
        else:
            columns.append("".join(columns))
            cumcount = mapping.groupby(
                [*mapping.columns], sort=False, observed=True
            ).cumcount()
            df.columns = [arr for _, arr in mapping.items()] + [cumcount]
            indexer = {
                ".value": outcome,
                "other": others,
                "cumcount": cumcount.unique(),
            }
        indexer = _computations_expand_grid(indexer)
        indexer.columns = columns
        df = df.reindex(columns=indexer)
        df.columns = df.columns.get_level_values(".value")
        values = _dict_from_grouped_names(df=df)
        outcome = indexer.loc[indexer[".value"] == outcome[0], other]
        group_max = len(outcome)
        if names_transform:
            outcome = _names_transform(
                names_transform, is_dataframe=True, values=outcome
            )
        outcome = {
            name: extract_array(arr, extract_numpy=True).repeat(len_index)
            for name, arr in outcome.items()
        }

    return _final_frame_longer(
        df=df,
        len_index=len_index,
        reps=group_max,
        index=index,
        outcome=outcome,
        values=values,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
    )


def _headers_single_series(df: pd.DataFrame, mapping: pd.Series) -> tuple:
    """
    Extract headers and values for a single level.
    Applies to `.value` for a single level extract,
    or where names_pattern is a sequence.
    """
    # get positions of columns,
    # to ensure interleaving is possible
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
    outcome = mapping.groupby(mapping, sort=False, observed=True)
    group_size = outcome.size()
    group_max = group_size.max()
    # the number of groups should be the same;
    # if not, build a MultiIndex and reindex
    # to get equal numbers for each label
    if group_size.nunique() > 1:
        positions = outcome.cumcount()
        df.columns = [mapping, positions]
        indexer = group_size.index, np.arange(group_max)
        indexer = pd.MultiIndex.from_product(indexer)
        df = df.reindex(columns=indexer).droplevel(axis=1, level=1)
    else:
        df.columns = mapping
    outcome = _dict_from_grouped_names(df=df)
    return outcome, group_max


def _dict_from_grouped_names(df: pd.DataFrame) -> dict:
    """
    Create dictionary from multiple same names.
    Applicable when collating the values for `.value`,
    or when names_pattern is a list/tuple.
    """
    outcome = defaultdict(list)
    for num, name in enumerate(df.columns):
        arr = df.iloc[:, num]
        arr = extract_array(arr, extract_numpy=True)
        outcome[name].append(arr)
    return {name: concat_compat(arr) for name, arr in outcome.items()}


def _names_transform(
    names_transform, is_dataframe, values
) -> Union[pd.DataFrame, Generator]:
    """
    Convert column type using names_transform.

    If is_dataframe (values is a DataFrame),
    then astype is applied directly; otherwise an iteration.

    Returns either a DataFrame or a generator.
    """
    if is_dataframe:
        return values.astype(names_transform)
    if isinstance(names_transform, dict):
        outcome = (
            (key, arr.astype(names_transform[key]))
            for key, arr in values
            if key in names_transform
        )
    else:
        outcome = ((key, arr.astype(names_transform)) for key, arr in values)
    return outcome


def _sort_by_appearance_for_melt(
    df: pd.DataFrame, len_index: int
) -> pd.DataFrame:
    """
    This function sorts the resulting dataframe by appearance,
    via the `sort_by_appearance` parameter in `computations_pivot_longer`.

    A dataframe that is sorted by appearance is returned.
    """

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
    # and after -> 4.
    # reshaping allows us to track the original positions
    # in the previous dataframe ->
    # np.reshape([0,1,2,3], (-1, 2)) becomes
    # array([[0, 1],
    #        [2, 3]])
    # ravel, with the Fortran order (`F`) ensures the John's are aligned
    # before the Mary's -> [0, 2, 1, 3]
    # the flattened array is then passed to `take`
    if not length_check:
        index_sorter = np.arange(len(df))
        index_sorter = np.reshape(index_sorter, (-1, len_index))
        index_sorter = index_sorter.ravel(order="F")
        df = df.take(index_sorter)

    return df


def _final_frame_longer(
    df: pd.DataFrame,
    len_index: int,
    reps: int,
    index: Union[dict, None],
    outcome: dict,
    values: dict,
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    Build final dataframe for pivot_longer.
    """
    indexer = np.tile(np.arange(len_index), reps)
    df_index = df.index[indexer]
    if index:
        index = {name: arr[indexer] for name, arr in index.items()}
    else:
        index = {}
    df = {**index, **outcome, **values}

    df = pd.DataFrame(df, copy=False, index=df_index)

    if sort_by_appearance:
        df = _sort_by_appearance_for_melt(df=df, len_index=len_index)

    if ignore_index:
        df.index = range(len(df))

    if df.columns.names:
        df.columns.names = [None]

    return df


@pf.register_dataframe_method
def pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[list, str]] = None,
    names_from: Optional[Union[list, str]] = None,
    values_from: Optional[Union[list, str]] = None,
    flatten_levels: Optional[bool] = True,
    names_sep: str = "_",
    names_glue: str = None,
) -> pd.DataFrame:
    """
    Reshapes data from *long* to *wide* form.

    The number of columns are increased, while decreasing
    the number of rows. It is the inverse of the
    [`pivot_longer`][janitor.functions.pivot.pivot_longer]
    method, and is a wrapper around `pd.DataFrame.pivot` method.

    This method does not mutate the original DataFrame.

    Column selection in `index`, `names_from` and `values_from`
    is possible using the
    [`select_columns`][janitor.functions.select_columns.select_columns] syntax.

    A ValueError is raised if the combination
    of the `index` and `names_from` is not unique.

    By default, values from `values_from` are always
    at the top level if the columns are not flattened.
    If flattened, the values from `values_from` are usually
    at the start of each label in the columns.


    Example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = [{'dep': 5.5, 'step': 1, 'a': 20, 'b': 30},
        ...       {'dep': 5.5, 'step': 2, 'a': 25, 'b': 37},
        ...       {'dep': 6.1, 'step': 1, 'a': 22, 'b': 19},
        ...       {'dep': 6.1, 'step': 2, 'a': 18, 'b': 29}]
        >>> df = pd.DataFrame(df)
        >>> df
           dep  step   a   b
        0  5.5     1  20  30
        1  5.5     2  25  37
        2  6.1     1  22  19
        3  6.1     2  18  29

    Pivot and flatten columns:

        >>> df.pivot_wider(
        ...     index = "dep",
        ...     names_from = "step",
        ... )
           dep  a_1  a_2  b_1  b_2
        0  5.5   20   25   30   37
        1  6.1   22   18   19   29

    Modify columns with `names_sep`:

        >>> df.pivot_wider(
        ...     index = "dep",
        ...     names_from = "step",
        ...     names_sep = "",
        ... )
           dep  a1  a2  b1  b2
        0  5.5  20  25  30  37
        1  6.1  22  18  19  29

    Modify columns with `names_glue`:

        >>> df.pivot_wider(
        ...     index = "dep",
        ...     names_from = "step",
        ...     names_glue = "{_value}_step{step}",
        ... )
           dep  a_step1  a_step2  b_step1  b_step2
        0  5.5       20       25       30       37
        1  6.1       22       18       19       29

    :param df: A pandas DataFrame.
    :param index: Name(s) of columns to use as identifier variables.
        It should be either a single column name, or a list of column names.
        If `index` is not provided, the DataFrame's index is used.
    :param names_from: Name(s) of column(s) to use to make the new
        DataFrame's columns. Should be either a single column name,
        or a list of column names.
    :param values_from: Name(s) of column(s) that will be used for populating
        the new DataFrame's values.
        If `values_from` is not specified,  all remaining columns
        will be used.
    :param flatten_levels: Default is `True`. If `False`, the DataFrame stays
        as a MultiIndex.
    :param names_sep: If `names_from` or `values_from` contain multiple
        variables, this will be used to join the values into a single string
        to use as a column name. Default is `_`.
        Applicable only if `flatten_levels` is `True`.
    :param names_glue: A string to control the output of the flattened columns.
        It offers more flexibility in creating custom column names,
        and uses python's `str.format_map` under the hood.
        Simply create the string template,
        using the column labels in `names_from`,
        and special `_value` as a placeholder
        if there are multiple `values_from`.
        Applicable only if `flatten_levels` is `True`.
    :returns: A pandas DataFrame that has been unpivoted from long to wide
        form.
    """

    df = df.copy()

    return _computations_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        flatten_levels,
        names_sep,
        names_glue,
    )


def _computations_pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[list, str]] = None,
    names_from: Optional[Union[list, str]] = None,
    values_from: Optional[Union[list, str]] = None,
    flatten_levels: Optional[bool] = True,
    names_sep: str = "_",
    names_glue: str = None,
) -> pd.DataFrame:
    """
    This is the main workhorse of the `pivot_wider` function.

    It is a wrapper around `pd.pivot`.
    The output for multiple `names_from` and/or `values_from`
    can be controlled with `names_glue` and/or `names_sep`.

    A dataframe pivoted from long to wide form is returned.
    """

    (
        df,
        index,
        names_from,
        values_from,
        flatten_levels,
        names_sep,
        names_glue,
    ) = _data_checks_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        flatten_levels,
        names_sep,
        names_glue,
    )
    if flatten_levels:
        # check dtype of `names_from` is string
        names_from_all_strings = (
            df.filter(names_from).agg(is_string_dtype).all().item()
        )

        # check dtype of columns
        column_dtype = is_string_dtype(df.columns)

    df = df.pivot(  # noqa: PD010
        index=index, columns=names_from, values=values_from
    )

    # an empty df is likely because
    # there is no `values_from`
    if any((df.empty, not flatten_levels)):
        return df

    if isinstance(df.columns, pd.MultiIndex):
        if (not names_from_all_strings) or (not column_dtype):
            new_columns = [tuple(map(str, entry)) for entry in df]
        else:
            new_columns = [entry for entry in df]
        if names_glue is not None:
            if ("_value" in names_from) and (None in df.columns.names):
                warnings.warn(
                    "For names_glue, _value is used as a placeholder "
                    "for the values_from section. "
                    "However, there is a '_value' in names_from; "
                    "this might result in incorrect output. "
                    "If possible, kindly change the column label "
                    "from '_value' to another name, "
                    "to avoid erroneous results."
                )
            try:
                # there'll only be one None
                names_from = [
                    "_value" if ent is None else ent
                    for ent in df.columns.names
                ]
                new_columns = [
                    names_glue.format_map(dict(zip(names_from, entry)))
                    for entry in new_columns
                ]
            except KeyError as error:
                raise KeyError(
                    f"{error} is not a column label in names_from."
                ) from error
        else:
            if names_sep is None:
                names_sep = "_"
            new_columns = [names_sep.join(entry) for entry in new_columns]

        df.columns = new_columns
    else:
        if (not names_from_all_strings) or (not column_dtype):
            df.columns = df.columns.astype(str)
        if names_glue is not None:
            try:
                df.columns = [
                    names_glue.format_map({names_from[0]: entry})
                    for entry in df
                ]
            except KeyError as error:
                raise KeyError(
                    f"{error} is not a column label in names_from."
                ) from error

    # if columns are of category type
    # this returns columns to object dtype
    # also, resetting index with category columns is not possible
    df.columns = [*df.columns]
    if index:
        df = df.reset_index()

    if df.columns.names:
        df.columns.names = [None]

    return df


def _data_checks_pivot_wider(
    df,
    index,
    names_from,
    values_from,
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

    check("flatten_levels", flatten_levels, [bool])

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    if names_glue is not None:
        check("names_glue", names_glue, [str])

    return (
        df,
        index,
        names_from,
        values_from,
        flatten_levels,
        names_sep,
        names_glue,
    )
