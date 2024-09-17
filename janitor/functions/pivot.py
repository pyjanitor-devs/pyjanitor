from __future__ import annotations

import operator
import re
import warnings
from collections import defaultdict
from functools import reduce
from itertools import zip_longest
from typing import Callable, Pattern

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_extension_array_dtype, is_list_like
from pandas.core.dtypes.concat import concat_compat

from janitor.functions.select import (
    _index_converter,
    _select_index,
    get_index_labels,
)
from janitor.utils import check, refactored_function


@pf.register_dataframe_method
def pivot_longer(
    df: pd.DataFrame,
    index: list | tuple | str | Pattern = None,
    column_names: list | tuple | str | Pattern = None,
    names_to: list | tuple | str = None,
    values_to: str = "value",
    column_level: int | str = None,
    names_sep: str | Pattern = None,
    names_pattern: list | tuple | str | Pattern = None,
    names_transform: str | Callable | dict = None,
    dropna: bool = False,
    sort_by_appearance: bool = False,
    ignore_index: bool = True,
) -> pd.DataFrame:
    """Unpivots a DataFrame from *wide* to *long* format.

    This method does not mutate the original DataFrame.

    It is modeled after the `pivot_longer` function in R's tidyr package,
    and also takes inspiration from R's data.table package.

    This function is useful to massage a DataFrame into a format where
    one or more columns are considered measured variables, and all other
    columns are considered as identifier variables.

    All measured variables are *unpivoted* (and typically duplicated) along the
    row axis.

    Column selection in `index` and `column_names` is possible using the
    [`select`][janitor.functions.select.select] syntax.

    For more granular control on the unpivoting, have a look at
    [`pivot_longer_spec`][janitor.functions.pivot.pivot_longer_spec].

    Examples:
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

        Convenient, flexible column selection in the `index` via the
        [`select`][janitor.functions.select.select] syntax:
        >>> from pandas.api.types import is_string_dtype
        >>> df.pivot_longer(index = is_string_dtype)
             Species      variable  value
        0     setosa  Sepal.Length    5.1
        1  virginica  Sepal.Length    5.9
        2     setosa   Sepal.Width    3.5
        3  virginica   Sepal.Width    3.0
        4     setosa  Petal.Length    1.4
        5  virginica  Petal.Length    5.1
        6     setosa   Petal.Width    0.2
        7  virginica   Petal.Width    1.8

        Split the column labels into individual columns:
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

        Split the column labels for the above dataframe using named groups in `names_pattern`:
        >>> df.pivot_longer(
        ...     index = 'id',
        ...     names_pattern = r"new_?(?P<diagnosis>.+)_(?P<gender>.)(?P<age>\\d+)",
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
        diagnosis   object
        gender    category
        age          int64
        value        int64
        dtype: object

        Use multiple `.value` to reshape the dataframe:
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

        Replicate the above with named groups in `names_pattern` - use `_` instead of `.value`:
        >>> df.pivot_longer(
        ...     index="unit",
        ...     names_pattern=r"(?P<_>x|y)_(?P<time>[0-9])(?P<__>_mean)",
        ... )
           unit time  x_mean  y_mean
        0    50    1      10      30
        1    50    2      20      40

        Convenient, flexible column selection in the `column_names` via
        the [`select`][janitor.functions.select.select] syntax:
        >>> df.pivot_longer(
        ...     column_names="*mean",
        ...     names_to=(".value", "time", ".value"),
        ...     names_pattern=r"(x|y)_([0-9])(_mean)",
        ... )
           unit time  x_mean  y_mean
        0    50    1      10      30
        1    50    2      20      40

        >>> df.pivot_longer(
        ...     column_names=slice("x_1_mean", "y_2_mean"),
        ...     names_to=(".value", "time", ".value"),
        ...     names_pattern=r"(x|y)_([0-9])(_mean)",
        ... )
           unit time  x_mean  y_mean
        0    50    1      10      30
        1    50    2      20      40

        Reshape the dataframe by passing a sequence to `names_pattern`:
        >>> df = pd.DataFrame({'hr1': [514, 573],
        ...                    'hr2': [545, 526],
        ...                    'team': ['Red Sox', 'Yankees'],
        ...                    'year1': [2007, 2007],
        ...                    'year2': [2008, 2008]})
        >>> df
           hr1  hr2     team  year1  year2
        0  514  545  Red Sox   2007   2008
        1  573  526  Yankees   2007   2008
        >>> df.pivot_longer(
        ...     index = 'team',
        ...     names_to = ['year', 'hr'],
        ...     names_pattern = ['year', 'hr']
        ... )
              team   hr  year
        0  Red Sox  514  2007
        1  Yankees  573  2007
        2  Red Sox  545  2008
        3  Yankees  526  2008


        Reshape the above dataframe by passing a dictionary to `names_pattern`:
        >>> df.pivot_longer(
        ...     index = 'team',
        ...     names_pattern = {"year":"year", "hr":"hr"}
        ... )
              team   hr  year
        0  Red Sox  514  2007
        1  Yankees  573  2007
        2  Red Sox  545  2008
        3  Yankees  526  2008

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
        ...         names_pattern=["M|O|W", "G|V"],
        ...     )
              City    State       Fruit  Drink  Pounds  Ounces
        0  Houston    Texas       Mango    Gin       4    16.0
        1   Austin    Texas       Mango    Gin      10   200.0
        2   Hoover  Alabama       Mango    Gin      90    34.0
        3  Houston    Texas      Orange  Vodka      10    20.0
        4   Austin    Texas      Orange  Vodka       8    33.0
        5   Hoover  Alabama      Orange  Vodka      14    18.0
        6  Houston    Texas  Watermelon   None      40     NaN
        7   Austin    Texas  Watermelon   None      99     NaN
        8   Hoover  Alabama  Watermelon   None      43     NaN

        Replicate the above transformation with a nested dictionary passed to `names_pattern`
        - the outer keys in the `names_pattern` dictionary are passed to `names_to`,
        while the inner keys are passed to `values_to`:
        >>> df.pivot_longer(
        ...     index=["City", "State"],
        ...     column_names=slice("Mango", "Vodka"),
        ...     names_pattern={
        ...         "Fruit": {"Pounds": "M|O|W"},
        ...         "Drink": {"Ounces": "G|V"},
        ...     },
        ... )
              City    State       Fruit  Drink  Pounds  Ounces
        0  Houston    Texas       Mango    Gin       4    16.0
        1   Austin    Texas       Mango    Gin      10   200.0
        2   Hoover  Alabama       Mango    Gin      90    34.0
        3  Houston    Texas      Orange  Vodka      10    20.0
        4   Austin    Texas      Orange  Vodka       8    33.0
        5   Hoover  Alabama      Orange  Vodka      14    18.0
        6  Houston    Texas  Watermelon   None      40     NaN
        7   Austin    Texas  Watermelon   None      99     NaN
        8   Hoover  Alabama  Watermelon   None      43     NaN

    !!! abstract "Version Changed"

        - 0.24.0
            - Added `dropna` parameter.
        - 0.24.1
            - `names_pattern` can accept a dictionary.
            - named groups supported in `names_pattern`.

    Args:
        df: A pandas DataFrame.
        index: Name(s) of columns to use as identifier variables.
            Should be either a single column name, or a list/tuple of
            column names.
            `index` should be a list of tuples if the columns are a MultiIndex.
        column_names: Name(s) of columns to unpivot. Should be either
            a single column name or a list/tuple of column names.
            `column_names` should be a list of tuples
            if the columns are a MultiIndex.
        names_to: Name of new column as a string that will contain
            what were previously the column names in `column_names`.
            The default is `variable` if no value is provided. It can
            also be a list/tuple of strings that will serve as new column
            names, if `name_sep` or `names_pattern` is provided.
            If `.value` is in `names_to`, new column names will be extracted
            from part of the existing column names and overrides `values_to`.
        values_to: Name of new column as a string that will contain what
            were previously the values of the columns in `column_names`.
            values_to can also be a list/tuple
            and requires that names_pattern is also a list/tuple.
        column_level: If columns are a MultiIndex, then use this level to
            unpivot the DataFrame. Provided for compatibility with pandas' melt,
            and applies only if neither `names_sep` nor `names_pattern` is
            provided.
        names_sep: Determines how the column name is broken up, if
            `names_to` contains multiple values. It takes the same
            specification as pandas' `str.split` method, and can be a string
            or regular expression. `names_sep` does not work with MultiIndex
            columns.
        names_pattern: Determines how the column name is broken up.
            It can be a regular expression containing matching groups.
            Under the hood it is processed with pandas' `str.extract` function.
            If it is a single regex, the number of groups must match
            the length of `names_to`.
            Named groups are supported, if `names_to` is none. `_` is used
            instead of `.value` as a placeholder in named groups.
            `_` can be overloaded for multiple `.value`
            calls - `_`, `__`, `___`, ...
            `names_pattern` can also be a list/tuple of regular expressions
            It can also be a list/tuple of strings;
            the strings will be treated as regular expressions.
            Under the hood it is processed with pandas' `str.contains` function.
            For a list/tuple of regular expressions,
            `names_to` must also be a list/tuple and the lengths of both
            arguments must match.
            `names_pattern` can also be a dictionary, where the keys are
            the new column names, while the values can be a regular expression
            or a string which will be evaluated as a regular expression.
            Alternatively, a nested dictionary can be used, where the sub
            key(s) are associated with `values_to`. Please have a look
            at the examples for usage.
            `names_pattern` does not work with MultiIndex columns.
        names_transform: Use this option to change the types of columns that
            have been transformed to rows. This does not applies to the values' columns.
            Accepts any argument that is acceptable by `pd.astype`.
        dropna: Determines whether or not to drop nulls
            from the values columns. Default is `False`.
        sort_by_appearance: Boolean value that determines
            the final look of the DataFrame. If `True`, the unpivoted DataFrame
            will be stacked in order of first appearance.
        ignore_index: If `True`,
            the original index is ignored. If `False`, the original index
            is retained and the index labels will be repeated as necessary.

    Returns:
        A pandas DataFrame that has been unpivoted from wide to long
            format.
    """  # noqa: E501

    # this code builds on the wonderful work of @benjaminjack’s PR
    # https://github.com/benjaminjack/pyjanitor/commit/e3df817903c20dd21634461c8a92aec137963ed0

    return _computations_pivot_longer(
        df=df,
        index=index,
        column_names=column_names,
        column_level=column_level,
        names_to=names_to,
        values_to=values_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
        names_transform=names_transform,
        dropna=dropna,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
    )


def pivot_longer_spec(
    df: pd.DataFrame,
    spec: pd.DataFrame,
    sort_by_appearance: bool = False,
    ignore_index: bool = True,
    dropna: bool = False,
    df_columns_is_unique: bool = True,
) -> pd.DataFrame:
    """A declarative interface to pivot a DataFrame from wide to long form,
    where you describe how the data will be unpivoted,
    using a DataFrame. This gives you, the user,
    more control over unpivoting, where you create a “spec”
    data frame that describes exactly how data stored
    in the column names becomes variables.
    It can come in handy for situations where
    [`pivot_longer`][janitor.functions.pivot.pivot_longer]
    seems inadequate for the transformation.

    !!! info "New in version 0.28.0"

    Examples:
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
        >>> spec = {'.name':['Sepal.Length','Petal.Length',
        ...                  'Sepal.Width','Petal.Width'],
        ...         '.value':['Length','Length','Width','Width'],
        ...         'part':['Sepal','Petal','Sepal','Petal']}
        >>> spec = pd.DataFrame(spec)
        >>> spec
                  .name  .value   part
        0  Sepal.Length  Length  Sepal
        1  Petal.Length  Length  Petal
        2   Sepal.Width   Width  Sepal
        3   Petal.Width   Width  Petal
        >>> pivot_longer_spec(df=df,spec=spec)
             Species   part  Length  Width
        0     setosa  Sepal     5.1    3.5
        1  virginica  Sepal     5.9    3.0
        2     setosa  Petal     1.4    0.2
        3  virginica  Petal     5.1    1.8

    Args:
        df: The source DataFrame to unpivot.
        spec: A specification DataFrame.
            At a minimum, the spec DataFrame
            must have a '.name' and a '.value' columns.
            The '.name' column  should contain the
            columns in the source DataFrame that will be
            transformed to long form.
            The '.value' column gives the name of the column(s)
            that the values in the source DataFrame will go into.
            Additional columns in spec should be named to match columns
            in the long format of the dataset and contain values
            corresponding to columns pivoted from the wide format.
            Note that these additional columns should not already exist
            in the source DataFrame.
        sort_by_appearance: Boolean value that determines
            the final look of the DataFrame. If `True`, the unpivoted DataFrame
            will be stacked in order of first appearance.
        ignore_index: If `True`,
            the original index is ignored. If `False`, the original index
            is retained and the index labels will be repeated as necessary.
        dropna: Determines whether or not to drop nulls
            from the values columns. Default is `False`.
        df_columns_is_unique: Boolean value to indicate if the source
            DataFrame's columns is unique. Default is `True`.

    Raises:
        KeyError: If '.name' or '.value' is missing from the spec's columns.
        ValueError: If the spec's columns is not unique,
            or the labels in spec['.name'] is not unique.

    Returns:
        A pandas DataFrame.
    """
    check("spec", spec, [pd.DataFrame])
    if not spec.columns.is_unique:
        raise ValueError("Kindly ensure the spec's columns is unique.")
    if ".name" not in spec.columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.name` column."
        )
    if ".value" not in spec.columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.value` column."
        )
    if spec.columns.tolist()[:2] != [".name", ".value"]:
        raise ValueError(
            "The first two columns of the spec DataFrame "
            "should be '.name' and '.value', "
            "with '.name' coming before '.value'."
        )
    if not spec[".name"].is_unique:
        raise ValueError("The labels in the `.name` column should be unique.")

    exclude = df.columns.intersection(spec.columns)
    if not exclude.empty:
        raise ValueError(
            f"Labels {*exclude, } in the spec DataFrame already exist "
            "as column labels in the source DataFrame. "
            "Kindly ensure the spec DataFrame's columns "
            "are not present in the source DataFrame."
        )

    check("dropna", dropna, [bool])
    check("sort_by_appearance", sort_by_appearance, [bool])
    check("ignore_index", ignore_index, [bool])
    check("df_columns_is_unique", df_columns_is_unique, [bool])

    index = df.columns.difference(spec[".name"], sort=False)
    index = {name: df[name]._values for name in index}

    df = df.loc[:, spec[".name"]]
    if not df_columns_is_unique:
        spec = pd.DataFrame({".name": df.columns}).merge(
            spec, on=".name", how="inner"
        )
    others = [label for label in spec if label not in {".name", ".value"}]
    return _pivot_longer_dot_value(
        df=df,
        spec=spec.drop(columns=".name"),
        index=index,
        others=others,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
        dropna=dropna,
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
    dropna,
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
    # checks here are only on the columns
    # a slice is safe
    df = df[:]

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

    if (index is None) and (column_names is None):
        column_names = slice(None)
        index = []

    elif (index is not None) and (column_names is not None):
        column_names = _select_index([column_names], df, axis="columns")
        index = _select_index([index], df, axis="columns")
        index = df.columns[index]

    elif (index is None) and (column_names is not None):
        column_names = _select_index([column_names], df, axis="columns")
        index = _index_converter(column_names, df.columns)
        index = pd.Index(pd.unique(index))
        index = index.get_indexer(range(df.columns.size)) == -1
        index = df.columns[index]

    elif (index is not None) and (column_names is None):
        index = _select_index([index], df, axis="columns")
        column_names = _index_converter(index, df.columns)
        column_names = pd.Index(pd.unique(column_names))
        column_names = column_names.get_indexer(range(df.columns.size)) == -1
        column_names = column_names.nonzero()[0]
        if not column_names.size:
            column_names = None
        index = df.columns[index]

    if column_names is None:
        return None
    # ideally index names should be unique
    # it is the column_names that may be duplicated
    # since we are ultimately flipping them into long form
    index = {name: df[name]._values for name in index}
    df = df.iloc[:, column_names]

    if names_to is not None:
        if isinstance(names_to, str):
            names_to = [names_to]
        elif isinstance(names_to, tuple):
            names_to = [*names_to]

        check("names_to", names_to, [list, str, tuple])

        uniques = set()
        for word in names_to:
            check(f"'{word}' in names_to", word, [str])
            if (word in uniques) and (word != ".value"):
                raise ValueError(f"'{word}' is duplicated in names_to.")
            uniques.add(word)

    else:
        if not any((names_sep, names_pattern)):
            names_to = ["variable"]

    check("values_to", values_to, [str, list, tuple])
    if isinstance(values_to, (list, tuple)):
        if not isinstance(names_pattern, (list, tuple)):
            raise TypeError(
                "values_to can be a list/tuple only "
                "if names_pattern is a list/tuple."
            )
        if index:
            exclude = set(values_to).intersection(index)
            if exclude:
                raise ValueError(
                    f"Labels {*exclude, } in values_to already exist as "
                    "column labels assigned to the dataframe's "
                    "index parameter. Kindly use unique labels."
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
        check(
            "names_pattern", names_pattern, [str, Pattern, list, tuple, dict]
        )
        if isinstance(names_pattern, (str, Pattern)):
            regex = re.compile(names_pattern)
            if names_to is None:
                if regex.groupindex:
                    names_to = regex.groupindex.keys()
                    names_to = [
                        (
                            ".value"
                            if ("_" in name) and (len(set(name)) == 1)
                            else name
                        )
                        for name in names_to
                    ]
                    len_names_to = len(names_to)
                else:
                    raise ValueError("Kindly provide values for names_to.")
            else:
                len_names_to = len(names_to)
            if len_names_to != regex.groups:
                raise ValueError(
                    f"The length of names_to does not match "
                    "the number of groups in names_pattern. "
                    f"The length of names_to is {len_names_to} "
                    "while the number of groups in the regex "
                    f"is {regex.groups}."
                )

        elif isinstance(names_pattern, (list, tuple)):
            if names_to is None:
                raise ValueError("Kindly provide values for names_to.")
            for word in names_pattern:
                check(f"'{word}' in names_pattern", word, [str, Pattern])
            len_names_to = len(names_to)
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
                            f"'{word}' in values_to "
                            "already exists in names_to."
                        )

                    if word in uniques:
                        raise ValueError(
                            f"'{word}' is duplicated in values_to."
                        )
                    uniques.add(word)
        # outer keys belong to names_to
        # if the values are dicts,
        # then the inner key belongs to values_to
        # inner keys should not exist in the outer keys
        # non keys belong to names_pattern
        elif isinstance(names_pattern, dict):
            if names_to is not None:
                raise ValueError(
                    "names_to should be None "
                    "when names_pattern is a dictionary"
                )
            for key, value in names_pattern.items():
                check(f"'{key}' in names_pattern", key, [str])
                if index and (key in index):
                    raise ValueError(
                        f"'{key}' in the names_pattern dictionary "
                        "already exists as a column label "
                        "assigned to the index parameter. "
                        "Kindly use a unique name"
                    )
            names_to = list(names_pattern)
            is_dict = (
                isinstance(arg, dict) for _, arg in names_pattern.items()
            )
            if all(is_dict):
                values_to = []
                patterns = []
                for key, value in names_pattern.items():
                    if len(value) != 1:
                        raise ValueError(
                            "The length of the dictionary paired "
                            f"with '{key}' in names_pattern "
                            "should be length 1, instead got "
                            f"{len(value)}"
                        )
                    for k, v in value.items():
                        if not isinstance(k, str):
                            raise TypeError(
                                "The key in the nested dictionary "
                                f"for '{key}' in names_pattern "
                                "should be a string, instead got {type(k)}"
                            )
                        if k in names_pattern:
                            raise ValueError(
                                f"'{k}' in the nested dictionary "
                                "already exists as one of the main "
                                "keys in names_pattern"
                            )
                        if index and (k in index):
                            raise ValueError(
                                f"'{k}' in the nested dictionary "
                                "already exists as a column label "
                                "assigned to the index parameter. "
                                "Kindly use a unique name"
                            )
                        check(
                            f"The value paired with '{k}' "
                            "in the nested dictionary in names_pattern",
                            v,
                            [str, Pattern],
                        )
                        patterns.append(v)
                        values_to.append(k)
            else:
                patterns = []
                for key, value in names_pattern.items():
                    check(
                        f"The value paired with '{key}' "
                        "in the names_pattern dictionary",
                        value,
                        [str, Pattern],
                    )

                    patterns.append(value)
            names_pattern = patterns
            patterns = None

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

    check("dropna", dropna, [bool])

    check("sort_by_appearance", sort_by_appearance, [bool])

    check("ignore_index", ignore_index, [bool])

    if isinstance(df.columns, pd.MultiIndex):
        if not any(df.columns.names):
            if len(names_to) == 1:
                names = [
                    f"{names_to[0]}_{i}" for i in range(df.columns.nlevels)
                ]
                df.columns = df.columns.set_names(names)
            elif len(names_to) == df.columns.nlevels:
                df.columns = df.columns.set_names(names_to)
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
        df.columns = pd.Index(df.columns, name=names_to[0])

    return (
        df,
        index,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        names_transform,
        dropna,
        sort_by_appearance,
        ignore_index,
    )


def _computations_pivot_longer(
    df: pd.DataFrame,
    index: list | tuple | str | Pattern | None,
    column_names: list | tuple | str | Pattern | None,
    names_to: list | tuple | str | None,
    values_to: str,
    column_level: int | str,
    names_sep: str | Pattern,
    names_pattern: list | tuple | str | Pattern | None,
    names_transform: str | Callable | dict | None,
    dropna: bool,
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
    # if sort_by_appearance is True, then the index will be repeated,
    # while the rest will be tiled ------------>
    # if index is [1,2,3] then repeats [1,1,1,2,2,2,3,3,3]
    # if column_names is [1,2,3] then tiling makes it [1,2,3,1,2,3,...]
    # for performance reasons, we dump down into arrays,
    # and build a new dataframe, with copy = False
    # since we have already made a copy of the original df

    checks = _data_checks_pivot_longer(
        df=df,
        index=index,
        column_names=column_names,
        column_level=column_level,
        names_to=names_to,
        values_to=values_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
        names_transform=names_transform,
        dropna=dropna,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
    )

    if checks is None:
        return df

    (
        df,
        index,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        names_transform,
        dropna,
        sort_by_appearance,
        ignore_index,
    ) = checks

    if all((names_pattern is None, names_sep is None)):
        spec = df.columns
        others = spec.names
        spec = {name: spec.get_level_values(name) for name in spec.names}
        spec = pd.DataFrame(spec, copy=False)
        spec[".value"] = values_to
        if names_transform is not None:
            spec = _names_transform(
                spec=spec, others=others, names_transform=names_transform
            )
        return _pivot_longer_dot_value(
            df=df,
            index=index,
            others=others,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
            dropna=dropna,
            spec=spec,
        )

    if names_sep is not None:
        return _pivot_longer_names_sep(
            df=df,
            index=index,
            names_to=names_to,
            names_sep=names_sep,
            names_transform=names_transform,
            values_to=values_to,
            dropna=dropna,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
        )

    if isinstance(names_pattern, (str, Pattern)):
        return _pivot_longer_names_pattern_str(
            df=df,
            index=index,
            names_to=names_to,
            names_pattern=names_pattern,
            names_transform=names_transform,
            values_to=values_to,
            dropna=dropna,
            sort_by_appearance=sort_by_appearance,
            ignore_index=ignore_index,
        )

    if isinstance(values_to, (list, tuple)):
        return _pivot_longer_values_to_sequence(
            df=df,
            index=index,
            names_to=names_to,
            names_pattern=names_pattern,
            names_transform=names_transform,
            dropna=dropna,
            sort_by_appearance=sort_by_appearance,
            values_to=values_to,
            ignore_index=ignore_index,
        )

    return _pivot_longer_names_pattern_sequence(
        df=df,
        index=index,
        names_to=names_to,
        names_pattern=names_pattern,
        dropna=dropna,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
    )


def _pivot_longer_values_to_sequence(
    df: pd.DataFrame,
    index: dict,
    names_to: list,
    names_pattern: list | tuple,
    names_transform: str | Callable | dict | None,
    dropna: bool,
    sort_by_appearance: bool,
    values_to: list | tuple,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    values_to is a list/tuple.
    """
    columns = df.columns
    booleans = [
        columns.str.contains(regex, na=False, regex=True)
        for regex in names_pattern
    ]
    # within each match, check the individual matches
    # and raise an error if any is False
    boolean_masks = (arr.any().item() for arr in booleans)
    boolean_masks = zip(names_pattern, boolean_masks)
    boolean_masks = (name for name, mask in boolean_masks if not mask)
    for pattern in boolean_masks:
        raise ValueError(f"No match was returned for the regex {pattern}")
    values = np.select(booleans, values_to, None)
    columns = np.select(booleans, names_to, None)
    # only matched columns are retained
    booleans = pd.notna(values)
    df = df.loc[:, booleans]
    values = values[booleans]
    columns = columns[booleans]
    # the aim is to ensure that values_to and names_to
    # ultimately have the same number of entries
    # let's take an example from SO
    # https://stackoverflow.com/q/51519101/7175713
    # In [6]: multiple_values_to
    #       City    State      Name  Mango  Orange  Watermelon  Gin  Vodka
    # 0  Houston    Texas      Aria      4      10          40   16     20
    # 1   Austin    Texas  Penelope     10       8          99  200     33
    # 2   Hoover  Alabama      Niko     90      14          43   34     18

    ###
    #  multiple_values_to.pivot_longer(
    #     index=["City", "State"],
    #     column_names=slice("Mango", "Vodka"),
    #     names_to=("Fruit", "Drink"),
    #     values_to=("Pounds", "Ounces"),
    #     names_pattern=[r"M|O|W", r"G|V"],
    #     names_transform={"Fruit": "category", "Drink": "category"},
    # )

    # from the above, we can see the expected pairing
    # values_to > names_to > column_names
    # pounds->fruits->[mango, orange, watermelon]
    # ounces->drink->[gin, vodka]
    # there are only two columns for ounces, drink
    # compared to three columns for pounds, fruit
    # we need to get ounces,drink to have three entries
    # in the final spec DataFrame
    # to match pounds, fruit
    # that is what the code below covers
    # with a combination of zip and zip_longest
    # pairing the columns with zip_longest we get:
    # [(mango, gin), (orange, vodka), (watermelon, None)]
    # we then pair appropriately with (pounds, ounces)
    # and (fruit, drink)
    # which ultimately gives us this spec:
    #    .value       Fruit  Drink
    # 0  Pounds       Mango    Gin
    # 1  Pounds      Orange  Vodka
    # 2  Pounds  Watermelon   None
    # 3  Ounces       Mango    Gin
    # 4  Ounces      Orange  Vodka
    # 5  Ounces  Watermelon   None
    data = defaultdict(list)
    headers = defaultdict(int)
    for value, col, cols in zip(values, columns, df.columns):
        data[col].append(cols)
        headers[value] += 1
    keys = list(headers.keys())
    max_size = max(headers.values())
    headers = np.repeat(keys, max_size)
    keys = data.keys()
    data = data.values()
    data = zip_longest(*data)
    data = zip(*data)
    data = map(np.array, data)
    data = dict(zip(keys, data))
    length = len(keys)
    shape = (max_size, length)
    indexer = np.empty(shape=shape, dtype=np.intp)
    arr = np.arange(max_size).reshape((max_size, 1))
    indexer[:] = arr
    indexer = indexer.ravel(order="F")
    spec = {".value": headers}
    for key, value in data.items():
        value = value[indexer]
        spec[key] = value
    spec = pd.DataFrame(spec, copy=False)
    others = list(keys)
    if names_transform is not None:
        spec = _names_transform(
            spec=spec, others=others, names_transform=names_transform
        )
    return _pivot_longer_dot_value(
        df=df,
        index=index,
        others=others,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
        dropna=dropna,
        spec=spec,
    )


def _pivot_longer_names_pattern_sequence(
    df: pd.DataFrame,
    index: dict,
    names_to: list,
    names_pattern: list | tuple,
    dropna: bool,
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_pattern is provided, and is a list/tuple.
    """
    columns = df.columns
    booleans = [
        columns.str.contains(regex, na=False, regex=True)
        for regex in names_pattern
    ]
    # within each match, check the individual matches
    # and raise an error if any is False
    boolean_masks = (arr.any().item() for arr in booleans)
    boolean_masks = zip(names_pattern, boolean_masks)
    boolean_masks = (name for name, mask in boolean_masks if not mask)
    for pattern in boolean_masks:
        raise ValueError(f"No match was returned for the regex {pattern}")
    values = np.select(booleans, names_to, None)
    # only matched columns are retained
    booleans = pd.notna(values)
    df = df.loc[:, booleans]
    values = values[booleans]
    spec = {".value": values}
    spec = pd.DataFrame(spec, copy=False)
    return _pivot_longer_dot_value(
        df=df,
        index=index,
        others=None,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
        dropna=dropna,
        spec=spec,
    )


def _pivot_longer_names_pattern_str(
    df: pd.DataFrame,
    index: dict,
    names_to: list,
    names_pattern: str | Pattern,
    names_transform: bool,
    values_to: str,
    dropna: bool,
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_pattern is provided, and is a string/regex.
    """
    spec = df.columns.str.extract(names_pattern, expand=True)

    nulls_found = spec.isna()

    if nulls_found.any(axis=None):
        no_match = df.columns[nulls_found.any(axis="columns")]
        raise ValueError(
            f"Column labels {*no_match,} "
            "could not be matched with any of the groups "
            "in the provided regex. Kindly provide a regular expression "
            "(with the correct groups) that matches all labels in the columns."
        )

    if ".value" in names_to:
        spec, index, others = _dot_value_extra_checks(
            index=index, names_to=names_to, spec=spec
        )
    else:
        spec.columns = names_to
        spec[".value"] = values_to
        others = [name for name in spec if name != ".value"]
    if names_transform is not None:
        spec = _names_transform(
            spec=spec, others=others, names_transform=names_transform
        )
    return _pivot_longer_dot_value(
        df=df,
        index=index,
        others=others,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
        dropna=dropna,
        spec=spec,
    )


def _pivot_longer_names_sep(
    df: pd.DataFrame,
    index: dict,
    names_to: list,
    names_sep: str | Pattern,
    values_to: str,
    names_transform: str | dict | Callable | None,
    dropna: bool,
    sort_by_appearance: bool,
    ignore_index: bool,
) -> pd.DataFrame:
    """
    This takes care of pivoting scenarios where
    names_sep is provided.
    """

    spec = pd.Series(df.columns).str.split(names_sep, expand=True)
    len_spec_columns = len(spec.columns)
    len_names_to = len(names_to)

    if len_names_to != len_spec_columns:
        raise ValueError(
            f"The length of names_to does not match "
            "the number of levels extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of levels extracted is "
            f"{len_spec_columns}."
        )

    if ".value" in names_to:
        spec, index, others = _dot_value_extra_checks(
            index=index, names_to=names_to, spec=spec
        )
    else:
        spec.columns = names_to
        spec[".value"] = values_to
        others = [name for name in spec if name != ".value"]
    if names_transform is not None:
        spec = _names_transform(
            spec=spec, others=others, names_transform=names_transform
        )
    return _pivot_longer_dot_value(
        df=df,
        index=index,
        others=others,
        sort_by_appearance=sort_by_appearance,
        ignore_index=ignore_index,
        dropna=dropna,
        spec=spec,
    )


def _pivot_longer_dot_value(
    df: pd.DataFrame,
    spec: pd.DataFrame,
    index: dict,
    others: list,
    sort_by_appearance: bool,
    ignore_index: bool,
    dropna: bool,
) -> pd.DataFrame:
    """
    Pivots the dataframe into the final form,
    for scenarios where names_pattern is a string/regex,
    or names_sep is provided, and .value is in names_to.

    Returns a DataFrame.
    """
    if not others:
        contents, reps = _stack_dot_value_only(
            spec=spec,
            df=df,
            sort_by_appearance=sort_by_appearance,
        )
        index, _, df_index = _stack_non_dot_value(
            spec=None,
            reps=reps,
            df=df,
            index=index,
            ignore_index=ignore_index,
            sort_by_appearance=sort_by_appearance,
        )
        df = {**index, **contents}
        df = pd.DataFrame(data=df, index=df_index, copy=False)
        if dropna:
            df = _pivot_longer_dropna(
                df=df, contents=contents, ignore_index=ignore_index
            )
        return df

    if spec.duplicated().any(axis=None):
        raise ValueError("spec contains duplicate entries, cannot reshape.")

    spec, contents = _stack_dot_value(
        spec=spec,
        others=others,
        df=df,
        sort_by_appearance=sort_by_appearance,
    )

    index, spec, df_index = _stack_non_dot_value(
        spec=spec,
        reps=len(spec),
        df=df,
        index=index,
        ignore_index=ignore_index,
        sort_by_appearance=sort_by_appearance,
    )

    df = {**index, **spec, **contents}
    df = pd.DataFrame(data=df, index=df_index, copy=False)
    if dropna:
        df = _pivot_longer_dropna(
            df=df, contents=contents, ignore_index=ignore_index
        )
    return df


def _dot_value_extra_checks(
    index: dict,
    names_to: list,
    spec: pd.DataFrame,
):
    """
    Extra checks if '.value' is present
    in names_to.
    """

    others = [name for name in names_to if name != ".value"]
    if names_to.count(".value") > 1:
        arrays = [
            pos
            for pos, name in zip(spec.columns, names_to)
            if name == ".value"
        ]
        arrays = [spec.pop(pos) for pos in arrays]
        arr = reduce(operator.add, arrays)
        if others:
            spec.columns = others
        spec[".value"] = arr
    else:
        spec.columns = names_to

    dot_value = spec[".value"]
    exclude = {
        word
        for word in dot_value.array
        if (word in names_to) and (word != ".value")
    }
    if exclude:
        raise ValueError(
            f"Labels {*exclude, } in names_to already exist "
            "in the new dataframe's columns. "
            "Kindly provide unique label(s)."
        )

    if index:
        exclude = set(index).intersection(dot_value.array)
        if exclude:
            raise ValueError(
                f"Labels {*exclude, } already exist "
                "as column labels assigned to the dataframe's "
                "index parameter. Kindly provide unique label(s)."
            )
    return spec, index, others


def _stack_dot_value_only(
    spec: pd.DataFrame,
    df: pd.DataFrame,
    sort_by_appearance: bool,
) -> tuple:
    """
    Flip the .value into long form.
    Applicable when only .value column exists in spec
    """
    if spec.nunique().item() == 1:
        return _stack_dot_value_only_single_label(
            spec=spec, df=df, sort_by_appearance=sort_by_appearance
        )
    return _stack_dot_value_only_multiple_labels(
        spec=spec, df=df, sort_by_appearance=sort_by_appearance
    )


def _stack_dot_value_only_single_label(
    spec: pd.DataFrame,
    df: pd.DataFrame,
    sort_by_appearance: bool,
) -> tuple:
    """
    Flip the .value into long form.
    Applicable when only .value column exists in spec,
    and .value.unique==1
    """
    reps = len(spec)
    if df.dtypes.map(is_extension_array_dtype).any(axis=None):
        contents = [arr._values for _, arr in df.items()]
        contents = concat_compat(contents)
        if sort_by_appearance:
            length = contents.size
            indexer = np.arange(length)
            indexer = indexer.reshape((reps, -1))
            indexer = indexer.ravel(order="F")
            contents = contents[indexer]
    else:
        if sort_by_appearance:
            contents = df._values.ravel()
        else:
            contents = df._values.ravel(order="F")
    key = spec[".value"]._values[0]
    contents = {key: contents}
    return contents, reps


def _stack_dot_value_only_multiple_labels(
    spec: pd.DataFrame,
    df: pd.DataFrame,
    sort_by_appearance: bool,
) -> tuple:
    """
    Flip the .value into long form.
    Applicable when only .value column exists in spec
    and .value.nunique > 1
    """
    # summary of implementation logic:
    # for this usecase, only .value exists,
    # there is no `ohers` ->
    # spec:
    # .value .name
    #     x     x1
    #     x     x2
    #     y     y1
    #     y     y2
    #
    # in the spec dataframe above,
    # we can see the pairing between .value and .name
    # we can also observe that there are two unique labels
    # in .value -> x and y
    # what we need to ensure is that when creating the long form
    # both x and y have the same number of entries
    # i.e if there are 2 counts of x and 3 counts of y,
    # then the final dataframe must have 3 counts of x
    # and 3 counts of y (the max size is what determines the final count)
    # if the counts of x do not match the counts of y,
    # then the label with the lesser size is augmented with
    # an array of nans
    # also, since pandas supports duplicate columns
    # we get the positions of columns,
    # instead of the column names.
    # the column labels are interleaved
    # on a first come first serve approach
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
    # this is because `others` does not exist here -
    # `others` would have acted as a guard/combiner
    grouped = spec.groupby(".value", sort=False, dropna=False, observed=True)
    grouping = grouped.size()
    reps = grouping.max()
    missing = grouping[grouping != reps]
    if not missing.empty:
        null_array = np.full(shape=len(df), fill_value=np.nan)
    else:
        null_array = None
    mapp = defaultdict(list)
    df.columns = range(df.columns.size)
    for header, pos in zip(spec[".value"], df.columns):
        mapp[header].append(pos)
    zipped = zip_longest(*mapp.values())
    zipped = zip(*zipped)
    zipped = zip(mapp, zipped)
    contents = defaultdict(list)
    for label, positions in zipped:
        for position in positions:
            arr = df.get(position, null_array)
            if isinstance(arr, pd.Series):
                arr = arr._values
            contents[label].append(arr)
    contents = {label: concat_compat(arr) for label, arr in contents.items()}
    if sort_by_appearance:
        length = contents[grouping.index[0]].size
        indexer = np.arange(length)
        indexer = indexer.reshape((reps, -1))
        indexer = indexer.ravel(order="F")
        contents = {label: arr[indexer] for label, arr in contents.items()}
    return contents, reps


def _stack_dot_value(
    spec: pd.DataFrame,
    others: list,
    df: pd.DataFrame,
    sort_by_appearance: bool,
) -> dict:
    """
    Flip the .value into long form.
    """
    if spec[".value"].nunique() == 1:
        return _stack_dot_value_single_label(
            spec=spec, sort_by_appearance=sort_by_appearance, df=df
        )
    return _stack_dot_value_multiple_labels(
        spec=spec, df=df, others=others, sort_by_appearance=sort_by_appearance
    )


def _stack_dot_value_single_label(
    spec: pd.DataFrame,
    df: pd.DataFrame,
    sort_by_appearance: bool,
) -> dict:
    """
    Flip the .value into long form.
    Applicable where .value.nunique == 1
    """
    _value = spec.pop(".value")
    key = _value._values[0]
    reps = len(spec)
    if df.dtypes.map(is_extension_array_dtype).any(axis=None):
        contents = [arr._values for _, arr in df.items()]
        contents = concat_compat(contents)
        if sort_by_appearance:
            length = contents.size
            indexer = np.arange(length)
            indexer = indexer.reshape((reps, -1))
            indexer = indexer.ravel(order="F")
            contents = contents[indexer]
    else:
        if sort_by_appearance:
            contents = df._values.ravel()
        else:
            contents = df._values.ravel(order="F")
    contents = {key: contents}
    return spec, contents


def _stack_dot_value_multiple_labels(
    spec: pd.DataFrame,
    others: list,
    df: pd.DataFrame,
    sort_by_appearance: bool,
) -> dict:
    """
    Flip the .value into long form.
    Applicable where .value.nunique > 1
    """
    # summary of implementation logic:
    # think of the `others` as a grouper
    # e.g df:
    #    Sepal.Length  Sepal.Width  Petal.Length  Petal.Width    Species
    # 0           5.1          3.5           1.4          0.2     setosa
    # 1           5.9          3.0           5.1          1.8  virginica
    # spec:
    #           .name  .value   part
    # 0  Sepal.Length  Length  Sepal
    # 1  Petal.Length  Length  Petal
    # 2   Sepal.Width   Width  Sepal
    # 3   Petal.Width   Width  Petal
    #
    # just like with _reshape_by_spec, we can see the pairing
    # between .value and the the column names('.name')
    # but also, we have a pairing for others -
    # in the spec dataframe above, it is `part`
    # we can see how `part` acts as a guard/grouper
    # for .value and .name ->
    #
    # other > .value > .name
    # Sepal-> {  Length,         Width}
    # Sepal->{[Sepal.Length, Sepal.Width]}
    # Petal->{   Length,         Width}
    # Petal->{Petal.Length,  Petal.Width}
    #
    # the above translates into this form:
    #
    # part    Length              Width
    # Sepal   Sepal.Length      Sepal.Width
    # Petal   Petal.Length      Petal.Width
    #
    # the final dataframe will contain the values for each
    # of the relevant column names
    # if there is an imbalance in the grouping,
    # then null arrays are introduced to keep things balanced
    # e.g, lets assume spec is of the form below:
    # spec:
    #           .name  .value   part
    # 0  Sepal.Length  Length  Sepal
    # 1  Petal.Length  Length  Petal
    # 2   Sepal.Width   Width  Sepal
    # 3   Petal.Width   Width  Metal
    #
    # we can see that `part` has three unique labels ->
    # Sepal -> 2 counts
    # Petal -> 1 count
    # Metal -> 1 count
    #
    # obviously there is an imbalance here,
    # and this reflects in the grouping
    #
    # other > .value > .name
    # Sepal-> {  Length,         Width}
    # Sepal->{[Sepal.Length, Sepal.Width]}
    # Petal->{   Length,         Width}
    # Petal->{Petal.Length,       None}
    # Metal->{   Length,         Width}
    # Metal->{    None,       Petal.Width}
    #
    # which translates into ->
    #
    # part    Length              Width
    # Sepal   Sepal.Length      Sepal.Width
    # Petal   Petal.Length      None
    # Metal   None              Petal.Width
    #
    # Based on the grouping above,
    # when creating the final DataFrame,
    # the entries with None will be filled with null arrays
    # to keep the balance
    #
    # if .value is just a single label
    # we can shortcut all the approach above
    # also, we use column positions, instead of column names
    # since pandas supports duplicate columns
    # also the combination of .value and others should be unique
    # identification per label in other should be clear/distinct
    _value = spec.pop(".value")
    grouped = spec.groupby(others, sort=False, observed=True, dropna=False)
    mapp = defaultdict(dict)
    df.columns = range(df.columns.size)
    for pos, header, _pos in zip(grouped.ngroup(), _value, df.columns):
        dictionary = {header: _pos}
        mapp[pos].update(dictionary)
    uniqs = _value.unique()
    _uniqs = set(uniqs)
    missing = (_uniqs.difference(header) for _, header in mapp.items())
    if any(missing):
        null_array = np.full(shape=len(df), fill_value=np.nan)
    else:
        null_array = None
    mapping = defaultdict(list)
    for _, dictionary in mapp.items():
        for label in uniqs:
            column_position = dictionary.get(label)
            mapping[label].append(column_position)
    contents = defaultdict(list)
    for label, list_of_positions in mapping.items():
        for position in list_of_positions:
            arr = df.get(position, null_array)
            if isinstance(arr, pd.Series):
                arr = arr._values
            contents[label].append(arr)
    contents = {label: concat_compat(arr) for label, arr in contents.items()}
    if sort_by_appearance:
        length = contents[uniqs[0]].size
        indexer = np.arange(length)
        indexer = indexer.reshape((grouped.ngroups, -1))
        indexer = indexer.ravel(order="F")
        contents = {label: arr[indexer] for label, arr in contents.items()}
    return spec.drop_duplicates(), contents


def _stack_non_dot_value(
    spec, df, reps, index, ignore_index, sort_by_appearance
) -> tuple:
    """
    Flip components that are not .value
    into long form.
    """
    if sort_by_appearance:
        return _stack_non_dot_value_sort(
            spec=spec, df=df, reps=reps, index=index, ignore_index=ignore_index
        )

    return _stack_non_dot_value_do_not_sort(
        spec=spec, df=df, reps=reps, index=index, ignore_index=ignore_index
    )


def _stack_non_dot_value_sort(spec, df, reps, index, ignore_index) -> tuple:
    """
    Flip components that are not .value
    into long form.
    """
    len_df = len(df)
    # taking a long route (instead of np.repeat/np.tile)
    # to get a bit more performance
    # in some cases perf has been 3x faster
    if spec is not None:
        shape = (len_df, reps)
        indexer = np.empty(shape=shape, dtype=np.intp)
        arr = np.arange(reps).reshape((1, reps))
        indexer[:] = arr
        indexer = indexer.ravel()
        spec = {label: arr._values for label, arr in spec.items()}
        spec = {label: arr[indexer] for label, arr in spec.items()}

    if index or not ignore_index:
        shape = (len_df, reps)
        indexer = np.empty(shape=shape, dtype=np.intp)
        arr = np.arange(len_df).reshape((len_df, 1))
        indexer[:] = arr
        indexer = indexer.ravel()
        if index:
            index = {label: arr[indexer] for label, arr in index.items()}
    if ignore_index:
        length = len_df * reps
        df_index = range(length)
    else:
        df_index = df.index[indexer]
    return index, spec, df_index


def _stack_non_dot_value_do_not_sort(
    spec, df, reps, index, ignore_index
) -> tuple:
    """
    Flip components that are not .value
    into long form.
    """
    len_df = len(df)
    # taking a long route (instead of np.repeat/np.tile)
    # to get a bit more performance
    # in some cases perf has been 3x faster
    if spec is not None:
        shape = (reps, len_df)
        indexer = np.empty(shape=shape, dtype=np.intp)
        arr = np.arange(reps).reshape((reps, 1))
        indexer[:] = arr
        indexer = indexer.ravel()
        spec = {label: arr._values for label, arr in spec.items()}
        spec = {label: arr[indexer] for label, arr in spec.items()}

    if index or not ignore_index:
        shape = (reps, len_df)
        indexer = np.empty(shape=shape, dtype=np.intp)
        arr = np.arange(len_df).reshape((1, len_df))
        indexer[:] = arr
        indexer = indexer.ravel()
        if index:
            index = {label: arr[indexer] for label, arr in index.items()}
    if ignore_index:
        length = len_df * reps
        df_index = range(length)
    else:
        df_index = df.index[indexer]
    return index, spec, df_index


def _pivot_longer_dropna(
    df: pd.DataFrame, contents: dict, ignore_index: bool
) -> pd.DataFrame:
    """
    Drop nulls from long form
    """
    nulls = [pd.isna(arr) for _, arr in contents.items()]
    nulls = np.logical_and.reduce(nulls)
    if not nulls.any():
        return df
    df = df.iloc[~nulls]
    if ignore_index:
        df.index = range(len(df))
    return df


def _names_transform(
    spec: pd.DataFrame, others: list, names_transform: str | Callable | dict
) -> pd.DataFrame:
    """
    Convert names extracted from the dataframe's columns
    to specified dtype(s) in names_transform.
    """
    if not others:
        return spec
    if isinstance(names_transform, dict):
        for label in others:
            dtype = names_transform.get(label)
            if not dtype:
                continue
            spec[label] = spec[label].astype(dtype)
        return spec
    if isinstance(names_transform, str):
        for label in others:
            spec[label] = spec[label].astype(names_transform)
        return spec
    for label in others:
        try:
            # built-in functions (float, int, ...)
            spec[label] = spec[label].astype(names_transform)
        except TypeError:
            spec[label] = names_transform(spec[label])
    return spec


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.pivot` instead."
    )
)
def pivot_wider(
    df: pd.DataFrame,
    index: list | str = None,
    names_from: list | str = None,
    values_from: list | str = None,
    flatten_levels: bool = True,
    names_sep: str = "_",
    names_glue: str = None,
    reset_index: bool = True,
    names_expand: bool = False,
    index_expand: bool = False,
) -> pd.DataFrame:
    """Reshapes data from *long* to *wide* form.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.pivot` instead.

    The number of columns are increased, while decreasing
    the number of rows. It is the inverse of the
    [`pivot_longer`][janitor.functions.pivot.pivot_longer]
    method, and is a wrapper around `pd.DataFrame.pivot` method.

    This method does not mutate the original DataFrame.

    Column selection in `index`, `names_from` and `values_from`
    is possible using the
    [`select`][janitor.functions.select.select] syntax.

    A ValueError is raised if the combination
    of the `index` and `names_from` is not unique.

    By default, values from `values_from` are always
    at the top level if the columns are not flattened.
    If flattened, the values from `values_from` are usually
    at the start of each label in the columns.

    Examples:
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
        >>> df.pivot_wider( # doctest: +SKIP
        ...     index = "dep",
        ...     names_from = "step",
        ... )
           dep  a_1  a_2  b_1  b_2
        0  5.5   20   25   30   37
        1  6.1   22   18   19   29

        Modify columns with `names_sep`:
        >>> df.pivot_wider( # doctest: +SKIP
        ...     index = "dep",
        ...     names_from = "step",
        ...     names_sep = "",
        ... )
           dep  a1  a2  b1  b2
        0  5.5  20  25  30  37
        1  6.1  22  18  19  29

        Modify columns with `names_glue`:
        >>> df.pivot_wider( # doctest: +SKIP
        ...     index = "dep",
        ...     names_from = "step",
        ...     names_glue = "{_value}_step{step}",
        ... )
           dep  a_step1  a_step2  b_step1  b_step2
        0  5.5       20       25       30       37
        1  6.1       22       18       19       29

        Expand columns to expose implicit missing values
        - this applies only to categorical columns:
        >>> weekdays = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        >>> daily = pd.DataFrame(
        ...     {
        ...         "day": pd.Categorical(
        ...             values=("Tue", "Thu", "Fri", "Mon"), categories=weekdays
        ...         ),
        ...         "value": (2, 3, 1, 5),
        ...     },
        ... index=[0, 0, 0, 0],
        ... )
        >>> daily
           day  value
        0  Tue      2
        0  Thu      3
        0  Fri      1
        0  Mon      5
        >>> daily.pivot_wider(names_from='day', values_from='value') # doctest: +SKIP
           Tue  Thu  Fri  Mon
        0    2    3    1    5
        >>> (daily # doctest: +SKIP
        ... .pivot_wider(
        ...     names_from='day',
        ...     values_from='value',
        ...     names_expand=True)
        ... )
           Mon  Tue  Wed  Thu  Fri  Sat  Sun
        0    5    2  NaN    3    1  NaN  NaN

        Expand the index to expose implicit missing values
        - this applies only to categorical columns:
        >>> daily = daily.assign(letter = list('ABBA'))
        >>> daily
           day  value letter
        0  Tue      2      A
        0  Thu      3      B
        0  Fri      1      B
        0  Mon      5      A
        >>> daily.pivot_wider(index='day',names_from='letter',values_from='value') # doctest: +SKIP
           day    A    B
        0  Tue  2.0  NaN
        1  Thu  NaN  3.0
        2  Fri  NaN  1.0
        3  Mon  5.0  NaN
        >>> (daily # doctest: +SKIP
        ... .pivot_wider(
        ...     index='day',
        ...     names_from='letter',
        ...     values_from='value',
        ...     index_expand=True)
        ... )
           day    A    B
        0  Mon  5.0  NaN
        1  Tue  2.0  NaN
        2  Wed  NaN  NaN
        3  Thu  NaN  3.0
        4  Fri  NaN  1.0
        5  Sat  NaN  NaN
        6  Sun  NaN  NaN


    !!! abstract "Version Changed"

        - 0.24.0
            - Added `reset_index`, `names_expand` and `index_expand` parameters.

    Args:
        df: A pandas DataFrame.
        index: Name(s) of columns to use as identifier variables.
            It should be either a single column name, or a list of column names.
            If `index` is not provided, the DataFrame's index is used.
        names_from: Name(s) of column(s) to use to make the new
            DataFrame's columns. Should be either a single column name,
            or a list of column names.
        values_from: Name(s) of column(s) that will be used for populating
            the new DataFrame's values.
            If `values_from` is not specified,  all remaining columns
            will be used.
        flatten_levels: If `False`, the DataFrame stays as a MultiIndex.
        names_sep: If `names_from` or `values_from` contain multiple
            variables, this will be used to join the values into a single string
            to use as a column name. Default is `_`.
            Applicable only if `flatten_levels` is `True`.
        names_glue: A string to control the output of the flattened columns.
            It offers more flexibility in creating custom column names,
            and uses python's `str.format_map` under the hood.
            Simply create the string template,
            using the column labels in `names_from`,
            and special `_value` as a placeholder for `values_from`.
            Applicable only if `flatten_levels` is `True`.
        reset_index: Determines whether to restore `index`
            as a column/columns. Applicable only if `index` is provided,
            and `flatten_levels` is `True`.
        names_expand: Expand columns to show all the categories.
            Applies only if `names_from` is a categorical column.
        index_expand: Expand the index to show all the categories.
            Applies only if `index` is a categorical column.

    Returns:
        A pandas DataFrame that has been unpivoted from long to wide form.
    """  # noqa: E501

    # no need for an explicit copy --> df = df.copy()
    # `pd.pivot` creates one
    return _computations_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        flatten_levels,
        names_sep,
        names_glue,
        reset_index,
        names_expand,
        index_expand,
    )


def _computations_pivot_wider(
    df: pd.DataFrame,
    index: list | str | None,
    names_from: list | str | None,
    values_from: list | str | None,
    flatten_levels: bool,
    names_sep: str,
    names_glue: str | None,
    reset_index: bool,
    names_expand: bool,
    index_expand: bool,
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
        reset_index,
        names_expand,
        index_expand,
    ) = _data_checks_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        flatten_levels,
        names_sep,
        names_glue,
        reset_index,
        names_expand,
        index_expand,
    )

    out = df.pivot(  # noqa: PD010
        index=index, columns=names_from, values=values_from
    )

    indexer = out.index
    if index_expand and index:
        any_categoricals = (indexer.get_level_values(name) for name in index)
        any_categoricals = (
            isinstance(entry, pd.CategoricalIndex)
            for entry in any_categoricals
        )
        any_categoricals = any(any_categoricals)
        if any_categoricals:
            indexer = _expand(indexer, retain_categories=True)
            out = out.reindex(index=indexer)

    indexer = out.columns
    if names_expand:
        any_categoricals = (
            indexer.get_level_values(name) for name in names_from
        )
        any_categoricals = (
            isinstance(entry, pd.CategoricalIndex)
            for entry in any_categoricals
        )
        any_categoricals = any(any_categoricals)
        if any_categoricals:
            retain_categories = True
            if flatten_levels & (
                (names_glue is not None)
                | isinstance(indexer, pd.MultiIndex)
                | ((index is not None) & reset_index)
            ):
                retain_categories = False
            indexer = _expand(indexer, retain_categories=retain_categories)
            out = out.reindex(columns=indexer)

    indexer = None
    if any((out.empty, not flatten_levels)):
        return out

    if isinstance(out.columns, pd.MultiIndex) and names_glue:
        new_columns = out.columns
        if ("_value" in names_from) and (None in new_columns.names):
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
                "_value" if ent is None else ent for ent in new_columns.names
            ]
            new_columns = [
                names_glue.format_map(dict(zip(names_from, entry)))
                for entry in new_columns
            ]
        except KeyError as error:
            raise KeyError(
                f"{error} is not a column label in names_from."
            ) from error

        out.columns = new_columns
    elif names_glue:
        try:
            new_columns = [
                names_glue.format_map({names_from[0]: entry})
                for entry in out.columns
            ]
        except KeyError as error:
            raise KeyError(
                f"{error} is not a column label in names_from."
            ) from error
        out.columns = new_columns
    else:
        names_sep = "_" if names_sep is None else names_sep
        out = out.collapse_levels(sep=names_sep)

    if index and reset_index:
        out = out.reset_index()

    if out.columns.names:
        out.columns.names = [None]

    return out


def _data_checks_pivot_wider(
    df,
    index,
    names_from,
    values_from,
    flatten_levels,
    names_sep,
    names_glue,
    reset_index,
    names_expand,
    index_expand,
):
    """
    This function raises errors if the arguments have the wrong
    python type, or if the column does not exist in the dataframe.
    This function is executed before proceeding to the computation phase.
    Type annotations are not provided because this function is where type
    checking happens.
    """

    is_multi_index = isinstance(df.columns, pd.MultiIndex)
    if index is not None:
        if is_multi_index:
            if not isinstance(index, list):
                raise TypeError(
                    "For a MultiIndex column, pass a list of tuples "
                    "to the index argument."
                )
            index = _check_tuples_multiindex(df.columns, index, "index")
        else:
            if is_list_like(index):
                index = list(index)
            index = get_index_labels(index, df, axis="columns")
            if not is_list_like(index):
                index = [index]
            else:
                index = list(index)

    if names_from is None:
        raise ValueError(
            "pivot_wider() is missing 1 required argument: 'names_from'"
        )

    if is_multi_index:
        if not isinstance(names_from, list):
            raise TypeError(
                "For a MultiIndex column, pass a list of tuples "
                "to the names_from argument."
            )
        names_from = _check_tuples_multiindex(
            df.columns, names_from, "names_from"
        )
    else:
        if is_list_like(names_from):
            names_from = list(names_from)
        names_from = get_index_labels(names_from, df, axis="columns")
        if not is_list_like(names_from):
            names_from = [names_from]
        else:
            names_from = list(names_from)

    if values_from is not None:
        if is_multi_index:
            if not isinstance(values_from, list):
                raise TypeError(
                    "For a MultiIndex column, pass a list of tuples "
                    "to the values_from argument."
                )
            out = _check_tuples_multiindex(
                df.columns, values_from, "values_from"
            )
        else:
            if is_list_like(values_from):
                values_from = list(values_from)
            out = get_index_labels(values_from, df, axis="columns")
            if not is_list_like(out):
                out = [out]
            else:
                out = list(out)
        # hack to align with pd.pivot
        if values_from == out[0]:
            values_from = out[0]
        else:
            values_from = out

    check("flatten_levels", flatten_levels, [bool])

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    if names_glue is not None:
        check("names_glue", names_glue, [str])

    check("reset_index", reset_index, [bool])
    check("names_expand", names_expand, [bool])
    check("index_expand", index_expand, [bool])

    return (
        df,
        index,
        names_from,
        values_from,
        flatten_levels,
        names_sep,
        names_glue,
        reset_index,
        names_expand,
        index_expand,
    )


def _expand(indexer, retain_categories):
    """
    Expand Index to all categories.
    Applies to categorical index, and used
    in _computations_pivot_wider for scenarios where
    names_expand and/or index_expand is True.
    Categories are preserved where possible.
    If `retain_categories` is False, a fastpath is taken
    to generate all possible combinations.

    Returns an Index.
    """
    if indexer.nlevels > 1:
        names = indexer.names
        if not retain_categories:
            indexer = pd.MultiIndex.from_product(indexer.levels, names=names)
        else:
            indexer = [
                indexer.get_level_values(n) for n in range(indexer.nlevels)
            ]
            indexer = [
                (
                    pd.Categorical(
                        values=arr.categories,
                        categories=arr.categories,
                        ordered=arr.ordered,
                    )
                    if isinstance(arr, pd.CategoricalIndex)
                    else arr.unique()
                )
                for arr in indexer
            ]
            indexer = pd.MultiIndex.from_product(indexer, names=names)

    else:
        if not retain_categories:
            indexer = indexer.categories
        else:
            indexer = pd.Categorical(
                values=indexer.categories,
                categories=indexer.categories,
                ordered=indexer.ordered,
            )
    return indexer


def _check_tuples_multiindex(indexer, args, param):
    """
    Check entries for tuples,
    if indexer is a MultiIndex.

    Returns a list of tuples.
    """
    all_tuples = (isinstance(arg, tuple) for arg in args)
    if not all(all_tuples):
        raise TypeError(
            f"{param} must be a list of tuples "
            "when the columns are a MultiIndex."
        )

    not_found = set(args).difference(indexer)
    if any(not_found):
        raise KeyError(
            f"Tuples {*not_found,} in the {param} "
            "argument do not exist in the dataframe's columns."
        )

    return args
