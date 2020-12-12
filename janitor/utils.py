"""Miscellaneous internal PyJanitor helper functions."""

import functools
import os
import sys
import warnings
from itertools import chain, product
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .errors import JanitorError


def check(varname: str, value, expected_types: list):
    """
    One-liner syntactic sugar for checking types.

    Should be used like this::

        check('x', x, [int, float])

    :param varname: The name of the variable.
    :param value: The value of the varname.
    :param expected_types: The types we expect the item to be.
    :raises TypeError: if data is not the expected type.
    """
    is_expected_type = False
    for t in expected_types:
        if isinstance(value, t):
            is_expected_type = True
            break

    if not is_expected_type:
        raise TypeError(
            "{varname} should be one of {expected_types}".format(
                varname=varname, expected_types=expected_types
            )
        )


def _clean_accounting_column(x: str) -> float:
    """
    Perform the logic for the `cleaning_style == "accounting"` attribute.

    This is a private function, not intended to be used outside of
    ``currency_column_to_numeric``.

    It is intended to be used in a pandas `apply` method.

    :returns: An object with a cleaned column.
    """
    y = x.strip()
    y = y.replace(",", "")
    y = y.replace(")", "")
    y = y.replace("(", "-")
    if y == "-":
        return 0.00
    return float(y)


def _currency_column_to_numeric(x, cast_non_numeric=None) -> str:
    """
    Perform logic for changing cell values.

    This is a private function intended to be used only in
    ``currency_column_to_numeric``.

    It is intended to be used in a pandas `apply` method, after being passed
    through `partial`.
    """
    acceptable_currency_characters = {
        "-",
        ".",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    }
    if len(x) == 0:
        return "ORIGINAL_NA"

    if cast_non_numeric:
        if x in cast_non_numeric.keys():
            check(
                "{%r: %r}" % (x, str(cast_non_numeric[x])),
                cast_non_numeric[x],
                [int, float],
            )
            return cast_non_numeric[x]
        else:
            return "".join(i for i in x if i in acceptable_currency_characters)
    else:
        return "".join(i for i in x if i in acceptable_currency_characters)


def _replace_empty_string_with_none(column_series):
    column_series.loc[column_series == ""] = None
    return column_series


def _replace_original_empty_string_with_none(column_series):
    column_series.loc[column_series == "ORIGINAL_NA"] = None
    return column_series


def _strip_underscores(
    df: pd.DataFrame, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
    """
    Strip underscores from DataFrames column names.

    Underscores can be stripped from the beginning, end or both.

    .. code-block:: python

        df = _strip_underscores(df, strip_underscores='left')

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer underscores from all
        column names. Default None keeps outer underscores. Values can be
        either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
        and True.
    :returns: A pandas DataFrame with underscores removed.
    """
    df = df.rename(
        columns=lambda x: _strip_underscores_func(x, strip_underscores)
    )
    return df


def _strip_underscores_func(
    col: str, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
    """Strip underscores from a string."""
    underscore_options = [None, "left", "right", "both", "l", "r", True]
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )

    if strip_underscores in ["left", "l"]:
        col = col.lstrip("_")
    elif strip_underscores in ["right", "r"]:
        col = col.rstrip("_")
    elif strip_underscores == "both" or strip_underscores is True:
        col = col.strip("_")
    return col


def import_message(
    submodule: str,
    package: str,
    conda_channel: str = None,
    pip_install: bool = False,
):
    """
    Return warning if package is not found.

    Generic message for indicating to the user when a function relies on an
    optional module / package that is not currently installed. Includes
    installation instructions. Used in `chemistry.py` and `biology.py`.

    :param submodule: pyjanitor submodule that needs an external dependency.
    :param package: External package this submodule relies on.
    :param conda_channel: Conda channel package can be installed from,
        if at all.
    :param pip_install: Whether package can be installed via pip.
    """
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    installable = True
    if is_conda:
        if conda_channel is None:
            installable = False
            installation = f"{package} cannot be installed via conda"
        else:
            installation = f"conda install -c {conda_channel} {package}"
    else:
        if pip_install:
            installation = f"pip install {package}"
        else:
            installable = False
            installation = f"{package} cannot be installed via pip"

    print(
        f"To use the janitor submodule {submodule}, you need to install "
        f"{package}."
    )
    print()
    if installable:
        print("To do so, use the following command:")
        print()
        print(f"    {installation}")
    else:
        print(f"{installation}")


def idempotent(func: Callable, df: pd.DataFrame, *args, **kwargs):
    """
    Raises error if a function operating on a `DataFrame` is not idempotent,
    that is, `func(func(df)) = func(df)` is not true for all `df`.

    :param func: A python method.
    :param df: A pandas `DataFrame`.
    :param args: Positional arguments supplied to the method.
    :param kwargs: Keyword arguments supplied to the method.
    :raises ValueError: If `func` is found to not be idempotent for the given
        `DataFrame` `df`.
    """
    if not func(df, *args, **kwargs) == func(
        func(df, *args, **kwargs), *args, **kwargs
    ):
        raise ValueError(
            "Supplied function is not idempotent for the given " "DataFrame."
        )


def deprecated_alias(**aliases) -> Callable:
    """
    Used as a decorator when deprecating old function argument names, while
    keeping backwards compatibility.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

    Functional usage example:

    .. code-block:: python

        @deprecated_alias(a='alpha', b='beta')
        def simple_sum(alpha, beta):
            return alpha + beta

    :param aliases: Dictionary of aliases for a function's arguments.
    :return: Your original function wrapped with the kwarg redirection
        function.
    """  # noqa: E501

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def refactored_function(message: str) -> Callable:
    """Used as a decorator when refactoring functions

    Implementation is inspired from `Hacker Noon`_.

    .. Hacker Noon: https://hackernoon.com/why-refactoring-how-to-restructure-python-package-51b89aa91987

    Functional usage example:

    .. code-block:: python

        @refactored_function(
            message="simple_sum() has been refactored. Use hard_sum() instead."
        )
        def simple_sum(alpha, beta):
            return alpha + beta

    :param message: Message to use in warning user about refactoring.
    :return: Your original function wrapped with the kwarg redirection
        function.
    """  # noqa: E501

    def decorator(func):
        def emit_warning(*args, **kwargs):
            warnings.warn(message, FutureWarning)
            return func(*args, **kwargs)

        return emit_warning

    return decorator


def rename_kwargs(func_name: str, kwargs: Dict, aliases: Dict):
    """
    Used to update deprecated argument names with new names. Throws a
    TypeError if both arguments are provided, and warns if old alias is used.
    Nothing is returned as the passed ``kwargs`` are modified directly.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

    :param func_name: name of decorated function.
    :param kwargs: Arguments supplied to the method.
    :param aliases: Dictionary of aliases for a function's arguments.
    :raises TypeError: if both arguments are provided.
    """  # noqa: E501
    for old_alias, new_alias in aliases.items():
        if old_alias in kwargs:
            if new_alias in kwargs:
                raise TypeError(
                    f"{func_name} received both {old_alias} and {new_alias}"
                )
            warnings.warn(
                f"{old_alias} is deprecated; use {new_alias}",
                DeprecationWarning,
            )
            kwargs[new_alias] = kwargs.pop(old_alias)


def check_column(
    df: pd.DataFrame, old_column_names: List, present: bool = True
):
    """
    One-liner syntactic sugar for checking the presence or absence of a column.

    Should be used like this::

        check(df, ['a', 'b'], present=True)

    :param df: The name of the variable.
    :param old_column_names: A list of column names we want to check to see if
        present (or absent) in df.
    :param present: If True (default), checks to see if all of old_column_names
        are in df.columns. If False, checks that none of old_column_names are
        in df.columns.
    :raises ValueError: if data is not the expected type.
    """
    for column_name in old_column_names:
        if present:
            if column_name not in df.columns:
                raise ValueError(
                    f"{column_name} not present in dataframe columns!"
                )
        else:  # Tests for exclusion
            if column_name in df.columns:
                raise ValueError(
                    f"{column_name} already present in dataframe columns!"
                )


def skipna(f: Callable) -> Callable:
    """
    Decorator for escaping np.nan and None in a function

    Should be used like this::

        df[column].apply(skipna(transform))

    or::

        @skipna
        def transform(x):
            pass

    :param f: the function to be wrapped
    :returns: _wrapped, the wrapped function
    """

    def _wrapped(x, *args, **kwargs):
        if (type(x) is float and np.isnan(x)) or x is None:
            return np.nan
        else:
            return f(x, *args, **kwargs)

    return _wrapped


def skiperror(
    f: Callable, return_x: bool = False, return_val=np.nan
) -> Callable:
    """
    Decorator for escaping errors in a function

    Should be used like this::

        df[column].apply(
            skiperror(transform, return_val=3, return_x=False))

    or::

        @skiperror(return_val=3, return_x=False)
        def transform(x):
            pass

    :param f: the function to be wrapped
    :param return_x: whether or not the original value that caused error
        should be returned
    :param return_val: the value to be returned when an error hits.
        Ignored if return_x is True
    :returns: _wrapped, the wrapped function
    """

    def _wrapped(x, *args, **kwargs):
        try:
            return f(x, *args, **kwargs)
        except Exception:
            if return_x:
                return x
            return return_val

    return _wrapped


def _check_instance(entry: Dict):
    """
    Function to check instances in the expand_grid function.
    This checks if entry is a dictionary,
    checks the instance of value in key:value pairs in entry,
    and makes changes to other types as deemed necessary.
    Additionally, ValueErrors are raised if empty containers are
    passed in as values into the dictionary.
    How each type is handled, and their associated exceptions,
    are pretty clear from the code.
    """
    # dictionary should not be empty
    if not entry:
        raise ValueError("passed dictionary cannot be empty")

    # couple of checks that should cause the program to fail early
    # if conditions are not met
    for _, value in entry.items():

        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("array cannot be empty")
            if value.ndim > 2:
                raise ValueError(
                    "expand_grid works only on 1D and 2D structures."
                )

        if isinstance(value, (pd.DataFrame, pd.Series)):
            if value.empty:
                raise ValueError("passed DataFrame cannot be empty")

        if isinstance(value, (list, tuple, set, dict)):
            if not value:
                raise ValueError("passed data cannot be empty")

    entry = {
        # If it is a scalar value, then wrap in a list
        # this is necessary, as we will use the itertools.product function
        # which works only on iterables.
        key: [value]
        if isinstance(value, (type(None), int, float, bool, str, np.generic))
        else value
        for key, value in entry.items()
    }

    return entry


def _grid_computation(entry: Dict) -> pd.DataFrame:
    """
    Return the final output of the expand_grid function as a dataframe.
     This kicks in after the ``_check_instance`` function is completed,
     and essentially creates a cross join of the values in the `entry`
     dictionary. If the `entry` dictionary is a collection of lists/tuples,
     then `itertools.product` will be used for the cross join, before a
    dataframe is created; if however, the `entry` contains a pandas dataframe
    or a pandas series or a numpy array, then identical indices are created for
    each entry and `pandas DataFrame join` is called to create the cross join.
    """

    # checks if the dictionary does not have any of
    # (pd.Dataframe, pd.Series, numpy) values and uses itertools.product.
    # numpy meshgrid is faster, but requires homogenous data to appreciate
    # the speed, and also to keep the data type for each column created.
    # As an example, if we have a mix in the dictionary of strings and numbers,
    # numpy will convert it to an object data type. Itertools product is
    # efficient and does not lose the data type.

    if not any(
        isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
        for key, value in entry.items()
    ):
        df_expand_grid = (value for key, value in entry.items())
        df_expand_grid = product(*df_expand_grid)
        return pd.DataFrame(df_expand_grid, columns=entry)

    # dictionary is a mix of different types - dataframe/series/numpy/...
    # so we check for each data type- if it is a pandas dataframe, then convert
    # to numpy and add to `df_expand_grid`; the other data types are added to
    # `df_expand_grid` as is. For each of the data types, new column names are
    # created if they do not have, and modified if names already exist. These
    # names are built through the for loop below and added to `df_columns`
    df_columns = []
    df_expand_grid = []
    for key, value in entry.items():
        if isinstance(value, pd.DataFrame):
            df_expand_grid.append(value.to_numpy())
            if isinstance(value.columns, pd.MultiIndex):
                df_columns.extend(
                    [f"{key}_{ind}" for ind, col in enumerate(value.columns)]
                )
            else:
                df_columns.extend([f"{key}_{col}" for col in value])
        elif isinstance(value, pd.Series):
            df_expand_grid.append(np.array(value))
            if value.name:
                df_columns.append(f"{key}_{value.name}")
            else:
                df_columns.append(str(key))
        elif isinstance(value, np.ndarray):
            df_expand_grid.append(value)
            if value.ndim == 1:
                df_columns.append(f"{key}_0")
            else:
                df_columns.extend(
                    [f"{key}_{ind}" for ind in range(value.shape[-1])]
                )
        else:
            df_expand_grid.append(value)
            df_columns.append(key)

        # here we run the product function from itertools only if there is
        # more than one item in the list; if only one item, we simply
        # create a dataframe with the new column names from `df_columns`
    if len(df_expand_grid) > 1:
        df_expand_grid = product(*df_expand_grid)
        df_expand_grid = (
            chain.from_iterable(
                [val]
                if not isinstance(val, (pd.DataFrame, pd.Series, np.ndarray))
                else val
                for val in value
            )
            for value in df_expand_grid
        )
        return pd.DataFrame(df_expand_grid, columns=df_columns)
    return pd.DataFrame(*df_expand_grid, columns=df_columns)


def _complete_groupings(df, list_of_columns):
    # this collects all the columns as individual labels, which will be
    # used to set the index of the dataframe
    index_columns = []
    # this will collect all the values associated with the respective
    # columns, and used to reindex the dataframe, to get the complete
    # pairings
    reindex_columns = []
    for item in list_of_columns:
        if not isinstance(item, (str, dict, list, tuple)):
            raise ValueError(
                """Value must either be a column label, a list/tuple of columns or a
                    dictionary where the keys are columns in the dataframe."""
            )
        if not item:
            raise ValueError("grouping cannot be empty")
        if isinstance(item, str):
            reindex_columns.append(set(df[item].array))
            index_columns.append(item)
        else:
            # this comes into play if we wish to input values that
            # do not exist in the data, say years, or alphabets, or
            # range of numbers
            if isinstance(item, dict):
                if len(item) > 1:
                    index_columns.extend(item.keys())
                else:
                    index_columns.append(*item.keys())
                    item_contents = [
                        # convert scalars to iterables; this is necessary
                        # when creating combinations with itertools' product
                        [value]
                        if isinstance(value, (int, float, str, bool))
                        else value
                        for key, value in item.items()
                    ]
                    reindex_columns.extend(item_contents)
            else:
                index_columns.extend(item)
                # TODO : change this to read as a numpy instead
                # instead of a list comprehension
                # it should be faster
                item = (df[sub_column].array for sub_column in item)
                item = set(zip(*item))
                reindex_columns.append(item)

    reindex_columns = product(*reindex_columns)
    # A list comprehension, coupled with itertools chain.from_iterable
    # would likely be faster; I fear that it may hamper readability with
    # nested list comprehensions; as such, I chose the for loop method.
    new_reindex_columns = []
    for row in reindex_columns:
        new_row = []
        for cell in row:
            if isinstance(cell, tuple):
                new_row.extend(cell)
            else:
                new_row.append(cell)
        new_reindex_columns.append(tuple(new_row))

    df = df.set_index(index_columns)

    return df, new_reindex_columns


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

    if index is not None:
        if isinstance(index, str):
            index = [index]
        check("index", index, [list, tuple, Pattern])

    if column_names is not None:
        if isinstance(column_names, str):
            column_names = [column_names]
        check("column_names", column_names, [list, tuple, Pattern])

    if isinstance(names_to, str):
        names_to = [names_to]

    else:
        if isinstance(names_to, tuple):
            names_to = list(names_to)

    check("names_to", names_to, [list])

    if not all(isinstance(word, str) for word in names_to):
        raise TypeError("All entries in `names_to` argument must be strings.")

    if len(names_to) > 1:
        if all((names_pattern, names_sep)):
            raise ValueError(
                """
                    Only one of `names_pattern` or `names_sep`
                    should be provided.
                    """
            )

        if ".value" in names_to:
            if names_to.count(".value") > 1:
                raise ValueError(
                    "There can be only one `.value` in `names_to`."
                )
    if len(names_to) == 1:
        # names_sep creates more than one column
        # whereas regex with names_pattern can be limited to one column
        if names_sep is not None:
            raise ValueError(
                """
                    For a single `names_to` value,
                    `names_sep` is not required.
                    """
            )
    if names_pattern is not None:
        check("names_pattern", names_pattern, [str, Pattern, List, Tuple])

        if isinstance(names_pattern, (list, tuple)):
            if not all(
                isinstance(word, (str, Pattern)) for word in names_pattern
            ):
                raise TypeError(
                    """
                    All entries in ``names_pattern`` argument
                    must be regular expressions.
                    """
                )

            if len(names_pattern) != len(names_to):
                raise ValueError(
                    """
                    Length of ``names_to`` does not match
                    number of patterns.
                    """
                )

            if ".value" in names_to:
                raise ValueError(
                    """
                    ``.value`` not accepted if ``names_pattern``
                    is a list/tuple.
                    """
                )

    if names_sep is not None:
        check("names_sep", names_sep, [str, Pattern])

    check("values_to", values_to, [str])

    if values_to in df.columns:
        if any(
            (
                ".value" not in names_to,
                not isinstance(names_pattern, (list, tuple)),
            )
        ):
            # copied from pandas' melt source code
            # with a minor tweak
            raise ValueError(
                """
                This dataframe has a column name that matches the
                'values_to' column name of the resulting Dataframe.
                Kindly set the 'values_to' parameter to a unique name.
                """
            )

    if column_level is not None:
        check("column_level", column_level, [int, str])

    if any((names_sep, names_pattern)):
        if isinstance(df.columns, pd.MultiIndex):
            raise ValueError(
                """
                Unpivoting a MultiIndex column dataframe when
                `names_sep` or `names_pattern` is supplied is
                not supported.
                """
            )

    if all((names_sep is None, names_pattern is None)):
        # adapted from pandas' melt source code
        if index is not None:
            if isinstance(df.columns, pd.MultiIndex) and not isinstance(
                index, list
            ):
                raise ValueError(
                    """
                    index must be a list of tuples
                    when columns are a MultiIndex.
                    """
                )

        if column_names is not None:
            if isinstance(df.columns, pd.MultiIndex) and not isinstance(
                column_names, list
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


def _pivot_longer_pattern_match(
    df: pd.DataFrame,
    index: Optional[Pattern] = None,
    column_names: Optional[Pattern] = None,
) -> Tuple:
    """
    This checks if a pattern (regular expression) is supplied
    to index or column_names and extracts the column labels
    that match the given regular expression.

    A dataframe, along with the `index` and `column_names` are
    returned.
    """

    # TODO: allow `janitor.patterns` to accept a list/tuple
    # of regular expresssions.
    if isinstance(column_names, Pattern):
        column_names = [
            column_name
            for column_name in df
            if column_names.search(column_name)
        ]

    if isinstance(index, Pattern):
        index = [
            column_name for column_name in df if index.search(column_name)
        ]

    return df, index, column_names


def __tile_compat(arr: pd.Index, df: pd.DataFrame) -> pd.Index:
    """
    Repeats index multiple times.
    """
    # adapted from pandas.core.reshape.utils (_tile_compat)
    # applies to MultiIndexes as well
    # numpy resize, with len(df) as length works too
    taker = np.tile(np.arange(len(arr)), len(df) // len(arr))
    return arr.take(taker)


def _restore_index_and_sort_by_appearance(
    df: pd.DataFrame, ignore_index, sort_by_appearance, df_index
) -> pd.DataFrame:
    """
    This function restores the original index via the `ignore_index`
    and `df_index` parameters, and sorts the resulting dataframe
    by appearance, via the `sort_by_appearance` parameter.
    It is meant for sections in the `_computations_pivot_longer`
    function that do not have `.value` in the dataframe's column names.

    An example for `ignore_index`:

    Say we have data like below:
           A  B  C
        0  a  1  2
        1  b  3  4
        2  c  5  6

    If `ignore_index` is False (this means the original index will be reused),
    then the resulting dataframe will look like below:

           A  variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5
        0  a        C      2
        1  b        C      4
        2  c        C      6

    Note how the index is repeated ([0,1,2,0,1,2]).

    An example for `sort_by_appearance` :

    Say data looks like this :
        id, a1, a2, a3, A1, A2, A3
         1, a, b, c, A, B, C

    when unpivoted into long form, it will look like this :
              id instance    a     A
        0     1     1        a     A
        1     1     2        b     B
        2     1     3        c     C

    where the column `a` comes before `A`, as it was in the source data,
    and in column `a`, `a > b > c`, also as it was in the source data.

    A dataframe that is possibly reindexed or sorted by appearance
    is returned.
    """
    primary_index_sorter = None
    index_sorter = None

    if sort_by_appearance:
        # index is already an array of integers(from the melt operation)
        # so let's use it
        primary_index_sorter = df.index

    # if the height of the new dataframe is the same as the old
    # dataframe, simply reuse the original index.
    if not ignore_index:
        if not len(df_index) == len(df):
            df.index = __tile_compat(df_index, df)
        else:
            df.index = df_index

    length_check = any((len(df_index) == 1, len(df_index) == len(df)))
    if sort_by_appearance:
        # if the height of the new dataframe is the same as the height
        # of the original dataframe, then there is no need to sort
        # by appearance
        if not length_check:
            index_sorter = np.reshape(
                primary_index_sorter, (-1, len(df_index))
            ).ravel(order="F")
            df = df.take(index_sorter)

        if ignore_index:
            df.index = primary_index_sorter

    return df


def _pivot_longer_extractions(
    df: pd.DataFrame,
    index: Optional[Union[List, Tuple]] = None,
    column_names: Optional[Union[List, Tuple]] = None,
    names_to: Optional[Union[List, Tuple, str]] = None,
    names_sep: Optional[Union[str, Pattern]] = None,
    names_pattern: Optional[
        Union[
            List[Union[str, Pattern]], Tuple[Union[str, Pattern]], str, Pattern
        ]
    ] = None,
) -> Tuple:

    """
    This is where the labels within the column names are separated
    into new columns, and is executed if `names_sep` or `names_pattern`
    is not None.

    A tuple, containing the dataframe and `others`, are returned.
    """

    # This is executed if the user is only interested in a subsection
    # put here because it is specific only for situations where `names_sep`
    # or `names_pattern` is not None
    if all((index, column_names)):
        df = df.filter(index + column_names)

    dot_value = any(
        ((".value" in names_to), isinstance(names_pattern, (list, tuple)))
    )

    if dot_value:
        # the extra indices (np.arange(len(df))) serve as a way to
        # associate each unpivoted value to a unique row identifier.
        # It also aids in sorting by appearance. append is used in set_index,
        # in case the user wishes to keep the original index.
        if index:
            df = df.set_index([*index, np.arange(len(df))], append=True)
        else:
            df = df.set_index(np.arange(len(df)), append=True)

    mapping = None
    reindex_columns = None
    if names_sep:
        mapping = pd.Series(df.columns).str.split(names_sep, expand=True)

        if len(mapping.columns) != len(names_to):
            raise ValueError(
                """
                The length of ``names_to`` does not match
                the number of columns extracted.
                """
            )
        mapping.columns = names_to

    else:
        if isinstance(names_pattern, str):
            mapping = df.columns.str.extract(names_pattern)

            if mapping.isna().all(axis=None):
                raise ValueError(
                    """
                    The regular expression in ``names_pattern``
                    did not return any matches.
                    """
                )
            if len(names_to) != len(mapping.columns):
                raise ValueError(
                    """
                    The length of ``names_to`` does not match
                    the number of columns extracted.
                    """
                )
            mapping.columns = names_to

        else:  # list/tuple of regular expressions
            mapping = [
                df.columns.str.contains(regex) for regex in names_pattern
            ]

            if not np.any(mapping):
                raise ValueError(
                    """
                    The regular expressions in ``names_pattern``
                    did not return any matches.
                    """
                )
            mapping = np.select(mapping, names_to, None)
            mapping = pd.DataFrame(mapping, columns=[".value"])

    group = None
    others = None
    positions = None
    if not dot_value:
        if index:
            # more efficient to do this, than having to
            # set the index and resetting.
            # This puts the index(or indices) into the first
            # column and set the remaining columns to empty string
            # on the same row(s). When melting, pd.melt will
            # gracefully handle the empty cells and replicate the
            # `index` values accordingly.
            positions = df.columns.get_indexer(index)
            mapping.iloc[positions, 0] = index
            mapping.iloc[positions, 1:] = ""

    else:
        # This ensures the complete representations of all column labels
        # in `.value` and in `others`. Idea is borrowed from the
        # `complete` method.
        reindex_columns = [column.unique() for _, column in mapping.items()]
        reindex_columns = product(*reindex_columns)
        reindex_columns = pd.DataFrame(
            reindex_columns, columns=mapping.columns
        )

        # The `others` variable comes in
        # handy when reshaping the data.
        others = [
            column_name for column_name in mapping if column_name != ".value"
        ]
        # creating categoricals allows us to sort the data; this is necessary
        # because we will be splitting the data into different chunks and
        # concatenating back into one whole. As such, having the data uniform
        # in all the chunks before merging back is crucial. Also, having
        # categoricals ensure the data stays in its original form in terms
        # of first appearance.
        category_dtypes = {
            key: CategoricalDtype(
                categories=column.dropna().unique(), ordered=True
            )
            if column.hasnans
            else CategoricalDtype(categories=column.unique(), ordered=True)
            for key, column in mapping.items()
            if key in others
        }
        mapping = mapping.astype(category_dtypes)
        # creating a group is primarily for sorting by appearance.
        # The secondary purpose is to indicate groups if a list/tuple
        # is passed to `names_pattern`.
        group = "group"
        for column_name in mapping:
            if column_name == group:
                group = f"{group}_1"
        mapping.loc[:, group] = mapping.groupby(".value").cumcount()
        others.append(group)
        # When merging with `mapping`, null values may be generated;
        # The only columns needed for complete cases are `.value` and
        # `others` (excluding `group`). Fill the null values with 0, to
        # keep them as a group. This will also help during the melting.
        # setting `mapping` to the left of the merge operation ensures
        # that all the actual values are kept in the data; at some point in
        # `computations_pivot_longer` the duplicate columns will be dropped;
        # only the leftmost columns will be retained.
        reindex_columns = mapping.merge(reindex_columns, how="outer").fillna(0)
        reindex_columns = pd.MultiIndex.from_frame(reindex_columns)

    if len(mapping.columns) > 1:
        df.columns = pd.MultiIndex.from_frame(mapping)
    else:
        df.columns = mapping.iloc[:, 0]
    # now there is a full representation of all the labels, both in
    # '.value' and others
    if dot_value:
        df = df.reindex(columns=reindex_columns)

    return df, others, group


def _computations_pivot_longer(
    df: pd.DataFrame,
    index: Optional[Union[List, Tuple]] = None,
    column_names: Optional[Union[List, Tuple]] = None,
    names_to: Optional[Union[List, Tuple, str]] = None,
    values_to: Optional[str] = "value",
    column_level: Optional[Union[int, str]] = None,
    names_sep: Optional[Union[str, Pattern]] = None,
    names_pattern: Optional[
        Union[
            List[Union[str, Pattern]], Tuple[Union[str, Pattern]], str, Pattern
        ]
    ] = None,
    sort_by_appearance: Optional[bool] = False,
    ignore_index: Optional[bool] = True,
) -> pd.DataFrame:
    """
    This is the main workhorse of the `pivot_longer` function.
    Below is a summary of how the function accomplishes its tasks:

    1. If `names_sep` or `names_pattern` is not provided, then regular data
       unpivoting is covered with pandas melt.

    2. If `names_sep` or `names_pattern` is not None, the first step is to
       extract the relevant values from the columns, using either
       `str.split(expand=True)`, if `names_sep` is provided, or `str.extract()`
       if `names_pattern` is provided. If `names_pattern` is a list/tuple of
       regular expressions, then `str.contains` along with `numpy` select is
       used for the extraction.

        After the extraction, `pd.melt` is executed.

    3. 'The labels in `names_to` become the new column names, if `.value`
        is not in `names_to`, or if `names_pattern` is not a list/tuple of
        regexes.

    4.  If, however, `names_to` contains `.value`, or `names_pattern` is a
        list/tuple of regexes, then the `.value` column is unstacked(in a
        manner of speaking, `pd.DataFrame.unstack` is not actually used) to
        become new column name(s), while the other values, if any, go under
        different column names. `values_to` is overriden.

    5.  If `ignore_index` is `False`, then the index of the source dataframe is
        returned, and repeated as necessary.

    6.  If the user wants the data in order of appearance, in which case, the
        unpivoted data appears in stacked form, then `sort_by_appearance`
        covers that.

    An unpivoted dataframe is returned.
    """

    if index is not None:
        check_column(df, index, present=True)

    if column_names is not None:
        check_column(df, column_names, present=True)

    if (index is None) and column_names:
        if len(df.columns) > len(column_names):
            index = [
                column_name
                for column_name in df
                if column_name not in column_names
            ]

    df_index = df.index

    # scenario 1
    if all((names_pattern is None, names_sep is None)):

        df = pd.melt(
            df,
            id_vars=index,
            value_vars=column_names,
            var_name=names_to,
            value_name=values_to,
            col_level=column_level,
        )

        df = _restore_index_and_sort_by_appearance(
            df=df,
            ignore_index=ignore_index,
            sort_by_appearance=sort_by_appearance,
            df_index=df_index,
        )

        return df

    if any((names_pattern, names_sep)):
        df, others, group = _pivot_longer_extractions(
            df=df,
            index=index,
            column_names=column_names,
            names_to=names_to,
            names_sep=names_sep,
            names_pattern=names_pattern,
        )

        if ".value" not in df.columns.names:
            df = pd.melt(df, id_vars=index, value_name=values_to)

            df = _restore_index_and_sort_by_appearance(
                df=df,
                ignore_index=ignore_index,
                sort_by_appearance=sort_by_appearance,
                df_index=df_index,
            )
            return df

        # .value
        stubnames = df.columns.get_level_values(".value").unique()
        df_index = df.index
        # ensures data is uniform in all the chunks
        df = df.sort_index(axis="columns", level=others, sort_remaining=False)
        # separate into chunks by stubname
        if len(stubnames) > 1:
            df = [
                df.xs(key=stub, level=".value", axis="columns").melt(
                    value_name=stub
                )
                for stub in stubnames
            ]

            first, *rest = df
            # the `first` frame already has `others`,
            # `others` can be safely discarded from the other dataframes
            # this is why sorting was done earlier (via the sort_index)
            # to ensure uniformity
            rest = [frame.iloc[:, -1] for frame in rest]
            df = pd.concat([first, *rest], axis="columns")

        else:
            df = df.xs(key=stubnames[0], level=".value", axis="columns").melt(
                value_name=stubnames[0]
            )

        # attach df_index
        df.index = __tile_compat(df_index, df)
        if sort_by_appearance:
            df = df.set_index(group, append=True)
            # sorting by the extra index(-2) and group(-1)
            df = df.sort_index(level=[-2, -1], sort_remaining=False)
            df = df.droplevel(level=[-2, -1])
        else:
            df = df.drop(columns=group)
            df = df.droplevel(level=-1)  # the extra index
        if index:
            df = df.reset_index(level=index)
        if ignore_index:
            df.index = np.arange(len(df))

    return df


def _data_checks_pivot_wider(
    df,
    index,
    names_from,
    values_from,
    names_sort,
    flatten_levels,
    values_from_first,
    names_prefix,
    names_sep,
    fill_value,
):

    """
    This function raises errors if the arguments have the wrong
    python type, or if the column does not exist in the dataframe.
    This function is executed before proceeding to the computation phase.
    Type annotations are not provided because this function is where type
    checking happens.
    """

    if index is not None:
        if isinstance(index, str):
            index = [index]
        check("index", index, [list])
        check_column(df, index, present=True)

    if names_from is None:
        raise ValueError(
            "pivot_wider() missing 1 required argument: 'names_from'"
        )

    if names_from is not None:
        if isinstance(names_from, str):
            names_from = [names_from]
        check("names_from", names_from, [list])
        check_column(df, names_from, present=True)

    if values_from is not None:
        check("values_from", values_from, [list, str])
        if isinstance(values_from, str):
            check_column(df, [values_from], present=True)
        else:
            check_column(df, values_from, present=True)

    if values_from_first is not None:
        check("values_from_first", values_from_first, [bool])

    check("names_sort", names_sort, [bool])

    check("flatten_levels", flatten_levels, [bool])

    if names_prefix is not None:
        check("names_prefix", names_prefix, [str])

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    if fill_value is not None:
        check("fill_value", fill_value, [int, float, str])

    return (
        df,
        index,
        names_from,
        values_from,
        names_sort,
        flatten_levels,
        values_from_first,
        names_prefix,
        names_sep,
        fill_value,
    )


def _computations_pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[List, str]] = None,
    names_from: Optional[Union[List, str]] = None,
    values_from: Optional[Union[List, str]] = None,
    names_sort: Optional[bool] = False,
    flatten_levels: Optional[bool] = True,
    values_from_first: Optional[bool] = True,
    names_prefix: Optional[str] = None,
    names_sep: Optional[str] = "_",
    fill_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """
    This is the main workhorse of the `pivot_wider` function.
    If `values_from` is a list, then every item in `values_from`
    will be added to the front of each output column. This option
    can be turned off with the `values_from_first` argument, in
    which case, the `names_from` variables (or `names_prefix`, if
    present) start each column.
    The `unstack` method is used here, and not `pivot`; multiple
    labels in the `index` or `names_from` are supported in the
    `pivot` function for Pandas 1.1 and above. Also, the
    `pivot_table` function is not used, because it is quite slow,
    compared to the `pivot` function and `unstack`.
    The columns are sorted in the order of appearance from the
    source data. This only occurs if `flatten_levels` is True.
    """

    # TODO : include an aggfunc argument
    if values_from is None:
        if index:
            values_from = [
                col for col in df.columns if col not in (index + names_from)
            ]
        else:
            values_from = [col for col in df.columns if col not in names_from]

    dtypes = None
    before = None
    # ensure `names_sort` is in combination with `flatten_levels`
    if all((names_sort is False, flatten_levels)):
        # dtypes only needed for names_from
        # since that is what will become the new column names
        dtypes = {
            column_name: CategoricalDtype(
                categories=column.unique(), ordered=True
            )
            for column_name, column in df.filter(names_from).items()
        }
        if index is not None:
            before = df.filter(index)
            if before.duplicated().any():
                before = before.drop_duplicates()

        df = df.astype(dtypes)

    if index is None:  # use existing index
        df = df.set_index(names_from, append=True)
    else:
        df = df.set_index(index + names_from)

    if not df.index.is_unique:
        raise ValueError(
            """
            There are non-unique values in your combination
            of `index` and `names_from`. Kindly provide a
            unique identifier for each row.
            """
        )

    df = df.loc[:, values_from]
    df = df.unstack(names_from, fill_value=fill_value)  # noqa: PD010
    if flatten_levels:
        if isinstance(values_from, list):
            df.columns = df.columns.set_names(level=0, names="values_from")
            if not values_from_first:
                df = df.reorder_levels(names_from + ["values_from"], axis=1)
        if df.columns.nlevels > 1:
            df.columns = [names_sep.join(entry) for entry in df]
        if names_prefix:
            df = df.add_prefix(names_prefix)
        df.columns = list(
            df.columns
        )  # blanket approach that covers categories

        if index:
            # this way we avoid having to convert index
            # from category to original dtype
            # while still maintaining order of appearance
            if names_sort is False:
                df = before.merge(
                    df, how="left", left_on=index, right_index=True
                ).reset_index(drop=True)
            else:
                df = df.reset_index()
        else:
            # remove default index, since it is irrelevant
            df = df.reset_index().iloc[:, 1:]

        return df

    return df
