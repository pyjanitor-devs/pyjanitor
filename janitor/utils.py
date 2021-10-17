"""Miscellaneous internal PyJanitor helper functions."""

import fnmatch
import functools
from collections import defaultdict
import os
import re
import socket
import sys
import warnings
import operator
from collections.abc import Callable as dispatch_callable
from itertools import chain, count
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
    Hashable,
)

import numpy as np
import pandas as pd
from pandas.core.reshape.merge import _MergeOperation
from pandas.core.construction import extract_array
from pandas.api.types import (
    CategoricalDtype,
    is_extension_array_dtype,
    is_list_like,
    is_scalar,
    is_integer_dtype,
    is_float_dtype,
    is_string_dtype,
    is_datetime64_dtype,
    is_categorical_dtype,
)
from pandas.core.common import apply_if_callable
from enum import Enum


from .errors import JanitorError


def check(varname: str, value, expected_types: list):
    """
    One-liner syntactic sugar for checking types.
    It can also check callables.

    Should be used like this::

        check('x', x, [int, float])

    :param varname: The name of the variable (for diagnostic error message).
    :param value: The value of the varname.
    :param expected_types: The types we expect the item to be.
    :raises TypeError: if data is not the expected type.
    """
    is_expected_type: bool = False
    for t in expected_types:
        if t is callable:
            is_expected_type = t(value)
        else:
            is_expected_type = isinstance(value, t)
        if is_expected_type:
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
        return "".join(i for i in x if i in acceptable_currency_characters)
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
    df: pd.DataFrame, column_names: Union[Iterable, str], present: bool = True
):
    """
    One-liner syntactic sugar for checking the presence or absence of columns.

    Should be used like this::

        check(df, ['a', 'b'], present=True)

    This will check whether columns "a" and "b" are present in df's columns.

    One can also guarantee that "a" and "b" are not present
    by switching to ``present = False``.

    :param df: The name of the variable.
    :param column_names: A list of column names we want to check to see if
        present (or absent) in df.
    :param present: If True (default), checks to see if all of column_names
        are in df.columns. If False, checks that none of column_names are
        in df.columns.
    :raises ValueError: if data is not the expected type.
    """
    if isinstance(column_names, str) or not isinstance(column_names, Iterable):
        column_names = [column_names]

    for column_name in column_names:
        if present and column_name not in df.columns:  # skipcq: PYL-R1720
            raise ValueError(
                f"{column_name} not present in dataframe columns!"
            )
        elif not present and column_name in df.columns:
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
        return f(x, *args, **kwargs)

    return _wrapped


def skiperror(
    f: Callable, return_x: bool = False, return_val=np.nan
) -> Callable:
    """
    Decorator for escaping any error in a function.

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
        except Exception:  # skipcq: PYL-W0703
            if return_x:
                return x
            return return_val

    return _wrapped


def _computations_expand_grid(others: dict) -> pd.DataFrame:
    """
    Creates a cartesian product of all the inputs in `others`.
    Combines Numpy's `mgrid`, with the `take` method in numpy/Pandas,
    to expand each input to the length of the cumulative product of
    all inputs in `others`.

    There is a performance penalty for small entries
    in using this method, instead of `itertools.product`;
    however, there is significant performance benefits
    as the size of the data increases.

    Another benefit of this approach,
    in addition to the significant performance gains,
    is the preservation of data types. This is particularly relevant for
    Pandas' extension arrays dtypes (categoricals, nullable integers, ...).

    A MultiIndex DataFrame of all possible combinations is returned.
    """

    for key in others:
        check("key", key, [Hashable])

    grid = {}

    for key, value in others.items():
        if is_scalar(value):
            value = pd.DataFrame([value])
        elif (not isinstance(value, pd.Series)) and is_extension_array_dtype(
            value
        ):
            value = pd.DataFrame(value)
        elif is_list_like(value) and (not hasattr(value, "shape")):
            value = np.asarray([*value])

        grid[key] = value

    others = None

    # slice obtained here is used in `np.mgrid`
    # to generate cartesian indices
    # which is then paired with grid.items()
    # to blow up each individual value
    # before finally recombining, via pd.concat,
    # to create a dataframe.
    grid_index = [slice(len(value)) for _, value in grid.items()]
    grid_index = np.mgrid[grid_index]
    grid_index = map(np.ravel, grid_index)
    grid = zip(grid.items(), grid_index)
    grid = ((*left, right) for left, right in grid)
    grid = {
        key: _expand_grid(value, grid_index) for key, value, grid_index in grid
    }

    # creates a MultiIndex with the keys
    # since grid is a dictionary
    return pd.concat(grid, axis="columns", sort=False, copy=False)


@functools.singledispatch
def _expand_grid(value, grid_index):
    """
    Base function for dispatch of `_expand_grid`.
    """

    raise TypeError(
        f"""
        {type(value).__name__} data type
        is not supported in `expand_grid`.
        """
    )


@_expand_grid.register(np.ndarray)
def _sub_expand_grid(value, grid_index):  # noqa: F811
    """
    Expands the numpy array based on `grid_index`.

    Returns Series if 1-D array,
    or DataFrame if 2-D array.
    """

    if not (value.size > 0):
        raise ValueError("""array cannot be empty.""")

    if value.ndim > 2:
        raise ValueError(
            """
            expand_grid works only
            on 1D and 2D arrays.
            """
        )

    value = value[grid_index]

    return pd.DataFrame(value)


@_expand_grid.register(pd.Series)
def _sub_expand_grid(value, grid_index):  # noqa: F811
    """
    Expands the Series based on `grid_index`.

    Returns Series.
    """
    if value.empty:
        raise ValueError("""Series cannot be empty.""")

    value = value.iloc[grid_index]
    value.index = pd.RangeIndex(start=0, stop=len(value))

    return value.to_frame()


@_expand_grid.register(pd.DataFrame)
def _sub_expand_grid(value, grid_index):  # noqa: F811
    """
    Expands the DataFrame based on `grid_index`.

    Returns a DataFrame.
    """
    if value.empty:
        raise ValueError("""DataFrame cannot be empty.""")

    value = value.iloc[grid_index]
    value.index = pd.RangeIndex(start=0, stop=len(value))
    if isinstance(value.columns, pd.MultiIndex):
        value.columns = ["_".join(map(str, ent)) for ent in value]

    return value


@_expand_grid.register(pd.Index)
def _sub_expand_grid(value, grid_index):  # noqa: F811
    """
    Expands the Index based on `grid_index`.

    Returns a DataFrame (if MultiIndex), or a Series.
    """
    if value.empty:
        raise ValueError("""Index cannot be empty.""")

    value = value[grid_index]

    return value.to_frame(index=False)


def _data_checks_complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]],
    sort: Optional[bool] = False,
    by: Optional[Union[list, str]] = None,
):
    """
    Function to check parameters in the `complete` function.
    Checks the type of the `columns` parameter, as well as the
    types within the `columns` parameter.

    Check is conducted to ensure that column names are not repeated.

    Also checks that the names in `columns` actually exist in `df`.

    Returns `df`, `columns`, `column_checker`,
    and `by` if all checks pass.

    """
    # TODO: get `complete` to work on MultiIndex columns,
    # if there is sufficient interest with use cases
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            """
            `complete` does not support MultiIndex columns.
            """
        )

    columns = [
        list(grouping) if isinstance(grouping, tuple) else grouping
        for grouping in columns
    ]
    column_checker = []
    for grouping in columns:
        check("grouping", grouping, [list, dict, str])
        if not grouping:
            raise ValueError("grouping cannot be empty")
        if isinstance(grouping, str):
            column_checker.append(grouping)
        else:
            column_checker.extend(grouping)

    # columns should not be duplicated across groups
    column_checker_no_duplicates = set()
    for column in column_checker:
        if column in column_checker_no_duplicates:
            raise ValueError(
                f"""{column} column should be in only one group."""
            )
        column_checker_no_duplicates.add(column)  # noqa: PD005

    check_column(df, column_checker)
    column_checker_no_duplicates = None

    check("sort", sort, [bool])

    if by is not None:
        if isinstance(by, str):
            by = [by]
        check("by", by, [list])
        check_column(df, by)

    return columns, column_checker, sort, by


def _computations_complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]],
    sort: bool = False,
    by: Optional[Union[list, str]] = None,
) -> pd.DataFrame:
    """
    This function computes the final output for the `complete` function.

    If `by` is present, then groupby apply is used.

    A DataFrame, with rows of missing values, if any, is returned.
    """

    columns, column_checker, sort, by = _data_checks_complete(
        df, columns, sort, by
    )

    all_strings = True
    for column in columns:
        if not isinstance(column, str):
            all_strings = False
            break

    # nothing to 'complete' here
    if all_strings and len(columns) == 1:
        return df

    # under the right conditions, stack/unstack can be faster
    # plus it always returns a sorted DataFrame
    # which does help in viewing the missing rows
    # however, using a merge keeps things simple
    # with a stack/unstack,
    # the relevant columns combination should be unique
    # and there should be no nulls
    # trade-off for the simplicity of merge is not so bad
    # of course there could be a better way ...
    if by is None:
        uniques = _generic_complete(df, columns, all_strings)
        return df.merge(uniques, how="outer", on=column_checker, sort=sort)

    uniques = df.groupby(by)
    uniques = uniques.apply(_generic_complete, columns, all_strings)
    uniques = uniques.droplevel(-1)
    return df.merge(uniques, how="outer", on=by + column_checker, sort=sort)


def _generic_complete(
    df: pd.DataFrame, columns: list, all_strings: bool = True
):
    """
    Generate cartesian product for `_computations_complete`.

    Returns a Series or DataFrame, with no duplicates.
    """
    if all_strings:
        uniques = {col: df[col].unique() for col in columns}
        uniques = _computations_expand_grid(uniques)
        uniques = uniques.droplevel(level=-1, axis="columns")
        return uniques

    uniques = {}
    for index, column in enumerate(columns):
        if isinstance(column, dict):
            column = _complete_column(column, df)
            uniques = {**uniques, **column}
        else:
            uniques[index] = _complete_column(column, df)

    if len(uniques) == 1:
        _, uniques = uniques.popitem()
        return uniques.to_frame()

    uniques = _computations_expand_grid(uniques)
    return uniques.droplevel(level=0, axis="columns")


@functools.singledispatch
def _complete_column(column, df):
    """
    Args:
        column : str/list/dict
        df: Pandas DataFrame

    A Pandas Series/DataFrame with no duplicates,
    or a list of unique Pandas Series is returned.
    """
    raise TypeError(
        """This type is not supported in the `complete` function."""
    )


@_complete_column.register(str)  # noqa: F811
def _sub_complete_column(column, df):  # noqa: F811
    """
    Args:
        column : str
        df: Pandas DataFrame

    Returns:
        Pandas Series
    """

    column = df[column]

    if not column.is_unique:
        return column.drop_duplicates()
    return column


@_complete_column.register(list)  # noqa: F811
def _sub_complete_column(column, df):  # noqa: F811
    """
    Args:
        column : list
        df: Pandas DataFrame

    Returns:
        Pandas DataFrame
    """

    column = df.loc[:, column]

    if column.duplicated().any(axis=None):
        return column.drop_duplicates()

    return column


@_complete_column.register(dict)  # noqa: F811
def _sub_complete_column(column, df):  # noqa: F811
    """
    Args:
        column : dictionary
        df: Pandas DataFrame

    Returns:
        A dictionary of unique pandas Series.
    """

    collection = {}
    for key, value in column.items():
        arr = apply_if_callable(value, df[key])
        if not is_list_like(arr):
            raise ValueError(
                f"""
                value for {key} should be a 1-D array.
                """
            )
        if not hasattr(arr, "shape"):
            arr = pd.Series([*arr], name=key)

        if (not isinstance(arr, pd.Series)) and is_extension_array_dtype(arr):
            arr = pd.Series(arr)

        if isinstance(arr, pd.Index):
            arr_ndim = arr.nlevels
        else:
            arr_ndim = arr.ndim

        if arr_ndim != 1:
            raise ValueError(
                f"""
                Kindly provide a 1-D array for {key}.
                """
            )

        if not (arr.size > 0):
            raise ValueError(
                f"""
                Kindly ensure the provided array for {key}
                has at least one value.
                """
            )

        if isinstance(arr, pd.Index):
            arr = arr.to_series(index=pd.RangeIndex(start=0, stop=arr.size))

        if isinstance(arr, np.ndarray):
            arr = pd.Series(arr)

        if not arr.is_unique:
            arr = arr.drop_duplicates()

        if arr.name is None:
            arr.name = key

        collection[key] = arr

    return collection


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
        index = _select_columns(index, df)

    if column_names is not None:
        if is_list_like(column_names) and (
            not isinstance(column_names, tuple)
        ):
            column_names = list(column_names)
        column_names = _select_columns(column_names, df)

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
    names_sort,
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
        index = _select_columns(index, df)

    if names_from is None:
        raise ValueError(
            "pivot_wider() is missing 1 required argument: 'names_from'"
        )

    if is_list_like(names_from):
        names_from = [*names_from]
    names_from = _select_columns(names_from, df)

    if values_from is not None:
        if is_list_like(values_from):
            values_from = [*values_from]
        values_from = _select_columns(values_from, df)
        if len(values_from) == 1:
            values_from = values_from[0]

    check("names_sort", names_sort, [bool])

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
        names_sort,
        levels_order,
        flatten_levels,
        names_sep,
        names_glue,
    )


def _computations_pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[List, str]] = None,
    names_from: Optional[Union[List, str]] = None,
    values_from: Optional[Union[List, str]] = None,
    names_sort: Optional[bool] = False,
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
    # check dtype of `names_from` is string
    names_from_all_strings = df.filter(names_from).agg(is_string_dtype).all()

    if names_sort:
        # Categorical dtypes created only for `names_from`
        # since that is what will become the new column names
        dtypes = {
            column_name: CategoricalDtype(
                df[column_name].factorize(sort=False)[-1], ordered=True
            )
            for column_name in names_from
        }
        df = df.astype(dtypes)

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
    if not names_from_all_strings:
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
    if names_sort:
        df.columns = [*df.columns]

    if index:
        df = df.reset_index()

    if df.columns.names:
        df = df.rename_axis(columns=None)

    return df


class CategoryOrder(Enum):
    """
    order types for encode_categorical.
    """

    SORT = "sort"
    APPEARANCE = "appearance"


def as_categorical_checks(df: pd.DataFrame, **kwargs) -> dict:
    """
    This function raises errors if columns in `kwargs` are
    absent in the the dataframe's columns.
    It also raises errors if the tuple in `kwargs`
    has a length greater than 2, or the `order` value,
    if not None, is not one of `appearance` or `sort`.
    Error is raised if the `categories` in the tuple in `kwargs`
    is not a 1-D array-like object.

    This function is executed before proceeding to the computation phase.

    If all checks pass, a dictionary of column names and tuple
    of (categories, order) is returned.

    :param df: The pandas DataFrame object.
    :param kwargs: A pairing of column name
        to a tuple of (`categories`, `order`).
    :returns: A dictionary.
    :raises TypeError: if the value in ``kwargs`` is not a tuple.
    :raises ValueError: if ``categories`` is not a 1-D array.
    :raises ValueError: if ``order`` is not one of
        `sort`, `appearance`, or `None`.
    """

    # column checks
    check_column(df, kwargs)

    categories_dict = {}

    for column_name, value in kwargs.items():
        # type check
        check("Pair of `categories` and `order`", value, [tuple])

        len_value = len(value)

        if len_value != 2:
            raise ValueError(
                f"""
                The tuple of (categories, order) for {column_name}
                should be length 2; the tuple provided is
                length {len_value}.
                """
            )

        cat, order = value
        if cat is not None:
            if not is_list_like(cat):
                raise TypeError(f"{cat} should be list-like.")

            if not hasattr(cat, "shape"):
                checker = pd.Index([*cat])
            else:
                checker = cat

            arr_ndim = checker.ndim
            if (arr_ndim != 1) or isinstance(checker, pd.MultiIndex):
                raise ValueError(
                    f"""
                    {cat} is not a 1-D array.
                    Kindly provide a 1-D array-like object to `categories`.
                    """
                )

            if not isinstance(checker, (pd.Series, pd.Index)):
                checker = pd.Index(checker)

            if checker.hasnans:
                raise ValueError(
                    "Kindly ensure there are no nulls in `categories`."
                )

            if not checker.is_unique:
                raise ValueError(
                    """
                    Kindly provide unique,
                    non-null values for `categories`.
                    """
                )

            if checker.empty:
                raise ValueError(
                    """
                    Kindly ensure there is at least
                    one non-null value in `categories`.
                    """
                )

            # uniques, without nulls
            uniques = df[column_name].factorize(sort=False)[-1]
            if uniques.empty:
                raise ValueError(
                    f"""
                     Kindly ensure there is at least
                     one non-null value in {column_name}.
                     """
                )

            missing = uniques.difference(checker, sort=False)
            if not missing.empty and (uniques.size > missing.size):
                warnings.warn(
                    f"""
                     Values {tuple(missing)} are missing from
                     the provided categories {cat}
                     for {column_name}; this may create nulls
                     in the new categorical column.
                     """,
                    UserWarning,
                    stacklevel=2,
                )

            elif uniques.equals(missing):
                warnings.warn(
                    f"""
                     None of the values in {column_name} are in
                     {cat};
                     this might create nulls for all values
                     in the new categorical column.
                     """,
                    UserWarning,
                    stacklevel=2,
                )

        if order is not None:
            check("order", order, [str])

            category_order_types = [ent.value for ent in CategoryOrder]
            if order.lower() not in category_order_types:
                raise ValueError(
                    """
                    `order` argument should be one of
                    "appearance", "sort" or `None`.
                    """
                )

        categories_dict[column_name] = value

    return categories_dict


def _computations_as_categorical(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    This function handles cases where
    categorical columns are created with an order,
    or specific values supplied for the categories.
    It uses a kwarg, where the key is the column name,
    and the value is a tuple of categories, order.
    The defaults for the tuple are (None, None)
    and will return a categorical dtype
    with no order and categories inferred from the column.
    A DataFrame, with catetorical columns, is returned.
    """

    categories_dict = as_categorical_checks(df, **kwargs)

    categories_dtypes = {}

    for column_name, (
        cat,
        order,
    ) in categories_dict.items():
        error_msg = f"""
                     Kindly ensure there is at least
                     one non-null value in {column_name}.
                     """
        if (cat is None) and (order is None):
            cat_dtype = pd.CategoricalDtype()

        elif (cat is None) and (order is CategoryOrder.SORT.value):
            cat = df[column_name].factorize(sort=True)[-1]
            if cat.empty:
                raise ValueError(error_msg)
            cat_dtype = pd.CategoricalDtype(categories=cat, ordered=True)

        elif (cat is None) and (order is CategoryOrder.APPEARANCE.value):
            cat = df[column_name].factorize(sort=False)[-1]
            if cat.empty:
                raise ValueError(error_msg)
            cat_dtype = pd.CategoricalDtype(categories=cat, ordered=True)

        elif cat is not None:  # order is irrelevant if cat is provided
            cat_dtype = pd.CategoricalDtype(categories=cat, ordered=True)

        categories_dtypes[column_name] = cat_dtype

    return df.astype(categories_dtypes)


def is_connected(url: str) -> bool:
    """
    This is a helper function to check if the client
    is connected to the internet.

    Example:
        print(is_connected("www.google.com"))
        console >> True

    :param url: We take a test url to check if we are
        able to create a valid connection.
    :raises OSError: if connection to ``URL`` cannot be
        established
    :return: We return a boolean that signifies our
        connection to the internet
    """
    try:
        sock = socket.create_connection((url, 80))
        if sock is not None:
            sock.close()
            return True
    except OSError as e:
        import warnings

        warnings.warn(
            "There was an issue connecting to the internet. "
            "Please see original error below."
        )
        raise e
    return False


@functools.singledispatch
def _select_columns(columns_to_select, df):
    """
    base function for column selection.
    Returns a list of column names.
    """
    raise TypeError("This type is not supported in column selection.")


@_select_columns.register(str)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to strings.
    It is also applicable to shell-like glob strings,
    specifically, the `*`.
    A list of column names is returned.
    """
    filtered_columns = None
    df_columns = df.columns
    if "*" in columns_to_select:  # shell-style glob string (e.g., `*_thing_*`)
        filtered_columns = fnmatch.filter(df_columns, columns_to_select)
    elif columns_to_select in df_columns:
        filtered_columns = [columns_to_select]
        return filtered_columns
    if not filtered_columns:
        raise KeyError(f"No match was returned for '{columns_to_select}'")
    df_columns = None
    return filtered_columns


@_select_columns.register(slice)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to slices.
    The start slice value must be a string or None;
    same goes for the stop slice value.
    The step slice value should be an integer or None.
    A slice, if passed correctly in a Multindex column,
    returns a list of tuples across all levels of the
    column.
    A list of column names is returned.
    """

    df_columns = df.columns
    filtered_columns = None
    start_check = None
    stop_check = None
    step_check = None

    if not df_columns.is_unique:
        raise ValueError(
            """
            The column labels are not unique.
            Kindly ensure the labels are unique
            to ensure the correct output.
            """
        )

    start, stop, step = (
        columns_to_select.start,
        columns_to_select.stop,
        columns_to_select.step,
    )
    start_check = any((start is None, isinstance(start, str)))
    stop_check = any((stop is None, isinstance(stop, str)))
    step_check = any((step is None, isinstance(step, int)))
    if not start_check:
        raise ValueError(
            """
            The start value for the slice
            must either be a string or `None`.
            """
        )
    if not stop_check:
        raise ValueError(
            """
            The stop value for the slice
            must either be a string or `None`.
            """
        )
    if not step_check:
        raise ValueError(
            """
            The step value for the slice
            must either be an integer or `None`.
            """
        )
    start_check = any((start is None, start in df_columns))
    stop_check = any((stop is None, stop in df_columns))
    if not start_check:
        raise ValueError(
            """
            The start value for the slice must either be `None`
            or exist in the dataframe's columns.
            """
        )
    if not stop_check:
        raise ValueError(
            """
            The stop value for the slice must either be `None`
            or exist in the dataframe's columns.
            """
        )

    if start is None:
        start = 0
    else:
        start = df_columns.get_loc(start)
    if stop is None:
        stop = len(df_columns) + 1
    else:
        stop = df_columns.get_loc(stop)

    if start > stop:
        filtered_columns = df_columns[slice(stop, start + 1, step)][::-1]
    else:
        filtered_columns = df_columns[slice(start, stop + 1, step)]
    df_columns = None
    return [*filtered_columns]


@_select_columns.register(dispatch_callable)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to callables.
    The callable is applied to every column in the dataframe.
    Either True or False is expected per column.
    A list of column names is returned.
    """
    # the function will be applied per series.
    # this allows filtration based on the contents of the series
    # or based on the name of the series,
    # which happens to be a column name as well.
    # whatever the case may be,
    # the returned values should be a sequence of booleans,
    # with at least one True.

    filtered_columns = df.agg(columns_to_select)

    if not filtered_columns.any():
        raise ValueError(
            """
            No match was returned for the provided callable.
            """
        )

    return [*df.columns[filtered_columns]]


# hack to get it to recognize typing.Pattern
# functools.singledispatch does not natively
# recognize types from the typing module
# ``type(re.compile(r"\d+"))`` returns re.Pattern
# which is a type and functools.singledispatch
# accepts it without drama;
# however, the same type from typing.Pattern
# is not accepted.
@_select_columns.register(type(re.compile(r"\d+")))  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to regular expressions.
    `re.compile` is required for the regular expression.
    A list of column names is returned.
    """
    df_columns = df.columns
    filtered_columns = [
        column_name
        for column_name in df_columns
        if re.search(columns_to_select, column_name)
    ]

    if not filtered_columns:
        raise KeyError("No column name matched the regular expression.")
    df_columns = None

    return filtered_columns


@_select_columns.register(tuple)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    This caters to columns that are of tuple type.
    The tuple is returned as is, if it exists in the columns.
    """
    if columns_to_select not in df.columns:
        raise KeyError(f"No match was returned for {columns_to_select}")
    return columns_to_select


@_select_columns.register(list)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types,
    or a combination of these types.
    A tuple of column names is returned.
    """

    # takes care of boolean entries
    if all(map(pd.api.types.is_bool, columns_to_select)):
        if len(columns_to_select) != len(df.columns):
            raise ValueError(
                """
                The length of the list of booleans
                does not match the number of columns
                in the dataframe.
                """
            )

        return [*df.columns[columns_to_select]]

    filtered_columns = []
    columns_to_select = (
        _select_columns(entry, df) for entry in columns_to_select
    )

    # this is required,
    # to maintain `tuple` status
    # when combining all the entries into a single list
    columns_to_select = (
        [entry] if isinstance(entry, tuple) else entry
        for entry in columns_to_select
    )

    columns_to_select = chain.from_iterable(columns_to_select)

    # get rid of possible duplicates
    for column_name in columns_to_select:
        if column_name not in filtered_columns:
            filtered_columns.append(column_name)

    return filtered_columns


class JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


class JoinTypes(Enum):
    """
    List of join types for conditional_join.
    """

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"


def _check_operator(op: str):
    """
    Check that operator is one of
    `>`, `>=`, `==`, `!=`, `<`, `<=`.

    Used in `conditional_join`.
    """
    sequence_of_operators = {op.value for op in JoinOperator}
    if op not in sequence_of_operators:
        raise ValueError(
            f"""
             The conditional join operator
             should be one of {", ".join(sequence_of_operators)}
             """
        )


def _conditional_join_preliminary_checks(
    df: pd.DataFrame,
    right: Union[pd.DataFrame, pd.Series],
    conditions: tuple,
    how: str,
    sort_by_appearance: tuple,
) -> tuple:
    """
    Preliminary checks for conditional_join are conducted here.

    This function checks for conditions such as
    MultiIndexed dataframe columns,
    improper `suffixes` configuration,
    as well as unnamed Series.

    A tuple of
    (`df`, `right`, `left_on`, `right_on`, `operator`)
    is returned.
    """

    if df.empty:
        raise ValueError(
            """
            The dataframe on the left should not be empty.
            """
        )

    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            """
            MultiIndex columns are not
            supported for conditional_join.
            """
        )

    check("`right`", right, [pd.DataFrame, pd.Series])

    df = df.copy()
    right = right.copy()

    if isinstance(right, pd.Series):
        if not right.name:
            raise ValueError(
                """
                Unnamed Series are not supported
                for conditional_join.
                """
            )
        right = right.to_frame()

    if right.empty:
        raise ValueError(
            """
            The Pandas object on the right
            should not be empty.
            """
        )

    if isinstance(right.columns, pd.MultiIndex):
        raise ValueError(
            """
            MultiIndex columns are not supported
            for conditional joins.
            """
        )

    if not conditions:
        raise ValueError(
            """
            Kindly provide at least one join condition.
            """
        )

    for condition in conditions:
        check("condition", condition, [tuple])
        len_condition = len(condition)
        if len_condition != 3:
            raise ValueError(
                f"""
                condition should have only three elements.
                {condition} however is of length {len_condition}.
                """
            )

    for left_on, right_on, op in conditions:
        check("left_on", left_on, [str])
        check("right_on", right_on, [str])
        check("operator", op, [str])
        check_column(df, left_on)
        check_column(right, right_on)
        _check_operator(op)

    check("how", how, [str])

    join_types = {jointype.value for jointype in JoinTypes}
    if how not in join_types:
        raise ValueError(f"`how` should be one of {', '.join(join_types)}.")

    check("sort_by_appearance", sort_by_appearance, [bool])

    return df, right, conditions, how, sort_by_appearance


def _conditional_join_type_check(
    left_column: pd.Series, right_column: pd.Series, op: str
) -> None:
    """
    Raise error if column type is not
    any of numeric or datetime.
    """

    # Allow merges on strings/categoricals,
    # but only on the `==` operator?
    permitted_types = {
        is_datetime64_dtype,
        is_integer_dtype,
        is_float_dtype,
        is_string_dtype,
        is_categorical_dtype,
    }
    for func in permitted_types:
        if func(left_column):
            break
    else:
        raise ValueError(
            """
            conditional_join only supports
            string, category, integer,
            float or date dtypes.
            """
        )
    cols = (left_column, right_column)
    for func in permitted_types:
        if all(map(func, cols)):
            break
    else:
        raise ValueError(
            f"""
             Both columns should have the same type.
             `{left_column.name}` has {left_column.dtype} type;
             `{right_column.name}` has {right_column.dtype} type.
             """
        )

    if (
        is_categorical_dtype(left_column)
        and op != JoinOperator.STRICTLY_EQUAL.value
    ):
        raise ValueError(
            """
            For categorical columns,
            only the `==` operator is supported.
            """
        )

    if (
        is_string_dtype(left_column)
        and op != JoinOperator.STRICTLY_EQUAL.value
    ):
        raise ValueError(
            """
            For string columns,
            only the `==` operator is supported.
            """
        )

    return None


def _interval_ranges(indices: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Create `range` indices for each value in
    `right_keys` in  `_less_than_indices`
    and `_greater_than_indices`.

    It is faster than a list comprehension, especially
    for large arrays.

    code copied from Stack Overflow
    https://stackoverflow.com/a/47126435/7175713
    """
    cum_length = right - indices
    cum_length = cum_length.cumsum()
    # generate ones
    # note that cum_length[-1] is the total
    # number of index positions to be generated
    ids = np.ones(cum_length[-1], dtype=int)
    ids[0] = indices[0]
    # at each specific point in id, replace the value
    # so, we should have say 0, 1, 1, 1, 1, -5, 1, 1, 1, -3, ...
    # when a cumsum is implemented in the next line,
    # we get, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, ...
    # our ranges is obtained, with more efficiency
    # for larger arrays
    ids[cum_length[:-1]] = indices[1:] - right[:-1] + 1
    # the cumsum here gives us the same output as
    # [np.range(start, len_right) for start in search_indices]
    # but much faster
    return ids.cumsum()


def _equal_indices(
    left_c: Union[pd.Series, pd.DataFrame],
    right_c: Union[pd.Series, pd.DataFrame],
    len_conditions: int,
) -> tuple:
    """
    Use Pandas' merge internal functions
    to find the matches, if any.

    Returns a tuple of (left_c, right_c)
    """

    if isinstance(left_c, pd.Series):
        left_on = left_c.name
        right_on = right_c.name
    else:
        left_on = [*left_c.columns]
        right_on = [*right_c.columns]

    outcome = _MergeOperation(
        left=left_c,
        right=right_c,
        left_on=left_on,
        right_on=right_on,
        sort=False,
    )

    left_index, right_index = outcome._get_join_indexers()

    if not left_index.size > 0:
        return None

    if len_conditions > 1:
        return left_index, right_index

    return left_c.index[left_index], right_c.index[right_index]


def _not_equal_indices(left_c: pd.Series, right_c: pd.Series) -> tuple:
    """
    Use binary search to get indices where
    `left_c` is exactly  not equal to `right_c`.

    It is a combination of strictly less than
    and strictly greater than indices.

    Returns a tuple of (left_c, right_c)
    """

    dummy = np.array([], dtype=int)

    outcome = _less_than_indices(left_c, right_c, True, 1)

    if outcome is None:
        lt_left = dummy
        lt_right = dummy
    else:
        lt_left, lt_right = outcome

    outcome = _greater_than_indices(left_c, right_c, True, 1)

    if outcome is None:
        gt_left = dummy
        gt_right = dummy
    else:
        gt_left, gt_right = outcome

    if (not lt_left.size > 0) and (not gt_left.size > 0):
        return None
    left_c = np.concatenate([lt_left, gt_left])
    right_c = np.concatenate([lt_right, gt_right])

    return left_c, right_c


def _less_than_indices(
    left_c: pd.Series, right_c: pd.Series, strict: bool, len_conditions: int
) -> tuple:
    """
    Use binary search to get indices where left_c
    is less than or equal to right_c.

    If strict is True,then only indices
    where `left_c` is less than
    (but not equal to) `right_c` are returned.

    Returns a tuple of (left_c, right_c)
    """

    # no point going through all the hassle
    if left_c.min() > right_c.max():
        return None

    if right_c.hasnans:
        right_c = right_c.dropna()
    if not right_c.is_monotonic_increasing:
        right_c = right_c.sort_values()
    if left_c.hasnans:
        left_c = left_c.dropna()
    left_index = left_c.index.to_numpy(dtype=int)
    left_c = extract_array(left_c, extract_numpy=True)
    right_index = right_c.index.to_numpy(dtype=int)
    right_c = extract_array(right_c, extract_numpy=True)

    search_indices = right_c.searchsorted(left_c, side="left")
    # if any of the positions in `search_indices`
    # is equal to the length of `right_keys`
    # that means the respective position in `left_c`
    # has no values from `right_c` that are less than
    # or equal, and should therefore be discarded
    len_right = right_c.size
    rows_equal = search_indices == len_right

    if rows_equal.any():
        left_c = left_c[~rows_equal]
        left_index = left_index[~rows_equal]
        search_indices = search_indices[~rows_equal]

    if search_indices.size == 0:
        return None

    # the idea here is that if there are any equal values
    # shift upwards to the immediate next position
    # that is not equal
    if strict:
        rows_equal = right_c[search_indices]
        rows_equal = left_c == rows_equal
        # replace positions where rows are equal
        # with positions from searchsorted('right')
        # positions from searchsorted('right') will never
        # be equal and will be the furthermost in terms of position
        # example : right_c -> [2, 2, 2, 3], and we need
        # positions where values are not equal for 2;
        # the furthermost will be 3, and searchsorted('right')
        # will return position 3.
        if rows_equal.any():
            replacements = right_c.searchsorted(left_c, side="right")
            # now we can safely replace values
            # with strictly less than positions
            search_indices = np.where(rows_equal, replacements, search_indices)
        # check again if any of the values
        # have become equal to length of right_c
        # and get rid of them
        rows_equal = search_indices == len_right

        if rows_equal.any():
            left_c = left_c[~rows_equal]
            left_index = left_index[~rows_equal]
            search_indices = search_indices[~rows_equal]

    if search_indices.size == 0:
        return None

    indices = np.repeat(len_right, search_indices.size)

    if len_conditions > 1:
        return (left_index, right_index, search_indices, indices)

    positions = _interval_ranges(search_indices, indices)
    search_indices = indices - search_indices

    right_c = right_index[positions]
    left_c = left_index.repeat(search_indices)
    return left_c, right_c


def _greater_than_indices(
    left_c: pd.Series, right_c: pd.Series, strict: bool, len_conditions: int
) -> tuple:
    """
    Use binary search to get indices where left_c
    is greater than or equal to right_c.

    If strict is True,then only indices
    where `left_c` is greater than
    (but not equal to) `right_c` are returned.

    Returns a tuple of (left_c, right_c).
    """

    # quick break, avoiding the hassle
    if left_c.max() < right_c.min():
        return None

    if right_c.hasnans:
        right_c = right_c.dropna()
    if not right_c.is_monotonic_increasing:
        right_c = right_c.sort_values()
    if left_c.hasnans:
        left_c = left_c.dropna()
    left_index = left_c.index.to_numpy(dtype=int)
    left_c = extract_array(left_c, extract_numpy=True)
    right_index = right_c.index.to_numpy(dtype=int)
    right_c = extract_array(right_c, extract_numpy=True)

    search_indices = right_c.searchsorted(left_c, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0 (less than 1), it implies that
    # left_c[position] is not greater than any value
    # in right_c
    rows_equal = search_indices < 1
    if rows_equal.any():
        left_c = left_c[~rows_equal]
        left_index = left_index[~rows_equal]
        search_indices = search_indices[~rows_equal]
    if search_indices.size == 0:
        return None

    # the idea here is that if there are any equal values
    # shift downwards to the immediate next position
    # that is not equal
    if strict:
        rows_equal = right_c[search_indices - 1]
        rows_equal = left_c == rows_equal
        # replace positions where rows are equal with
        # searchsorted('left');
        # however there can be scenarios where positions
        # from searchsorted('left') would still be equal;
        # in that case, we shift down by 1
        if rows_equal.any():
            replacements = right_c.searchsorted(left_c, side="left")
            # return replacements
            # `left` might result in values equal to len right_c
            replacements = np.where(
                replacements == right_c.size, replacements - 1, replacements
            )
            # now we can safely replace values
            # with strictly greater than positions
            search_indices = np.where(rows_equal, replacements, search_indices)
        # any value less than 1 should be discarded
        rows_equal = search_indices < 1
        if rows_equal.any():
            left_c = left_c[~rows_equal]
            left_index = left_index[~rows_equal]
            search_indices = search_indices[~rows_equal]

    if search_indices.size == 0:
        return None

    indices = np.zeros(search_indices.size, dtype=np.int8)

    if len_conditions > 1:
        return (left_index, right_index, search_indices, indices)

    positions = _interval_ranges(indices, search_indices)
    right_c = right_index[positions]
    left_c = left_index.repeat(search_indices)
    return left_c, right_c


def _create_conditional_join_empty_frame(
    df: pd.DataFrame, right: pd.DataFrame, how: str
):
    """
    Create final dataframe for conditional join,
    if there are no matches.
    """

    df.columns = pd.MultiIndex.from_product([["left"], df.columns])
    right.columns = pd.MultiIndex.from_product([["right"], right.columns])

    if how == JoinTypes.INNER.value:
        df = df.dtypes.to_dict()
        right = right.dtypes.to_dict()
        df = {**df, **right}
        df = {key: pd.Series([], dtype=value) for key, value in df.items()}
        return pd.DataFrame(df)

    if how == JoinTypes.LEFT.value:
        right = right.dtypes.to_dict()
        right = {
            key: float if dtype.kind == "i" else dtype
            for key, dtype in right.items()
        }
        right = {
            key: pd.Series([], dtype=value) for key, value in right.items()
        }
        right = pd.DataFrame(right)
        return df.join(right, how=how, sort=False)

    if how == JoinTypes.RIGHT.value:
        df = df.dtypes.to_dict()
        df = {
            key: float if dtype.kind == "i" else dtype
            for key, dtype in df.items()
        }
        df = {key: pd.Series([], dtype=value) for key, value in df.items()}
        df = pd.DataFrame(df)
        return df.join(right, how=how, sort=False)


def _create_conditional_join_frame(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: pd.Index,
    right_index: pd.Index,
    how: str,
    sort_by_appearance: bool,
):
    """
    Create final dataframe for conditional join,
    if there are matches.
    """
    if sort_by_appearance:
        sorter = np.lexsort((right_index, left_index))
        right_index = right_index[sorter]
        left_index = left_index[sorter]

    df.columns = pd.MultiIndex.from_product([["left"], df.columns])
    right.columns = pd.MultiIndex.from_product([["right"], right.columns])

    if how == JoinTypes.INNER.value:
        df = df.loc[left_index]
        right = right.loc[right_index]
        df.index = pd.RangeIndex(start=0, stop=left_index.size)
        right.index = df.index
        return pd.concat([df, right], axis="columns", join=how, sort=False)

    if how == JoinTypes.LEFT.value:
        right = right.loc[right_index]
        right.index = left_index
        return df.join(right, how=how, sort=False).reset_index(drop=True)

    if how == JoinTypes.RIGHT.value:
        df = df.loc[left_index]
        df.index = right_index
        return df.join(right, how=how, sort=False).reset_index(drop=True)


less_than_join_types = {
    JoinOperator.LESS_THAN.value,
    JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    JoinOperator.GREATER_THAN.value,
    JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _generic_func_cond_join(
    left_c: pd.Series, right_c: pd.Series, op: str, len_conditions: int
):
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices, _equal_indices,
    or _not_equal_indices).
    """
    strict = False

    if op in {
        JoinOperator.GREATER_THAN.value,
        JoinOperator.LESS_THAN.value,
        JoinOperator.NOT_EQUAL.value,
    }:
        strict = True

    if op in less_than_join_types:
        return _less_than_indices(left_c, right_c, strict, len_conditions)
    elif op in greater_than_join_types:
        return _greater_than_indices(left_c, right_c, strict, len_conditions)
    elif op == JoinOperator.NOT_EQUAL.value:
        return _not_equal_indices(left_c, right_c)
    else:
        return _equal_indices(left_c, right_c, len_conditions)


def _conditional_join_compute(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    how: str,
    sort_by_appearance: bool,
) -> pd.DataFrame:
    """
    This is where the actual computation
    for the conditional join takes place.
    A pandas DataFrame is returned.
    """

    (
        df,
        right,
        conditions,
        how,
        sort_by_appearance,
    ) = _conditional_join_preliminary_checks(
        df, right, conditions, how, sort_by_appearance
    )

    eq_check = False
    less_great = False
    less_greater_types = less_than_join_types.union(greater_than_join_types)

    for condition in conditions:
        left_on, right_on, op = condition
        left_c = df[left_on]
        right_c = right[right_on]

        _conditional_join_type_check(left_c, right_c, op)

        if op == JoinOperator.STRICTLY_EQUAL.value:
            eq_check = True
        elif op in less_greater_types:
            less_great = True

    df.index = pd.RangeIndex(start=0, stop=len(df))
    right.index = pd.RangeIndex(start=0, stop=len(right))

    if len(conditions) == 1:
        left_on, right_on, op = conditions[0]

        left_c = df[left_on]
        right_c = right[right_on]

        if eq_check & left_c.hasnans:
            left_c = left_c.dropna()
        if eq_check & right_c.hasnans:
            right_c = right_c.dropna()

        result = _generic_func_cond_join(left_c, right_c, op, 1)

        if result is None:
            return _create_conditional_join_empty_frame(df, right, how)

        left_c, right_c = result

        return _create_conditional_join_frame(
            df, right, left_c, right_c, how, sort_by_appearance
        )

    # multiple conditions
    if eq_check:
        result = _multiple_conditional_join_eq(df, right, conditions)
    elif less_great:
        result = _multiple_conditional_join_le_lt(df, right, conditions)
    else:
        result = _multiple_conditional_join_ne(df, right, conditions)

    if result is None:
        return _create_conditional_join_empty_frame(df, right, how)

    left_c, right_c = result

    return _create_conditional_join_frame(
        df, right, left_c, right_c, how, sort_by_appearance
    )


operator_map = {
    JoinOperator.STRICTLY_EQUAL.value: operator.eq,
    JoinOperator.LESS_THAN.value: operator.lt,
    JoinOperator.LESS_THAN_OR_EQUAL.value: operator.le,
    JoinOperator.GREATER_THAN.value: operator.gt,
    JoinOperator.GREATER_THAN_OR_EQUAL.value: operator.ge,
    JoinOperator.NOT_EQUAL.value: operator.ne,
}


def _multiple_conditional_join_eq(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    if any of the conditions has an `==` operator.

    Returns a tuple of (df_index, right_index)
    """

    eq_cond = [
        cond
        for cond in conditions
        if cond[-1] == JoinOperator.STRICTLY_EQUAL.value
    ]
    rest = [
        cond
        for cond in conditions
        if cond[-1] != JoinOperator.STRICTLY_EQUAL.value
    ]

    # get rid of nulls, if any
    if len(eq_cond) == 1:
        left_on, right_on, _ = eq_cond[0]
        left_c = df.loc[:, left_on]
        right_c = right.loc[:, right_on]

        if left_c.hasnans:
            left_c = left_c.dropna()
            df = df.loc[left_c.index]

        if right_c.hasnans:
            right_c = right_c.dropna()
            right = right.loc[right_c.index]

    else:
        left_on, right_on, _ = zip(*eq_cond)
        left_c = df.loc[:, [*left_on]]
        right_c = right.loc[:, [*right_on]]

        if left_c.isna().any(axis=None):
            left_c = left_c.dropna()
            df = df.loc[left_c.index]

        if right_c.isna().any(axis=None):
            right_c = right_c.dropna()
            right = right.loc[right_c.index]

    # get join indices
    # these are positional, not label indices
    result = _generic_func_cond_join(
        left_c, right_c, JoinOperator.STRICTLY_EQUAL.value, 2
    )

    if result is None:
        return None

    df_index, right_index = result

    if not rest:
        return df.index[df_index], right.index[right_index]

    # non-equi conditions are present
    mask = None
    for left_on, right_on, op in rest:
        left_c = extract_array(df[left_on], extract_numpy=True)
        left_c = left_c[df_index]
        right_c = extract_array(right[right_on], extract_numpy=True)
        right_c = right_c[right_index]

        op = operator_map[op]
        if mask is None:
            mask = op(left_c, right_c)
        else:
            mask &= op(left_c, right_c)

    if not mask.any():
        return None

    df_index = df_index[mask]
    right_index = right_index[mask]

    return df.index[df_index], right.index[right_index]


def _multiple_conditional_join_ne(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    where all the operators are `!=`.

    Returns a tuple of (df_index, right_index)
    """

    first, *rest = conditions
    left_on, right_on, op = first
    left_c = df[left_on]
    right_c = right[right_on]
    result = _generic_func_cond_join(left_c, right_c, op, 1)
    if result is None:
        return None

    df_index, right_index = result

    mask = None
    for left_on, right_on, op in rest:
        left_c = df.loc[df_index, left_on]
        left_c = extract_array(left_c, extract_numpy=True)
        right_c = right.loc[right_index, right_on]
        right_c = extract_array(right_c, extract_numpy=True)
        op = operator_map[op]

        if mask is None:
            mask = op(left_c, right_c)
        else:
            mask &= op(left_c, right_c)

    if not mask.any():
        return None
    return df_index[mask], right_index[mask]


def _multiple_conditional_join_le_lt(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list
) -> tuple:
    """
    Get indices for multiple conditions,
    if there is no `==` operator, and there is
    at least one `<`, `<=`, `>`, or `>=` operator.

    Returns a tuple of (df_index, right_index)
    """

    # find minimum df_index and right_index
    # aim is to reduce search space
    df_index = df.index
    right_index = right.index
    lt_gt = None
    less_greater_types = less_than_join_types.union(greater_than_join_types)
    for left_on, right_on, op in conditions:
        if op in less_greater_types:
            lt_gt = left_on, right_on, op
        # no point checking for `!=`, since best case scenario
        # they'll have the same no of rows for the less/greater operators
        elif op == JoinOperator.NOT_EQUAL.value:
            continue

        left_c = df.loc[df_index, left_on]
        right_c = right.loc[right_index, right_on]

        result = _generic_func_cond_join(left_c, right_c, op, 2)

        if result is None:
            return None

        df_index, right_index, *_ = result

    # move le,lt,ge,gt to the fore
    # less rows to search, compared to !=
    if conditions[0][-1] not in less_greater_types:
        conditions = [*conditions]
        conditions.remove(lt_gt)
        conditions = [lt_gt] + conditions

    first, *rest = conditions
    left_on, right_on, op = first
    left_c = df.loc[df_index, left_on]
    right_c = right.loc[right_index, right_on]

    result = _generic_func_cond_join(left_c, right_c, op, 2)

    if result is None:
        return None

    df_index, right_index, search_indices, indices = result
    if op in less_than_join_types:
        low, high = search_indices, indices
    else:
        low, high = indices, search_indices

    first, *rest = rest
    left_on, right_on, op = first
    left_c = df.loc[df_index, left_on]
    left_c = extract_array(left_c, extract_numpy=True)
    right_c = right.loc[right_index, right_on]
    right_c = extract_array(right_c, extract_numpy=True)
    op = operator_map[op]
    index_df = []
    repeater = []
    index_right = []
    # offers a bit of a speed up, compared to broadcasting
    # we go through each search space, and keep only matching rows
    # constrained to just one loop;
    # if the join conditions are limited to two, this is helpful;
    # for more than two, then broadcasting kicks in after this step
    # running this within numba should offer more speed
    for indx, val, lo, hi in zip(df_index, left_c, low, high):
        search = right_c[lo:hi]
        indexer = right_index[lo:hi]
        mask = op(val, search)
        if not mask.any():
            continue
        # pandas boolean arrays do not play well with numpy
        # hence the conversion
        if is_extension_array_dtype(mask):
            mask = mask.to_numpy(dtype=bool, na_value=False)
        indexer = indexer[mask]
        index_df.append(indx)
        index_right.append(indexer)
        repeater.append(indexer.size)

    if not index_df:
        return None

    df_index = np.repeat(index_df, repeater)
    right_index = np.concatenate(index_right)

    if not rest:
        return df_index, right_index

    # blow it up
    mask = None
    for left_on, right_on, op in rest:
        left_c = df.loc[df_index, left_on]
        left_c = extract_array(left_c, extract_numpy=True)
        right_c = right.loc[right_index, right_on]
        right_c = extract_array(right_c, extract_numpy=True)
        op = operator_map[op]

        if mask is None:
            mask = op(left_c, right_c)
        else:
            mask &= op(left_c, right_c)

    if not mask.any():
        return None
    if is_extension_array_dtype(mask):
        mask = mask.to_numpy(dtype=bool, na_value=False)

    return df_index[mask], right_index[mask]


def _case_when_checks(df: pd.DataFrame, args, column_name):
    """
    Preliminary checks on the case_when function.
    """
    if len(args) < 3:
        raise ValueError(
            """
            At least three arguments are required
            for the `args` parameter.
            """
        )
    if len(args) % 2 != 1:
        raise ValueError(
            """
            It seems the `default` argument is missing
            from the variable `args` parameter.
            """
        )

    check("column_name", column_name, [str])

    *args, default = args

    booleans = []
    replacements = []
    for index, value in enumerate(args):
        if index % 2 == 0:
            booleans.append(value)
        else:
            replacements.append(value)

    conditions = []
    for condition in booleans:
        if callable(condition):
            condition = apply_if_callable(condition, df)
        elif isinstance(condition, str):
            condition = df.eval(condition)
        conditions.append(condition)

    targets = []
    for replacement in replacements:
        if callable(replacement):
            replacement = apply_if_callable(replacement, df)
        targets.append(replacement)

    if callable(default):
        default = apply_if_callable(default, df)
    if not is_list_like(default):
        default = pd.Series([default]).repeat(len(df))
        default.index = df.index
    if not hasattr(default, "shape"):
        default = pd.Series([*default])
    if isinstance(default, pd.Index):
        arr_ndim = default.nlevels
    else:
        arr_ndim = default.ndim
    if arr_ndim != 1:
        raise ValueError(
            """
            The `default` argument should either be a 1-D array,
            a scalar, or a callable that can evaluate to
            a 1-D array.
            """
        )
    if not isinstance(default, pd.Series):
        default = pd.Series(default)
    if default.size != len(df):
        raise ValueError(
            """
            The length of the `default` argument
            should be equal to the length
            of the DataFrame.
            """
        )
    return conditions, targets, default


def _case_when(df: pd.DataFrame, args, column_name):
    """
    Actual computation of the case_when function.
    """
    conditions, targets, default = _case_when_checks(df, args, column_name)

    if len(conditions) == 1:
        default = default.mask(conditions[0], targets[0])
        return df.assign(**{column_name: default})

    # ensures value assignment is on a first come basis
    conditions = conditions[::-1]
    targets = targets[::-1]
    for condition, value, index in zip(conditions, targets, count()):
        try:
            default = default.mask(condition, value)
        # error `feedoff` idea from SO
        # https://stackoverflow.com/a/46091127/7175713
        except Exception as e:
            raise ValueError(
                f"""
                condition{index} and value{index}
                failed to evaluate.
                Original error message: {e}
                """
            ) from e

    return df.assign(**{column_name: default})
