"""Miscellaneous internal PyJanitor helper functions."""

import fnmatch
import functools
import os
import re
import socket
import sys
import warnings
from collections.abc import Callable as dispatch_callable
from itertools import chain, combinations, tee
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_extension_array_dtype,
    is_list_like,
    is_scalar,
    is_numeric_dtype,
    is_string_dtype,
    is_datetime64_dtype,
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

    There is a performance penalty for small entries (length less than 10)
    in using this method, instead of `itertools.product`; however, there is
    significant performance benefits as the size of the data increases.

    Another benefit of this approach,
    in addition to the significant performance gains,
    is the preservation of data types. This is particularly relevant for
    Pandas' extension arrays dtypes (categoricals, nullable integers, ...).

    A dataframe of all possible combinations is returned.
    """

    for key, _ in others.items():
        check("key", key, [str])

    grid = {}

    for key, value in others.items():
        if is_scalar(value):
            grid[key] = pd.Series([value])
        elif is_extension_array_dtype(value) and not (
            isinstance(value, pd.Series)
        ):
            grid[key] = pd.Series(value)
        elif is_list_like(value):
            if not isinstance(
                value, (pd.DataFrame, pd.Series, np.ndarray, list, pd.Index)
            ):
                grid[key] = list(value)
            else:
                grid[key] = value

    others = None

    mgrid_values = [slice(len(value)) for _, value in grid.items()]
    mgrid_values = np.mgrid[mgrid_values]
    mgrid_values = map(np.ravel, mgrid_values)
    grid = zip([*grid.items()], mgrid_values)
    grid = ((*left, right) for left, right in grid)
    grid = (
        _expand_grid(value, key, mgrid_values)
        for key, value, mgrid_values in grid
    )

    grid = pd.concat(grid, axis="columns", sort=False)

    return grid


@functools.singledispatch
def _expand_grid(value, key, mgrid_values, mode="expand_grid"):
    """
    Base function for dispatch of `_expand_grid`.

    `mode` parameter is added, to make the function reusable
    in the `_computations_complete` function.
    Also, allowing `key` as None enables reuse in the
    `_computations_complete` function.
    """

    raise TypeError(
        f"{type(value).__name__} data type is not supported in `expand_grid`."
    )


@_expand_grid.register(list)  # noqa: F811
def _sub_expand_grid(value, key, mgrid_values):  # noqa: F811
    """
    Expands the list object based on `mgrid_values`.
    Converts to an array and passes it
    to the `_expand_grid` function for arrays.
    `mode` parameter is added, to make the function reusable
    in the `_computations_complete` function.
    Also, allowing `key` as None enables reuse in the
    `_computations_complete` function.
    Returns Series with name if 1-Dimensional array
    or DataFrame if 2-Dimensional array with column names.
    """
    if not value:
        raise ValueError("""list object cannot be empty.""")
    value = np.array(value)
    return _expand_grid(value, key, mgrid_values)


@_expand_grid.register(np.ndarray)
def _sub_expand_grid(  # noqa: F811
    value, key, mgrid_values, mode="expand_grid"
):
    """
    Expands the numpy array based on `mgrid_values`.

    Ensures array dimension is either 1 or 2.

    `mode` parameter is added, to make the function reusable
    in the `_computations_complete` function.
    Also, allowing `key` as None enables reuse in the
    `_computations_complete` function.

    Returns Series with name if 1-Dimensional array
    or DataFrame if 2-Dimensional array with column names.

    The names are derived from the `key` parameter.
    """
    if not (value.size > 0):
        raise ValueError("""array cannot be empty.""")
    if value.ndim > 2:
        raise ValueError("""expand_grid works only on 1D and 2D structures.""")

    value = value.take(mgrid_values, axis=0)

    if value.ndim == 1:
        value = pd.Series(value)
        # a tiny bit faster than chaining with `rename`
        value.name = key
    else:
        value = pd.DataFrame(value)
        # a tiny bit faster than using `add_prefix`
        value.columns = value.columns.map(lambda column: f"{key}_{column}")

    return value


@_expand_grid.register(pd.Series)
def _sub_expand_grid(  # noqa: F811
    value, key, mgrid_values, mode="expand_grid"
):
    """
    Expands the Series based on `mgrid_values`.

    `mode` parameter is added, to make the function reusable
    in the `_computations_complete` function.
    Also, allowing `key` as None enables reuse in the
    `_computations_complete` function.

    Checks for empty Series and returns modified keys.
    Returns Series with new Series name.
    """
    if value.empty:
        raise ValueError("""Series cannot be empty.""")

    value = value.take(mgrid_values)
    value.index = np.arange(len(value))

    if mode != "expand_grid":
        return value

    if value.name:
        value.name = f"{key}_{value.name}"
    else:
        value.name = key
    return value


@_expand_grid.register(pd.DataFrame)
def _sub_expand_grid(  # noqa: F811
    value, key, mgrid_values, mode="expand_grid"
):
    """
    Expands the DataFrame based on `mgrid_values`.

    `mode` parameter is added, to make the function reusable
    in the `_computations_complete` function.
    Also, allowing `key` as None enables reuse in the
    `_computations_complete` function.

    Checks for empty dataframe and returns modified keys.

    Returns a DataFrame with new column names.
    """
    if value.empty:
        raise ValueError("""DataFrame cannot be empty.""")

    value = value.take(mgrid_values)
    value.index = np.arange(len(value))

    if mode != "expand_grid":
        return value

    if isinstance(value.columns, pd.MultiIndex):
        value.columns = [f"{key}_{num}" for num, _ in enumerate(value.columns)]
    else:
        value.columns = value.columns.map(lambda column: f"{key}_{column}")

    return value


@_expand_grid.register(pd.Index)
def _sub_expand_grid(  # noqa: F811
    value, key, mgrid_values, mode="expand_grid"
):
    """
    Expands the Index based on `mgrid_values`.

    `mode` parameter is added, to make the function reusable
    in the `_computations_complete` function.
    Also, allowing `key` as None enables reuse in the
    `_computations_complete` function.

    Checks for empty Index and returns modified keys.

    Returns a DataFrame (if MultiIndex) with new column names,
    or a Series with a new name.
    """
    if value.empty:
        raise ValueError("""Index cannot be empty.""")

    value = value.take(mgrid_values)

    if mode != "expand_grid":
        return value

    if isinstance(value, pd.MultiIndex):
        value = value.to_frame(index=False)
        value.columns = value.columns.map(lambda column: f"{key}_{column}")
    else:
        value = value.to_series(index=np.arange(len(value)))
        if value.name:
            value.name = f"{key}_{value.name}"
        else:
            value.name = key

    return value


def _data_checks_complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]],
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

    if by is not None:
        if isinstance(by, str):
            by = [by]
        check("by", by, [list])

    return df, columns, column_checker, by


def _computations_complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]],
    by: Optional[Union[list, str]] = None,
) -> pd.DataFrame:
    """
    This function computes the final output for the `complete` function.

    If `by` is present, then groupby apply is used.

    For some cases, the `stack/unstack` combination is preferred; it is more
    efficient than `reindex`, as the size of the data grows. It is only
    applicable if all the entries in `columns` are strings, there are
    no nulls(stacking implicitly removes nulls in columns),
    the length of `columns` is greater than 1, and the index
    has no duplicates.

    If there is a dictionary in `columns`, it is possible that all the values
    of a key, or keys, may not be in the existing column with the same key(s);
    as such, a union of the current index and the generated index is executed,
    to ensure that all combinations are in the final dataframe.

    A dataframe, with rows of missing values, if any, is returned.
    """

    df, columns, column_checker, by = _data_checks_complete(df, columns, by)

    dict_present = any((isinstance(entry, dict) for entry in columns))
    all_strings = all(isinstance(column, str) for column in columns)

    df = df.set_index(column_checker)

    df_index = df.index
    df_names = df_index.names

    any_nulls = any(
        df_index.get_level_values(name).hasnans for name in df_names
    )

    if not by:

        df = _base_complete(df, columns, all_strings, any_nulls, dict_present)

    # a better (and faster) way would be to create a dataframe
    # from the groupby ...
    # solution here got me thinking
    # https://stackoverflow.com/a/66667034/7175713
    # still thinking on how to improve speed of groupby apply
    else:
        df = df.groupby(by).apply(
            _base_complete,
            columns,
            all_strings,
            any_nulls,
            dict_present,
        )
        df = df.drop(columns=by)

    df = df.reset_index()

    return df


def _base_complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]],
    all_strings: bool,
    any_nulls: bool,
    dict_present: bool,
) -> pd.DataFrame:

    df_empty = df.empty
    df_index = df.index
    unique_index = df_index.is_unique
    columns_to_stack = None

    # stack...unstack implemented here if conditions are met
    # usually faster than reindex
    if all_strings and (not any_nulls) and (len(columns) > 1) and unique_index:
        if df_empty:
            df["dummy"] = 1

        columns_to_stack = columns[1:]
        df = df.unstack(columns_to_stack)  # noqa: PD010
        df = df.stack(columns_to_stack, dropna=False)  # noqa: PD013
        if df_empty:
            df = df.drop(columns="dummy")
        columns_to_stack = None
        return df

    indexer = _create_indexer_for_complete(df_index, columns)

    if unique_index:
        if dict_present:
            indexer = df_index.union(indexer, sort=None)
        df = df.reindex(indexer)

    else:  # reindex not possible on duplicate indices
        df = df.join(pd.DataFrame([], index=indexer), how="outer")

    return df


def _create_indexer_for_complete(
    df_index: pd.Index,
    columns: List[Union[List, Dict, str]],
) -> pd.DataFrame:
    """
    This creates the index that will be used
    to expand the dataframe in the `complete` function.

    A pandas Index is returned.
    """

    complete_columns = (
        _complete_column(column, df_index) for column in columns
    )

    complete_columns = (
        (entry,) if not isinstance(entry, list) else entry
        for entry in complete_columns
    )
    complete_columns = chain.from_iterable(complete_columns)
    indexer = [*complete_columns]

    if len(indexer) > 1:
        indexer = _complete_indexer_expand_grid(indexer)

    else:
        indexer = indexer[0]

    return indexer


def _complete_indexer_expand_grid(indexer):
    """
    Generate indices to expose explicitly missing values,
    using the `expand_grid` function.

    Returns a pandas Index.
    """
    indexers = []
    mgrid_values = [slice(len(value)) for value in indexer]
    mgrid_values = np.mgrid[mgrid_values]
    mgrid_values = map(np.ravel, mgrid_values)

    indexer = zip(indexer, mgrid_values)
    indexer = (
        _expand_grid(value, None, mgrid_values, mode=None)
        for value, mgrid_values in indexer
    )

    for entry in indexer:
        if isinstance(entry, pd.MultiIndex):
            names = entry.names
            val = (entry.get_level_values(name) for name in names)
            indexers.extend(val)
        else:
            indexers.append(entry)
    indexer = pd.MultiIndex.from_arrays(indexers)
    indexers = None
    return indexer


@functools.singledispatch
def _complete_column(column, index):
    """
    This function processes the `columns` argument,
    to create a pandas Index or a list.

    Args:
        column : str/list/dict
        index: pandas Index

    A unique pandas Index or a list of unique pandas Indices is returned.
    """
    raise TypeError(
        """This type is not supported in the `complete` function."""
    )


@_complete_column.register(str)  # noqa: F811
def _sub_complete_column(column, index):  # noqa: F811
    """
    This function processes the `columns` argument,
    to create a pandas Index.

    Args:
        column : str
        index: pandas Index

    Returns:
        pd.Index: A pandas Index with a single level
    """

    arr = index.get_level_values(column)

    if not arr.is_unique:
        arr = arr.drop_duplicates()
    return arr


@_complete_column.register(list)  # noqa: F811
def _sub_complete_column(column, index):  # noqa: F811
    """
    This function processes the `columns` argument,
    to create a pandas Index.

    Args:
        column : list
        index: pandas Index

    Returns:
        pd.MultiIndex
    """

    level_to_drop = [name for name in index.names if name not in column]
    arr = index.droplevel(level_to_drop)
    if not arr.is_unique:
        return arr.drop_duplicates()
    return arr


@_complete_column.register(dict)  # noqa: F811
def _sub_complete_column(column, index):  # noqa: F811
    """
    This function processes the `columns` argument,
    to create a pandas Index or a list.

    Args:
        column : dict
        index: pandas Index

    Returns:
        list: A list of unique pandas Indices.
    """

    collection = []
    for key, value in column.items():
        arr = apply_if_callable(value, index.get_level_values(key))
        if not is_list_like(arr):
            raise ValueError(
                """
                Input in the supplied dictionary
                must be list-like.
                """
            )
        if (
            not isinstance(
                arr, (pd.DataFrame, pd.Series, np.ndarray, pd.Index)
            )
        ) and (not is_extension_array_dtype(arr)):
            arr = pd.Index([*arr], name=key)

        if arr.ndim != 1:
            raise ValueError(
                """
                It seems the supplied pair in the supplied dictionary
                cannot be converted to a 1-dimensional Pandas object.
                Kindly provide data that can be converted to
                a 1-dimensional Pandas object.
                """
            )
        if isinstance(arr, pd.MultiIndex):
            raise ValueError(
                """
                MultiIndex object not acceptable
                in the supplied dictionary.
                """
            )

        if not isinstance(arr, pd.Index):
            arr = pd.Index(arr, name=key)

        if arr.empty:
            raise ValueError(
                """
                Input in the supplied dictionary
                cannot be empty.
                """
            )

        if not arr.is_unique:
            arr = arr.drop_duplicates()

        if arr.name is None:
            arr.name = key

        collection.append(arr)

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

    if isinstance(names_to, str):
        names_to = [names_to]

    elif isinstance(names_to, tuple):
        names_to = list(names_to)

    check("names_to", names_to, [list])

    if not all((isinstance(word, str) for word in names_to)):
        raise TypeError("All entries in `names_to` argument must be strings.")

    if len(names_to) > 1:
        if all((names_pattern, names_sep)):
            raise ValueError(
                """
                Only one of `names_pattern` or `names_sep`
                should be provided.
                """
            )

        if (".value" in names_to) and (names_to.count(".value") > 1):
            raise ValueError("There can be only one `.value` in `names_to`.")

    # names_sep creates more than one column
    # whereas regex with names_pattern can be limited to one column
    if (len(names_to) == 1) and (names_sep is not None):
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
                    All entries in the ``names_pattern`` argument
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
                    ``.value`` is not accepted
                    if ``names_pattern``
                    is a list/tuple.
                    """
                )

    if names_sep is not None:
        check("names_sep", names_sep, [str, Pattern])

    check("values_to", values_to, [str])

    if (values_to in df.columns) and not any(
        (
            ".value" in names_to,
            isinstance(names_pattern, (list, tuple)),
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

    if any((names_sep, names_pattern)) and (
        isinstance(df.columns, pd.MultiIndex)
    ):
        raise ValueError(
            """
            Unpivoting a MultiIndex column dataframe
            when `names_sep` or `names_pattern` is supplied
            is not supported.
            """
        )

    if all((names_sep is None, names_pattern is None)):
        # adapted from pandas' melt source code
        if (
            (index is not None)
            and isinstance(df.columns, pd.MultiIndex)
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
            and isinstance(df.columns, pd.MultiIndex)
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
    df: pd.DataFrame, ignore_index: bool, len_index: int
) -> pd.DataFrame:
    """
    This function sorts the resulting dataframe by appearance,
    via the `sort_by_appearance` parameter in `computations_pivot_longer`.

    An example for `sort_by_appearance`:

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

    A dataframe that is sorted by appearance is returned.
    """

    index_sorter = None

    # if the height of the new dataframe
    # is the same as the height of the original dataframe,
    # then there is no need to sort by appearance
    length_check = any((len_index == 1, len_index == len(df)))

    if not length_check:
        index_sorter = np.reshape(np.arange(len(df)), (-1, len_index)).ravel(
            order="F"
        )
        df = df.take(index_sorter)

        if ignore_index:
            df.index = np.arange(len(df))

    return df


def _pivot_longer_extractions(
    df: pd.DataFrame,
    index: Optional[Union[List, Tuple]] = None,
    column_names: Optional[Union[List, Tuple]] = None,
    names_to: Optional[List] = None,
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

    A dataframe is returned.
    """

    if any((names_sep, names_pattern)):
        if index:
            df = df.set_index(index, append=True)

        if column_names:
            df = df.loc[:, column_names]

    mapping = None
    if names_sep:
        mapping = df.columns.str.split(names_sep, expand=True)

        if len(mapping.names) != len(names_to):
            raise ValueError(
                """
                The length of ``names_to`` does not match
                the number of columns extracted.
                """
            )
        mapping.names = names_to

    elif isinstance(names_pattern, str):
        mapping = df.columns.str.extract(names_pattern, expand=True)

        if mapping.isna().all(axis=None):
            raise ValueError(
                """
                No labels in the columns
                matched the regular expression
                in ``names_pattern``.
                Kindly provide a regular expression
                that matches all labels in the columns.
                """
            )

        if mapping.isna().any(axis=None):
            raise ValueError(
                """
                Not all labels in the columns
                matched the regular expression
                in ``names_pattern``.
                Kindly provide a regular expression
                that matches all labels in the columns.
                """
            )

        if len(names_to) != len(mapping.columns):
            raise ValueError(
                """
                The length of ``names_to`` does not match
                the number of columns extracted.
                """
            )

        if len(mapping.columns) == 1:
            mapping = pd.Index(mapping.iloc[:, 0], name=names_to[0])
        else:
            mapping = pd.MultiIndex.from_frame(mapping, names=names_to)

    elif isinstance(names_pattern, (list, tuple)):
        mapping = [
            df.columns.str.contains(regex, na=False) for regex in names_pattern
        ]

        if not np.any(mapping):
            raise ValueError(
                """
                Not all labels in the columns
                matched the regular expression
                in ``names_pattern``.
                Kindly provide a regular expression
                that matches all labels in the columns.
                """
            )

        mapping = np.select(mapping, names_to, None)
        mapping = pd.Index(mapping, name=".value")

        if np.any(mapping.isna()):
            raise ValueError(
                """
                The regular expressions in ``names_pattern``
                did not return all matches.
                Kindly provide a regular expression that
                captures all patterns.
                """
            )

    outcome = None
    single_index_mapping = not isinstance(mapping, pd.MultiIndex)
    if single_index_mapping:
        outcome = pd.Series(mapping)
        outcome = outcome.groupby(outcome).cumcount()
        mapping = pd.MultiIndex.from_arrays([mapping, outcome])
        outcome = None

    df.columns = mapping

    dot_value = any(
        ((".value" in names_to), isinstance(names_pattern, (list, tuple)))
    )

    first = None
    last = None
    complete_index = None
    dtypes = None
    cumcount = None
    if dot_value:
        if not mapping.is_unique:
            cumcount = pd.factorize(mapping)[0]
            cumcount = pd.Series(cumcount).groupby(cumcount).cumcount()
        cumcount_check = cumcount is not None
        mapping_names = mapping.names
        mapping = [mapping.get_level_values(name) for name in mapping_names]
        dtypes = [
            CategoricalDtype(categories=entry.unique(), ordered=True)
            for entry in mapping
        ]
        mapping = [
            entry.astype(dtype) for entry, dtype in zip(mapping, dtypes)
        ]

        if cumcount_check:
            mapping.append(cumcount)
        mapping = pd.MultiIndex.from_arrays(mapping)
        df.columns = mapping

        mapping = df.columns
        if cumcount_check:
            mapping = mapping.droplevel(-1)

        # test if all combinations are present
        first = mapping.get_level_values(".value")
        last = mapping.droplevel(".value")
        outcome = first.groupby(last)
        outcome = (value for _, value in outcome.items())
        outcome = combinations(outcome, 2)
        outcome = (
            left.symmetric_difference(right).empty for left, right in outcome
        )

        # include all combinations into the columns
        if not all(outcome):
            if isinstance(last, pd.MultiIndex):
                indexer = (first.drop_duplicates(), last.drop_duplicates())
                complete_index = _complete_indexer_expand_grid(indexer)
                complete_index = complete_index.reorder_levels(
                    [*mapping.names]
                )

            else:
                complete_index = pd.MultiIndex.from_product(mapping.levels)

            df = df.reindex(columns=complete_index)
        if cumcount_check:
            df = df.droplevel(-1, axis=1)

    return df, single_index_mapping


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

    if (
        (index is None)
        and column_names
        and (len(df.columns) > len(column_names))
    ):
        index = [
            column_name
            for column_name in df
            if column_name not in column_names
        ]

    len_index = len(df)

    # scenario 1
    if all((names_pattern is None, names_sep is None)):

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
            df = _sort_by_appearance_for_melt(
                df=df, ignore_index=ignore_index, len_index=len_index
            )

        return df

    df, single_index_mapping = _pivot_longer_extractions(
        df=df,
        index=index,
        column_names=column_names,
        names_to=names_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
    )

    # df_columns = df.columns
    unique_names = None
    drop_column = None
    dot_value = ".value" in df.columns.names

    if not dot_value:
        if single_index_mapping:
            unique_names = df.columns.names[0]
            drop_column = "_".join([unique_names, values_to])
            df.columns.names = [unique_names, drop_column]
        df = pd.melt(
            df, id_vars=None, value_name=values_to, ignore_index=False
        )

    else:
        unique_names = df.columns.get_level_values(".value").categories
        if single_index_mapping:
            # passing `drop_column` to `melt` downstream
            # avoids any name conflict with `var_name`,
            # especially if var_name exists in the names
            # associated with .value.
            drop_column = "_".join(unique_names)
        # ensures that the correct values are aligned,
        # in preparation for the recombination
        # of the columns downstream
        df = df.sort_index(axis=1)

        df = [
            df.xs(key=name, level=".value", axis=1).melt(
                ignore_index=False, var_name=drop_column, value_name=name
            )
            for name in unique_names
        ]

        first, *rest = df

        # `first` has all the required columns;
        # as such, there is no need to keep these columns in
        # the other dataframes in `rest`;
        # plus, we avoid duplicate columns during concatenation
        # the only column we need is the last column,
        # from each dataframe in `rest`
        # uniformity in the data is already assured
        # with the categorical dtype creation,
        # followed by the sorting on the columns earlier.
        rest = [frame.iloc[:, -1] for frame in rest]
        # df = first.join(rest, how = 'outer', sort = False)
        df = pd.concat([first, *rest], axis=1)

    if single_index_mapping:
        df = df.drop(columns=drop_column)

    if index:
        df = df.reset_index(level=index)

    if sort_by_appearance:
        df = _sort_by_appearance_for_melt(
            df=df, ignore_index=ignore_index, len_index=len_index
        )

    elif ignore_index:
        df.index = np.arange(len(df))

    return df


def _data_checks_pivot_wider(
    df,
    index,
    names_from,
    values_from,
    names_sort,
    flatten_levels,
    names_from_position,
    names_prefix,
    names_sep,
    aggfunc,
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
        if is_list_like(index):
            index = list(index)
        index = _select_columns(index, df)

    if names_from is None:
        raise ValueError(
            "pivot_wider() missing 1 required argument: 'names_from'"
        )

    if is_list_like(names_from):
        names_from = list(names_from)
    names_from = _select_columns(names_from, df)

    if values_from is not None:
        check("values_from", values_from, [list, str])
        values_from = _select_columns(values_from, df)

    check("names_sort", names_sort, [bool])

    check("flatten_levels", flatten_levels, [bool])

    if names_from_position is not None:
        check("names_from_position", names_from_position, [str])
        if names_from_position not in ("first", "last"):
            raise ValueError(
                """
                The position of `names_from`
                must be either "first" or "last".
                """
            )

    if names_prefix is not None:
        check("names_prefix", names_prefix, [str])

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    if aggfunc is not None:
        check("aggfunc", aggfunc, [str, list, dict, callable])

    if fill_value is not None:
        check("fill_value", fill_value, [int, float, str])

    return (
        df,
        index,
        names_from,
        values_from,
        names_sort,
        flatten_levels,
        names_from_position,
        names_prefix,
        names_sep,
        aggfunc,
        fill_value,
    )


def _computations_pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[List, str]] = None,
    names_from: Optional[Union[List, str]] = None,
    values_from: Optional[Union[List, str]] = None,
    names_sort: Optional[bool] = False,
    flatten_levels: Optional[bool] = True,
    names_from_position: Optional[str] = "first",
    names_prefix: Optional[str] = None,
    names_sep: Optional[str] = "_",
    aggfunc: Optional[Union[str, list, dict, Callable]] = None,
    fill_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """
    This is the main workhorse of the `pivot_wider` function.

    By default, values from `names_from` are at the front of
    each output column. If there are multiple `values_from`,
    this can be changed via the `names_from_position`,
    by setting it to `last`.

    A dataframe is returned.
    """

    if not names_sort:
        # Categorical dtypes created only for `names_from`
        # since that is what will become the new column names
        dtypes = {
            column_name: CategoricalDtype(
                categories=column.dropna().unique(), ordered=True
            )
            if column.hasnans
            else CategoricalDtype(categories=column.unique(), ordered=True)
            for column_name, column in df.filter(names_from).items()
        }

        df = df.astype(dtypes)

    if aggfunc is None:
        df = df.pivot(  # noqa: PD010
            index=index, columns=names_from, values=values_from
        )

    else:
        if index:
            df = df.set_index(index + names_from)
        else:
            df = df.set_index(names_from, append=True)

        if values_from:
            df = df.groupby(
                level=[*range(df.index.nlevels)],
                observed=True,
                dropna=False,
                sort=False,
            )[values_from].agg(aggfunc)
        else:
            df = df.groupby(
                level=[*range(df.index.nlevels)],
                observed=True,
                dropna=False,
                sort=False,
            ).agg(aggfunc)

        df = df.unstack(level=names_from)  # noqa: PD010

    if fill_value is not None:
        df = df.fillna(fill_value)

    # no point keeping `values_from`
    # if it's just one name;
    # could do same for aggfunc; but not worth the extra check
    if df.columns.get_level_values(0).unique().size == 1:
        df = df.droplevel(0, axis="columns")

    other_levels = None
    names_from_levels = None
    df_columns = df.columns
    df_columns_names = df_columns.names
    if names_from_position == "first":
        if df_columns_names[: len(names_from)] != names_from:
            other_levels = [
                num
                for num, name in enumerate(df_columns_names)
                if name not in names_from
            ]
            names_from_levels = [
                num
                for num, name in enumerate(df_columns_names)
                if name in names_from
            ]
            df = df.reorder_levels(
                names_from_levels + other_levels, axis="columns"
            )

    if not flatten_levels:
        return df

    if df_columns.nlevels > 1:
        df.columns = [names_sep.join(column_tuples) for column_tuples in df]

    if names_prefix:
        df = df.add_prefix(names_prefix)

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


class asCategorical(NamedTuple):
    """
    Helper class for `encode_categorical`. It makes creating the
    `categories` and `order` more explicit. Inspired by pd.NamedAgg.

    :param categories: list-like object to create new categorical column.
    :param order: string object that can be either "sort" or "appearance".
        If "sort", the `categories` argument will be sorted with np.sort;
        if "apperance", the `categories` argument will be used as is.
    :returns: A namedtuple of (`categories`, `order`).
    """

    categories: list = None
    order: str = None


def as_categorical_checks(df: pd.DataFrame, **kwargs) -> tuple:
    """
    This function raises errors if columns in `kwargs` are
    absent in the the dataframe's columns.
    It also raises errors if the tuple in `kwargs`
    has a length greater than 2, or the `order` value,
    if not None, is not one of `appearance` or `sort`.
    Error is raised if the `categories` in the tuple in `kwargs`
    is not a 1-D array-like object.

    This function is executed before proceeding to the computation phase.

    If all checks pass, the dataframe,
    and a pairing of column names and namedtuple
    of (categories, order) is returned.

    :param df: The pandas DataFrame object.
    :param kwargs: A pairing of column name
        to a tuple of (`categories`, `order`).
    :returns: A tuple (pandas DataFrame, dictionary).
    :raises TypeError: if ``kwargs`` is not a tuple.
    :raises ValueError: if ``categories`` is not a 1-D array.
    :raises ValueError: if ``order`` is not one of
        `sort`, `appearance`, or `None`.
    """

    # column checks
    check_column(df, kwargs)

    categories_dict = {}

    # type checks
    for column_name, value in kwargs.items():
        check("Pair of `categories` and `order`", value, [tuple])
        if len(value) != 2:
            raise ValueError("Must provide tuple of (categories, order).")
        value = asCategorical(*value)
        value_categories = value.categories
        if value_categories is not None:
            if not is_list_like(value_categories):
                raise TypeError(f"{value_categories} should be list-like.")
            value_categories = [*value_categories]
            arr_ndim = np.asarray(value_categories).ndim
            if any((arr_ndim < 1, arr_ndim > 1)):
                raise ValueError(
                    f"""
                    {value_categories} is not a 1-D array.
                    """
                )
        value_order = value.order
        if value_order is not None:
            check("order", value_order, [str])
            if value_order not in ("appearance", "sort"):
                raise ValueError(
                    """
                    `order` argument should be one of
                    "appearance", "sort" or `None`.
                    """
                )
        categories_dict[column_name] = asCategorical(
            categories=value_categories, order=value_order
        )

    return df, categories_dict


def _computations_as_categorical(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    This function handles cases where categorical columns are created with
    an order, or specific values supplied for the categories. It uses a kwarg,
    with a namedtuple - `column_name: (categories, order)`, with the idea
    inspired by Pandas' NamedAggregation. The defaults for the namedtuple are
    (None, None) and will return a categorical dtype with no order and
    categories inferred from the column.
    """

    df, categories_dict = as_categorical_checks(df, **kwargs)

    categories_dtypes = {}
    for column_name, ascategorical in categories_dict.items():
        categories = _encode_categories(
            ascategorical.categories, df, column_name
        )
        categories_dtypes[column_name] = _encode_order(
            ascategorical.order, categories
        )

    df = df.astype(categories_dtypes)

    return df


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
def _encode_categories(cat, df, column):
    """
    base function for processing `categories`
    in `_computations_as_categorical`.
    Returns a Series.
    """
    raise TypeError("This type is not supported in `categories`.")


@_encode_categories.register(type(None))  # noqa: F811
def _sub_categories(cat, df, column):  # noqa: F811
    """
    base function for processing `categories`
    in `_computations_as_categorical`.
    Apllies to only NoneType.
    Returns a Series.
    """
    column = df[column]
    if column.hasnans:
        column = column.dropna()
    return column


@_encode_categories.register(list)  # noqa: F811
def _sub_categories(cat, df, column):  # noqa: F811
    """
    base function for processing `categories`
    in `_computations_as_categorical`.
    Apllies to only list type.
    Returns a Series.
    """
    if pd.isna(cat).any():
        raise ValueError("""`categories` cannot have null values.""")
    col = df[column]
    check_presence = col.isin(cat)
    check_presence_sum = check_presence.sum()

    if check_presence_sum == 0:
        warnings.warn(
            f"""
            None of the values in `{column}` are in
            {cat};
            this might create nulls for all your values
            in the new categorical column.
            """,
            UserWarning,
            stacklevel=2,
        )
    elif check_presence_sum != check_presence.size:
        missing_values = col[~check_presence]
        if missing_values.hasnans:
            missing_values = missing_values.dropna()
        warnings.warn(
            f"""
            Values {tuple(missing_values)} are missing from
            categories {cat}
            for {column}; this may create nulls
            in the new categorical column.
            """,
            UserWarning,
            stacklevel=2,
        )
    return pd.Series(cat)


@functools.singledispatch
def _encode_order(order, categories):
    """
    base function for processing `order`
    in `_computations_as_categorical`.
    Returns a pd.CategoricalDtype().
    """
    raise TypeError("This type is not supported in `order`.")


@_encode_order.register(type(None))  # noqa: F811
def _sub_encode_order(order, categories):  # noqa: F811
    """
    base function for processing `order`
    in `_computations_as_categorical`.
    Apllies to only NoneType.
    Returns a pd.CategoricalDtype().
    """
    if not categories.is_unique:
        categories = categories.unique()

    return pd.CategoricalDtype(categories=categories, ordered=False)


@_encode_order.register(str)  # noqa: F811
def _sub_encode_order(order, categories):  # noqa: F811
    """
    base function for processing `order`
    in `_computations_as_categorical`.
    Apllies to only strings.
    Returns a pd.CategoricalDtype().
    """
    if (order == "sort") and (not categories.is_monotonic_increasing):
        categories = categories.sort_values()
    if not categories.is_unique:
        categories = categories.unique()

    return pd.CategoricalDtype(categories=categories, ordered=True)


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


@functools.singledispatch
def _process_text(result: str, df, column_name, new_column_names, merge_frame):
    """
    Base function for `process_text` when `result` is of ``str`` type.
    """
    if new_column_names:
        return df.assign(**{new_column_names: result})
    df[column_name] = result
    return df


@_process_text.register
def _sub_process_text(
    result: pd.Series, df, column_name, new_column_names, merge_frame
):
    """
    Base function for `process_text` when `result` is of ``pd.Series`` type.
    """
    if new_column_names:
        return df.assign(**{new_column_names: result})
    df[column_name] = result
    return df


@_process_text.register  # noqa: F811
def _sub_process_text(  # noqa: F811
    result: pd.DataFrame, df, column_name, new_column_names, merge_frame
):  # noqa: F811
    """
    Base function for `process_text` when `result` is of ``pd.DataFrame`` type.
    """
    result = _process_text_result_is_frame(new_column_names, result)
    if not merge_frame:
        return result
    return _process_text_result_MultiIndex(result.index, result, df)


@functools.singledispatch
def _process_text_result_is_frame(new_column_names: str, result):
    """
    Function to modify `result` columns from `process_text` if
    `result` is a dataframe. Applies only if `new_column_names`
    is a string type.
    """
    if new_column_names:
        return result.add_prefix(new_column_names)
    return result


@_process_text_result_is_frame.register
def _sub_process_text_result_is_frame(new_column_names: list, result):
    """
    Function to modify `result` columns from `process_text` if
    `result` is a dataframe. Applies only if `new_column_names`
    is a list type.
    """
    if len(new_column_names) != len(result.columns):
        raise ValueError(
            """
            The length of `new_column_names` does not
            match the number of columns in the new
            dataframe generated from the text processing.
            """
        )
    result.columns = new_column_names
    return result


@functools.singledispatch
def _process_text_result_MultiIndex(index: pd.Index, result, df):
    """
    Function to modify `result` columns from `process_text` if
    `result` is a dataframe and it has a single Index.
    """
    return pd.concat([df, result], axis="columns")


@_process_text_result_MultiIndex.register
def _sub_process_text_result_MultiIndex(index: pd.MultiIndex, result, df):
    """
    Function to modify `result` columns from `process_text` if
    `result` is a dataframe and it has a MultiIndex.
    At the moment, this function is primarily to cater for `str.extractall`,
    since at the moment,
    this is the only string method that returns a MultiIndex.
    The function may be modified,
    if another string function that returns a  MultIndex
    is added to Pandas string methods.

    For this function, `df` has been converted to a MultiIndex,
    with the extra index added to create unique indices.
    This comes in handy when merging back the dataframe,
    especially if `result` returns duplicate indices.
    """
    result = result.reset_index(level="match")
    df = df.join(result, how="outer")
    # droplevel gets rid of the extra index added at the start
    # (# extra_index_line)
    df = df.droplevel(-1).set_index("match", append=True)
    return df


class JOINOPERATOR(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


class JOINTYPES(Enum):
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
    sequence_of_operators = {op.value for op in JOINOPERATOR}
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
    how: str = "inner",
    sort_by_appearance: bool = False,
    suffixes=("_x", "_y"),
) -> tuple:
    """
    Preliminary checks are conducted here.
    This function checks for conditions such as
    MultiIndexed dataframe columns,
    improper `suffixes` configuration,
    as well as unnamed Series.

    A tuple of
    (`df`, `right`,
     `left_on`, `right_on`,
     `operator`, `sort_by_appearance`,
     `suffixes`)
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

    # each condition should be a tuple of length 3:
    for condition in conditions:
        check("condition", condition, [tuple])
        if len(condition) != 3:
            raise ValueError(
                f"""
                condition should have only three elements.
                Your condition however is of length {len(condition)}
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

    join_types = {jointype.value for jointype in JOINTYPES}
    if how not in join_types:
        raise ValueError(f"`how` should be one of {', '.join(join_types)}.")

    check("sort_by_appearance", sort_by_appearance, [bool])

    check("suffixes", suffixes, [tuple])

    if len(suffixes) != 2:
        raise ValueError("`suffixes` argument should be a 2-length tuple")

    if suffixes == (None, None):
        raise ValueError("At least one of the suffixes should be non-null.")

    for suffix in suffixes:
        check("suffix", suffix, [str, type(None)])

    return (df, right, conditions, how, sort_by_appearance, suffixes)


def _cond_join_suffixes(
    df: pd.DataFrame, right: pd.DataFrame, conditions: tuple, suffixes: tuple
):
    """
    If there are overlapping columns in `df` and `right`,
    modify the columns, using the suffix in suffixes.
    A tuple of (df, right, conditions) is returned.
    """

    common_columns = df.columns.intersection(right.columns, sort=False)

    left_on, right_on, operators = zip(*conditions)
    if not common_columns.empty:
        left_suffix, right_suffix = suffixes

        if left_suffix:
            mapping = {}
            for common in common_columns:
                new_label = f"{common}{left_suffix}"
                if new_label in df.columns:
                    raise ValueError(
                        f"""
                        {new_label} is present in `df` columns.
                        Kindly provide unique suffixes to create
                        columns that are not present in `df`.
                        """
                    )
                mapping[common] = new_label
            left_on = [
                f"{label}{left_suffix}" if label in common_columns else label
                for label in left_on
            ]
            df = df.rename(columns=mapping)
        if right_suffix:
            mapping = {}
            for common in common_columns:
                new_label = f"{common}{right_suffix}"
                if new_label in right.columns:
                    raise ValueError(
                        f"""
                        {new_label} is present in `right` columns.
                        Kindly provide unique suffixes to create
                        columns that are not present in `right`.
                        """
                    )

                mapping[common] = new_label
            right_on = [
                f"{label}{right_suffix}" if label in common_columns else label
                for label in right_on
            ]
            right = right.rename(columns=mapping)

    conditions = [*zip(left_on, right_on, operators)]

    return df, right, conditions


def _conditional_join_type_check(
    left_column: pd.Series, right_column: pd.Series, op: str
) -> None:
    """
    Raise error if column type is not any of
    numeric, datetime, or string.
    Strings are not supported on non-equi operators.
    """

    error_msg = """
          conditional_join only supports
          numeric, date, or string dtypes.
          The columns must also be of the same type.
          """
    error_msg_string = """
                       Strings can only be compared
                       on the equal(`==`) operator.
                       """
    if is_string_dtype(left_column):
        if not is_string_dtype(right_column):
            raise ValueError(error_msg)
        if op.name != JOINOPERATOR.STRICTLY_EQUAL.value:
            raise ValueError(error_msg_string)
        return None
    if is_numeric_dtype(left_column):
        if not is_numeric_dtype(right_column):
            raise ValueError(error_msg)
        return None
    if is_datetime64_dtype(left_column):
        if not is_datetime64_dtype(right_column):
            raise ValueError(error_msg)
        return None


def _le_create_ranges(indices: np.array, len_right: int) -> np.array:
    """
    Create ordered indices for each value in
    `right_keys` in `_less_than_indices`.
    Faster than a list comprehension, as
    the array size increases.

    code copied from Stack Overflow
    https://stackoverflow.com/a/47126435/7175713
    """
    cum_length = len_right - indices
    cum_length = cum_length.cumsum()
    # generate ones
    # note that cum_length[-1] is the total
    # number of inidices to be generated
    ids = np.ones(cum_length[-1], dtype=int)
    ids[0] = indices[0]
    # at each specific point in id, replace the value
    # so, we should have say 0, 1, 1, 1, 1, -5, 1, 1, 1, -3, ...
    # when a cumsum is implemented in the next line,
    # we get, 0, 1, 2, 3, 4, 0, 1,2, 3, 0, ...
    # our ranges is obtained, with more efficiency
    # for larger arrays
    ids[cum_length[:-1]] = indices[1:] - len_right + 1
    # the cumsum here gives us the same output as
    # [np.range(start, len_right) for start in search_indices]
    # but much faster
    return ids.cumsum()


def _ge_create_ranges(indices: np.array) -> np.array:
    """
    Create ordered indices for each value in
    `right_keys` in `_greater_than_indices`.
    Faster than a list comprehension, as
    the array size increases.

    code copied from Stack Overflow
    https://stackoverflow.com/a/47126435/7175713
    """
    cum_length = indices.cumsum()
    ids = np.ones(cum_length[-1], dtype=int)
    ids[0] = 0
    ids[cum_length[:-1]] = -1 * indices[:-1] + 1
    return ids.cumsum()


def _equal_indices(left_c: pd.Series, right_c: pd.Series, len_conditions: int):
    """
    Use pandas' join method to get the index labels.
    If len_condition is > 1, then get_indexer is used
    to get relevant index labels for left_c.

    Returns a tuple of (left_c, right_c)
    if len_conditions is == 1, else left_c.
    """

    if len_conditions > 1:
        if not right_c.is_unique:
            right_c = right_c.factorize()[-1]
        result = pd.Index(left_c).get_indexer_for(right_c)
        exclude_rows = result == -1
        if exclude_rows.all():
            return None
        if exclude_rows.any():
            result = result[~exclude_rows]
        return left_c.index.take(result)

    left_c.index.name = "l"
    right_c.index.name = "r"
    left_c.name = "merge"
    right_c.name = "merge"
    left_c = left_c.reset_index()
    right_c = right_c.reset_index()
    result = left_c.merge(right_c, how="inner", sort=False, on="merge")
    if result.empty:
        return None
    return pd.Index(result["l"]), pd.Index(result["r"])


def _not_equal_indices(
    left_c: pd.Series,
    right_c: pd.Series,
    len_conditions: int,
    strict: bool = True,
):
    """
    Use binary search to get indices where
    `left_c` is exactly  not equal to `right_c`.
    It is a combination of strictly less than
    and strictly greater than indices.

    Returns a tuple of (left_c, right_c)
    if len_conditions is == 1, else left_c.
    """

    # get nulls, since they are not equal to anything
    # NaNs are not equal to NaNs
    l_nulls = pd.Index([], dtype=int)
    r_nulls = pd.Index([], dtype=int)
    nulls_l = pd.Index([], dtype=int)
    nulls_r = pd.Index([], dtype=int)
    left_hasnans = left_c.hasnans
    right_hasnans = right_c.hasnans
    if len_conditions == 1:
        if left_hasnans:
            left_c_isna = left_c.isna()
            nulls_count = left_c_isna.sum()
            l_nulls = left_c.index[left_c_isna]
            # each value in right_c MUST be matched to all the null groups
            l_nulls = pd.Int64Index(np.tile(l_nulls, right_c.size))
            if nulls_count > 1:
                nulls_r = right_c.index.repeat(nulls_count)
            else:
                nulls_r = right_c.index

        if right_hasnans:
            if left_hasnans:  # avoids duplication of NaN matching NaN
                left_c = left_c.dropna()
            right_c_isna = right_c.isna()
            nulls_count = right_c_isna.sum()
            r_nulls = right_c.index[right_c_isna]
            # each value in left_c must be matched to all the null groups
            r_nulls = pd.Int64Index(np.tile(r_nulls, left_c.size))
            if nulls_count > 1:
                nulls_l = left_c.index.repeat(nulls_count)
            else:
                nulls_l = left_c.index

        l_nulls = l_nulls.append(nulls_l)
        r_nulls = nulls_r.append(r_nulls)
    else:
        if right_c.hasnans:
            # every row in left_c
            # will not be equal to NaN
            return left_c.index
        if left_c.hasnans:
            l_nulls = left_c.index[left_c.isna()]

    # sort and drop nulls here
    # to avoid sorting twice, in less_than_indices
    # and greater_than _indices functions
    if not right_c.is_monotonic_increasing:
        right_c = right_c.sort_values()
    if right_hasnans:
        right_c = right_c.dropna()

    # get less than index labels
    result = _less_than_indices(left_c, right_c, len_conditions, strict, True)
    if len_conditions == 1:
        if result is None:
            lt_left = pd.Index([], dtype=int)
            lt_right = pd.Index([], dtype=int)
        else:
            lt_left, lt_right = result
    else:
        if result is None:
            lt_left = pd.Index([], dtype=int)
        else:
            lt_left = result

    # greater than index labels
    result = _greater_than_indices(
        left_c, right_c, len_conditions, strict, True
    )
    if len_conditions == 1:
        if result is None:
            gt_left = pd.Index([], dtype=int)
            gt_right = pd.Index([], dtype=int)
        else:
            gt_left, gt_right = result
    else:
        if result is None:
            gt_left = pd.Index([], dtype=int)
        else:
            gt_left = result
    if len_conditions > 1:
        left_c = lt_left.append([gt_left, l_nulls]).unique()
        return left_c
    else:
        left_c = lt_left.append([gt_left, l_nulls])
        right_c = lt_right.append([gt_right, r_nulls])

    return left_c, right_c


def _less_than_indices(
    left_c: pd.Series,
    right_c: pd.Series,
    len_conditions: int,
    strict: bool,
    not_equal: bool = False,
):
    """
    Use binary search to get indices where left_c is less than
    or equal to right_c. If strict is True,
    then only indices where `left_c`
    is less than (but not equal to) `right_c` are returned.

    Returns a tuple of (left_c, right_c)
    if len_conditions is == 1, else left_c.
    """

    # no point going through all the hassle
    if left_c.min() > right_c.max():
        return None

    if not_equal is False:
        if right_c.hasnans:
            right_c = right_c.dropna()
        if not right_c.is_monotonic_increasing:
            right_c = right_c.sort_values()

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
        search_indices = search_indices[~rows_equal]
    if search_indices.size == 0:
        return None

    if len_conditions > 1:
        return left_c.index
    # for each index in `search_indices`,
    # generate all indices for `right_keys`,
    # where the values in `right_keys` are greater than
    # or equal to `left_c`
    positions = _le_create_ranges(search_indices, len_right)
    search_indices = len_right - search_indices
    if strict:
        right_c = right_c.take(positions)
        left_c = left_c.repeat(search_indices)
        rows_equal = left_c.array == right_c.array
        if rows_equal.all():
            return None
        if rows_equal.any():
            left_c = left_c.index[~rows_equal]
            right_c = right_c.index[~rows_equal]
            return left_c, right_c
        return left_c.index, right_c.index

    right_c = right_c.index.take(positions)
    left_c = left_c.index.repeat(search_indices)

    return left_c, right_c


def _greater_than_indices(
    left_c: pd.Series,
    right_c: pd.Series,
    len_conditions: int,
    strict: bool,
    not_equal: bool = False,
):
    """
    Use binary search to get indices where left_c is greater than
    or equal to right_c. If strict is True,
    then only indices where `left_c`
    is greater than (but not equal to) `right_c` are returned.

    Returns a tuple of (left_c, right_c)
    if len_conditions is == 1, else left_c.
    """

    # quick break, avoiding the hassle
    if left_c.max() < right_c.min():
        return None

    if not_equal is False:
        if right_c.hasnans:
            right_c = right_c.dropna()
        if not right_c.is_monotonic_increasing:
            right_c = right_c.sort_values()
    if left_c.hasnans:
        left_c = left_c.dropna()

    search_indices = right_c.searchsorted(left_c, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0
    # that means the respective position in `left_c`
    # has no values from `right_c` that are greater than
    # or equal, and should therefore be discarded
    rows_equal = search_indices == 0
    if rows_equal.any():
        left_c = left_c[~rows_equal]
        search_indices = search_indices[~rows_equal]
    if search_indices.size == 0:
        return None

    if len_conditions > 1:
        return left_c.index
    # for each index in `search_indices`,
    # generate all indices for `right_keys`,
    # where the values in `right_keys` are less than
    # or equal to `left_c`
    positions = _ge_create_ranges(search_indices)
    if strict:
        right_c = right_c.take(positions)
        left_c = left_c.repeat(search_indices)
        rows_equal = left_c.array == right_c.array
        if rows_equal.all():
            return None
        if rows_equal.any():
            left_c = left_c.index[~rows_equal]
            right_c = right_c.index[~rows_equal]
            return left_c, right_c
        return left_c.index, right_c.index

    right_c = right_c.index.take(positions)
    left_c = left_c.index.repeat(search_indices)

    return left_c, right_c


def _interval_indices(
    left_c: pd.Series,
    lower_boundary: pd.Series,
    upper_boundary: pd.Series,
    left_operator: str,
    right_operator: str,
):
    """
    Use Intervals to get indices where left_c is between
    lower_boundary and upper_boundary.

    Returns a tuple of (left_c, right_c),
    which are the indices of the left and right dataframes
    respectively (the dataframes in the conditional join).
    """

    # quick break, avoiding the hassle
    if left_c.max() < lower_boundary.min():
        return None
    actual_intervals = lower_boundary > upper_boundary

    if actual_intervals.all():
        return None

    if actual_intervals.any():
        lower_boundary = lower_boundary[~actual_intervals]
        upper_boundary = upper_boundary[~actual_intervals]

    closed_mapping = {
        (
            JOINOPERATOR.GREATER_THAN.value,
            JOINOPERATOR.LESS_THAN.value,
        ): "neither",
        (
            JOINOPERATOR.GREATER_THAN_OR_EQUAL.value,
            JOINOPERATOR.LESS_THAN.value,
        ): "right",
        (
            JOINOPERATOR.GREATER_THAN.value,
            JOINOPERATOR.LESS_THAN_OR_EQUAL.value,
        ): "left",
        (
            JOINOPERATOR.GREATER_THAN_OR_EQUAL.value,
            JOINOPERATOR.LESS_THAN_OR_EQUAL.value,
        ): "both",
    }

    closed = closed_mapping[(left_operator, right_operator)]

    interval = pd.IntervalIndex.from_arrays(lower_boundary, upper_boundary, closed = closed)

    positions = interval.get_indexer_for(left_c)

    return positions

    exclude_rows = positions == -1

    if exclude_rows.all():
        return None

    if exclude_rows.any():
        positions = positions[~exclude_rows]

    # return [interval.contains(val) for val in left_c]

    left_repeat = np.row_stack([interval.contains(val) for val in left_c]).sum(1)
    #return left_repeat
    left_c = left_c.index.repeat(left_repeat)
    return left_c, positions


def _create_conditional_join_empty_frame(
    df: pd.DataFrame, right: pd.DataFrame, how: str
):
    """
    Create final dataframe for conditional join,
    if there are no matches.
    """

    if how == JOINTYPES.INNER.value:
        df = df.dtypes.to_dict()
        right = right.dtypes.to_dict()
        df = {**df, **right}
        df = {key: pd.Series([], dtype=value) for key, value in df.items()}
        return pd.DataFrame(df)

    if how == JOINTYPES.LEFT.value:
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

    if how == JOINTYPES.RIGHT.value:
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
        left_index = left_index.take(sorter)

    if how == JOINTYPES.INNER.value:
        df = df.reindex(left_index)
        right = right.reindex(right_index)
        df.index = np.arange(left_index.size)
        right.index = df.index
        return pd.concat([df, right], axis="columns", join=how, sort=False)

    if how == JOINTYPES.LEFT.value:
        right = right.reindex(right_index)
        right.index = left_index
        return df.join(right, how=how, sort=False).reset_index(drop=True)

    if how == JOINTYPES.RIGHT.value:
        df = df.reindex(left_index)
        df.index = right_index
        return df.join(right, how=how, sort=False).reset_index(drop=True)


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
        JOINOPERATOR.GREATER_THAN.value,
        JOINOPERATOR.LESS_THAN.value,
        JOINOPERATOR.NOT_EQUAL.value,
    }:
        strict = True

    if op in {
        JOINOPERATOR.LESS_THAN.value,
        JOINOPERATOR.LESS_THAN_OR_EQUAL.value,
    }:
        return _less_than_indices(left_c, right_c, len_conditions, strict)
    elif op in {
        JOINOPERATOR.GREATER_THAN.value,
        JOINOPERATOR.GREATER_THAN_OR_EQUAL.value,
    }:
        return _greater_than_indices(left_c, right_c, len_conditions, strict)
    elif op == JOINOPERATOR.STRICTLY_EQUAL.value:
        return _equal_indices(left_c, right_c, len_conditions)
    elif op == JOINOPERATOR.NOT_EQUAL.value:
        return _not_equal_indices(left_c, right_c, len_conditions, strict)


def _conditional_join_compute(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    how: str,
    sort_by_appearance: bool,
) -> pd.DataFrame:
    """
    This is where the actual computation for the conditional join takes place.
    If there are no matches, None is returned; if however, there is a match,
    then a pandas DataFrame is returned.
    """

    len_conditions = len(conditions)

    if len_conditions == 1:
        left_on, right_on, op = conditions[0]

        left_c = df[left_on]
        right_c = right[right_on]

        _conditional_join_type_check(left_c, right_c, op)

        result = _generic_func_cond_join(left_c, right_c, op, len_conditions)

        if result is None:
            return _create_conditional_join_empty_frame(df, right, how)

        left_c, right_c = result

        return _create_conditional_join_frame(
            df, right, left_c, right_c, how, sort_by_appearance
        )

    df_index = df.index
    # iteratively reduce the number of rows
    # from df, until we have the certain index labels
    # that will be in the final dataframe
    # usually much smaller, which should help
    # reduce overall processing time
    for condition in conditions:
        left_on, right_on, op = condition
        left_c = df.loc[df_index, left_on]
        right_c = right[right_on]

        _conditional_join_type_check(left_c, right_c, op)

    less_than_operators = (
        JOINOPERATOR.LESS_THAN_OR_EQUAL.value,
        JOINOPERATOR.LESS_THAN.value,
    )
    greater_than_operators = (
        JOINOPERATOR.GREATER_THAN_OR_EQUAL.value,
        JOINOPERATOR.GREATER_THAN.value,
    )

    # check for possible interval join combinations
    # converts [('a', 'b', '>'), ('a', 'c', '<)] to
    # [('a', ('b', 'c'), ('>', '<'))]
    # this simulates the between operation -> 'b < a < c'
    interval_joins = []
    paired_conditions = pairwise(conditions)
    for left_tuple, right_tuple in paired_conditions:
        left_l, _, left_op = left_tuple
        right_l, _, right_op = right_tuple
        if (
            (left_l == right_l)
            & (left_op in greater_than_operators)
            & (right_op in less_than_operators)
        ):
            interval_joins.append((left_tuple, right_tuple))

    multiple_conditions = []
    paired_conditions = []
    if interval_joins:
        combo = chain.from_iterable(interval_joins)
        combo = [*combo]
        for condition in conditions:
            if condition not in combo:
                multiple_conditions.append(condition)
        for left_tuple, right_tuple in interval_joins:
            left_l, left_r, left_op = left_tuple
            _, right_r, right_op = right_tuple
            zipped = (left_l, left_r, right_r, left_op, right_op)
            paired_conditions.append(zipped)
        multiple_conditions.extend(paired_conditions)
    else:
        multiple_conditions = conditions
    return multiple_conditions


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
