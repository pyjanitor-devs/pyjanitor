"""Miscellaneous internal PyJanitor helper functions."""

import fnmatch
import functools
import os
import re
import socket
import sys
import warnings
from collections.abc import Callable as dispatch_callable
from itertools import chain
from typing import (
    Callable,
    Dict,
    Iterable,
    Union,
)

import numpy as np
import pandas as pd


def check(varname: str, value, expected_types: list):
    """
    One-liner syntactic sugar for checking types.
    It can also check callables.

    Example usage:

    ```python
    check('x', x, [int, float])
    ```

    :param varname: The name of the variable (for diagnostic error message).
    :param value: The value of the `varname`.
    :param expected_types: The type(s) the item is expected to be.
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


@functools.singledispatch
def _select_column_names(columns_to_select, df):
    """
    base function for column selection.
    Returns a list of column names.
    """
    raise TypeError("This type is not supported in column selection.")


@_select_column_names.register(type(re.compile(r"\d+")))  # noqa: F811
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


@_select_column_names.register(tuple)  # noqa: F811
def _column_sel_dispatch(columns_to_select, df):  # noqa: F811
    """
    Base function for column selection.
    This caters to columns that are of tuple type.
    The tuple is returned as is, if it exists in the columns.
    """
    if columns_to_select not in df.columns:
        raise KeyError(f"No match was returned for {columns_to_select}")
    return columns_to_select


@_select_column_names.register(list)  # noqa: F811
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
        _select_column_names(entry, df) for entry in columns_to_select
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


@_select_column_names.register(str)  # noqa: F811
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


@_select_column_names.register(slice)  # noqa: F811
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


@_select_column_names.register(dispatch_callable)  # noqa: F811
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

    :param submodule: `pyjanitor` submodule that needs an external dependency.
    :param package: External package this submodule relies on.
    :param conda_channel: `conda` channel package can be installed from,
        if at all.
    :param pip_install: Whether package can be installed via `pip`.
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
    Raises an error if a function operating on a DataFrame is not idempotent.
    That is, `func(func(df)) = func(df)` is not `True` for all `df`.

    :param func: A Python method.
    :param df: A pandas `DataFrame`.
    :param args: Positional arguments supplied to the method.
    :param kwargs: Keyword arguments supplied to the method.
    :raises ValueError: If `func` is found to not be idempotent for the given
        DataFrame (`df`).
    """
    if not func(df, *args, **kwargs) == func(
        func(df, *args, **kwargs), *args, **kwargs
    ):
        raise ValueError(
            "Supplied function is not idempotent for the given DataFrame."
        )


def deprecated_alias(**aliases) -> Callable:
    """
    Used as a decorator when deprecating old function argument names, while
    keeping backwards compatibility. Implementation is inspired from [`StackOverflow`][stack_link].

    [stack_link]: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

    Functional usage example:

    ```python
    @deprecated_alias(a='alpha', b='beta')
    def simple_sum(alpha, beta):
        return alpha + beta
    ```

    :param aliases: Dictionary of aliases for a function's arguments.
    :return: Your original function wrapped with the `kwarg` redirection
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
    """
    Used as a decorator when refactoring functions.

    Implementation is inspired from [`Hacker Noon`][hacker_link].

    [hacker_link]: https://hackernoon.com/why-refactoring-how-to-restructure-python-package-51b89aa91987

    Functional usage example:

    ```python
    @refactored_function(
        message="simple_sum() has been refactored. Use hard_sum() instead."
    )
    def simple_sum(alpha, beta):
        return alpha + beta
    ```

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
    `TypeError` if both arguments are provided, and warns if old alias
    is used. Nothing is returned as the passed `kwargs` are modified
    directly. Implementation is inspired from [`StackOverflow`][stack_link].

    [stack_link]: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

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
    One-liner syntactic sugar for checking the presence or absence
    of columns.

    Example usage:

    ```python
    check(df, ['a', 'b'], present=True)
    ```

    This will check whether columns `'a'` and `'b'` are present in
    `df`'s columns.

    One can also guarantee that `'a'` and `'b'` are not present
    by switching to `present=False`.

    :param df: The name of the variable.
    :param column_names: A list of column names we want to check to see if
        present (or absent) in `df`.
    :param present: If `True` (default), checks to see if all of `column_names`
        are in `df.columns`. If `False`, checks that none of `column_names` are
        in `df.columns`.
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
    Decorator for escaping `np.nan` and `None` in a function.

    Example usage:

    ```python
    df[column].apply(skipna(transform))

    # Can also be used as shown below
    @skipna
    def transform(x):
        pass
    ```

    :param f: the function to be wrapped.
    :returns: the wrapped function.
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

    Example usage:

    ```python
    df[column].apply(
        skiperror(transform, return_val=3, return_x=False))

    # Can also be used as shown below
    @skiperror(return_val=3, return_x=False)
    def transform(x):
        pass
    ```
    :param f: the function to be wrapped.
    :param return_x: whether or not the original value that caused error
        should be returned.
    :param return_val: the value to be returned when an error hits.
        Ignored if `return_x` is `True`.
    :returns: the wrapped function.
    """

    def _wrapped(x, *args, **kwargs):
        try:
            return f(x, *args, **kwargs)
        except Exception:  # skipcq: PYL-W0703
            if return_x:
                return x
            return return_val

    return _wrapped


def is_connected(url: str) -> bool:
    """
    This is a helper function to check if the client
    is connected to the internet.

    Example:
        print(is_connected("www.google.com"))
        console >> True

    :param url: We take a test url to check if we are
        able to create a valid connection.
    :raises OSError: if connection to `URL` cannot be
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

        warnings.warn(
            "There was an issue connecting to the internet. "
            "Please see original error below."
        )
        raise e
    return False


# hack to get it to recognize typing.Pattern
# functools.singledispatch does not natively
# recognize types from the typing module
# `type(re.compile(r"\d+"))` returns re.Pattern
# which is a type and functools.singledispatch
# accepts it without drama;
# however, the same type from typing.Pattern
# is not accepted.
