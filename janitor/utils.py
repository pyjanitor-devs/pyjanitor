"""Miscellaneous utility functions."""

from __future__ import annotations

import importlib
import os
import socket
import sys
from functools import singledispatch, wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Union
from warnings import warn

import numpy as np
import pandas as pd


def check(varname: str, value, expected_types: list):
    """One-liner syntactic sugar for checking types.

    It can also check callables.

    Examples:
        ```python
        check('x', x, [int, float])
        ```

    Args:
        varname: The name of the variable (for diagnostic error message).
        value: The value of the `varname`.
        expected_types: The type(s) the item is expected to be.

    Raises:
        TypeError: If data is not the expected type.
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
        raise TypeError(f"{varname} should be one of {expected_types}.")


@singledispatch
def _expand_grid(value, key, uniqs, grid_index, contents) -> tuple:
    """
    Base function for dispatch of `_expand_grid`.

    Args:
        value: list-like object.
        key: label or tuple of labels.
        uniqs: unique labels.
            Ensures label in `key` does not already exists.
        grid_index: numpy array used to reindex `value`.
        contents: dictionary created after reindexing `value` with `grid_index`.

    Raises:
        ValueError: If `key` is duplicated.
        TypeError: If ndim==2 and tuple is not provided.

    Returns:
        A dictionary
    """
    if value.ndim == 1:
        if key in uniqs:
            raise ValueError(
                f"{key} is duplicated. Ensure the labels in the dictionary are unique."
            )
        uniqs.add(key)
        new_values = value[grid_index]
        contents[key] = new_values
        return contents, uniqs
    if not isinstance(key, tuple):
        raise TypeError(
            f"Expected a tuple of labels as key; instead got {type(key).__name__}"
        )
    if len(key) != value.shape[-1]:
        raise ValueError(
            f"The number of labels in {key} -> {len(key)} "
            f"is not equal to the number of columns -> {value.shape[-1]} "
            "in the array."
        )
    for label in key:
        if label in uniqs:
            raise ValueError(
                f"{label} in {key} is duplicated. "
                "Ensure the labels in the dictionary are unique."
            )
        uniqs.add(label)
    new_values = value[grid_index]
    new_values = {label: new_values[:, num] for num, label in enumerate(key)}
    contents.update(new_values)
    return contents, uniqs


@_expand_grid.register(pd.DataFrame)
def _sub_expand_grid(
    value, key, uniqs, grid_index, contents
) -> tuple:  # noqa: F811
    """
    Expands the DataFrame based on `grid_index`.
    """
    if not isinstance(key, tuple):
        raise TypeError(
            f"Expected a tuple of labels as key; instead got {type(key).__name__}"
        )
    if len(key) != value.columns.size:
        raise ValueError(
            f"The number of labels in {key} - {len(key)} "
            f"is not equal to the number of columns - {value.columns.size} "
            "in the DataFrame."
        )
    for label in key:
        if label in uniqs:
            raise ValueError(
                f"{label} in {key} is duplicated. "
                "Ensure the labels in the dictionary are unique."
            )
        uniqs.add(label)
    new_values = [arr._values for _, arr in value.items()]
    new_values = {
        label: new_values[num][grid_index] for num, label in enumerate(key)
    }
    contents.update(new_values)
    return contents, uniqs


@_expand_grid.register(pd.MultiIndex)
def _sub_expand_grid(
    value, key, uniqs, grid_index, contents
) -> tuple:  # noqa: F811
    """
    Expands the MultiIndex based on `grid_index`.
    """

    if not isinstance(key, tuple):
        raise TypeError(
            f"Expected a tuple of labels as key; instead got {type(key).__name__}"
        )
    if len(key) != value.nlevels:
        raise ValueError(
            f"The number of labels in {key} - {len(key)} "
            f"is not equal to the number of levels -{value.nlevels} "
            "in the MultiIndex."
        )
    for label in key:
        if label in uniqs:
            raise ValueError(
                f"{label} in {key} is duplicated. "
                "Ensure the labels are unique."
            )
        uniqs.add(label)
    new_values = [
        value.get_level_values(num)._values for num in range(value.nlevels)
    ]
    new_values = {
        label: new_values[num][grid_index] for num, label in enumerate(key)
    }
    contents.update(new_values)
    return contents, uniqs


@_expand_grid.register(pd.Series)
@_expand_grid.register(pd.Index)
def _sub_expand_grid(
    value, key, uniqs, grid_index, contents
) -> tuple:  # noqa: F811
    """
    Expands the Index/Series based on `grid_index`.
    """
    if key in uniqs:
        raise ValueError(
            f"{key} is duplicated. Ensure the labels in the dictionary are unique."
        )
    uniqs.add(key)
    new_values = value._values[grid_index]
    contents[key] = new_values
    return contents, uniqs


def import_message(
    submodule: str,
    package: str,
    conda_channel: str = None,
    pip_install: bool = False,
):
    """Return warning if package is not found.

    Generic message for indicating to the user when a function relies on an
    optional module / package that is not currently installed. Includes
    installation instructions. Used in `chemistry.py` and `biology.py`.

    Args:
        submodule: `pyjanitor` submodule that needs an external dependency.
        package: External package this submodule relies on.
        conda_channel: `conda` channel package can be installed from,
            if at all.
        pip_install: Whether package can be installed via `pip`.
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


def idempotent(func: Callable, df: pd.DataFrame, *args: Any, **kwargs: Any):
    """Raises an error if a function operating on a DataFrame is not
    idempotent.

    That is, `func(func(df)) = func(df)` is not `True` for all `df`.

    Args:
        func: A Python method.
        df: A pandas `DataFrame`.
        *args: Positional arguments supplied to the method.
        **kwargs: Keyword arguments supplied to the method.

    Raises:
        ValueError: If `func` is found to not be idempotent for the given
            DataFrame (`df`).
    """
    if not func(df, *args, **kwargs) == func(
        func(df, *args, **kwargs), *args, **kwargs
    ):
        raise ValueError(
            "Supplied function is not idempotent for the given DataFrame."
        )


def deprecated_kwargs(
    *arguments: tuple[str],
    message: str = (
        "The keyword argument '{argument}' of '{func_name}' is deprecated."
    ),
    error: bool = True,
) -> Callable:
    """Used as a decorator when deprecating function's keyword arguments.

    Examples:

        ```python
        from janitor.utils import deprecated_kwargs

        @deprecated_kwargs('x', 'y')
        def plus(a, b, x=0, y=0):
            return a + b
        ```

    Args:
        *arguments: The list of deprecated keyword arguments.
        message: The message of `ValueError` or `DeprecationWarning`.
            It should be a string or a string template. If a string template
            defaults input `func_name` and `argument`.
        error: If True, raises `ValueError` else returns `DeprecationWarning`.

    Raises:
        ValueError: If one of `arguments` is in the decorated function's
            keyword arguments.

    Returns:
        The original function wrapped with the deprecated `kwargs`
            checking function.

    <!--
    # noqa: DAR402
    -->
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for argument in arguments:
                if argument in kwargs:
                    msg = message.format(
                        func_name=func.__name__,
                        argument=argument,
                    )
                    if error:
                        raise ValueError(msg)
                    else:
                        warn(msg, DeprecationWarning)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_alias(**aliases) -> Callable:
    """
    Used as a decorator when deprecating old function argument names, while
    keeping backwards compatibility. Implementation is inspired from [`StackOverflow`][stack_link].

    [stack_link]: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

    Examples:
        ```python
        @deprecated_alias(a='alpha', b='beta')
        def simple_sum(alpha, beta):
            return alpha + beta
        ```

    Args:
        **aliases: Dictionary of aliases for a function's arguments.

    Returns:
        Your original function wrapped with the `kwarg` redirection
            function.
    """  # noqa: E501

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def refactored_function(message: str, category=FutureWarning) -> Callable:
    """Used as a decorator when refactoring functions.

    Implementation is inspired from [`Hacker Noon`][hacker_link].

    [hacker_link]: https://hackernoon.com/why-refactoring-how-to-restructure-python-package-51b89aa91987

    Examples:
        ```python
        @refactored_function(
            message="simple_sum() has been refactored. Use hard_sum() instead."
        )
        def simple_sum(alpha, beta):
            return alpha + beta
        ```

    Args:
        message: Message to use in warning user about refactoring.
        category: Type of Warning. Default is `FutureWarning`.

    Returns:
        Your original function wrapped with the kwarg redirection function.
    """  # noqa: E501

    def decorator(func):
        @wraps(func)
        def emit_warning(*args, **kwargs):
            warn(message, category, stacklevel=find_stack_level())
            return func(*args, **kwargs)

        return emit_warning

    return decorator


def rename_kwargs(func_name: str, kwargs: dict, aliases: dict):
    """Used to update deprecated argument names with new names.

    Throws a
    `TypeError` if both arguments are provided, and warns if old alias
    is used. Nothing is returned as the passed `kwargs` are modified
    directly. Implementation is inspired from [`StackOverflow`][stack_link].

    [stack_link]: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias


    Args:
        func_name: name of decorated function.
        kwargs: Arguments supplied to the method.
        aliases: Dictionary of aliases for a function's arguments.

    Raises:
        TypeError: If both arguments are provided.
    """  # noqa: E501
    for old_alias, new_alias in aliases.items():
        if old_alias in kwargs:
            if new_alias in kwargs:
                raise TypeError(
                    f"{func_name} received both {old_alias} and {new_alias}"
                )
            warn(
                f"{old_alias} is deprecated; use {new_alias}",
                DeprecationWarning,
            )
            kwargs[new_alias] = kwargs.pop(old_alias)


def check_column(
    df: pd.DataFrame, column_names: Union[Iterable, str], present: bool = True
):
    """One-liner syntactic sugar for checking the presence or absence
    of columns.

    Examples:
        ```python
        check(df, ['a', 'b'], present=True)
        ```

    This will check whether columns `'a'` and `'b'` are present in
    `df`'s columns.

    One can also guarantee that `'a'` and `'b'` are not present
    by switching to `present=False`.

    Args:
        df: The name of the variable.
        column_names: A list of column names we want to check to see if
            present (or absent) in `df`.
        present: If `True` (default), checks to see if all of `column_names`
            are in `df.columns`. If `False`, checks that none of `column_names`
            are in `df.columns`.

    Raises:
        ValueError: If data is not the expected type.
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
    """Decorator for escaping `np.nan` and `None` in a function.

    Examples:
        ```python
        df[column].apply(skipna(transform))
        ```

        Can also be used as shown below

        ```python
        @skipna
        def transform(x):
            pass
        ```

    Args:
        f: The function to be wrapped.

    Returns:
        The wrapped function.
    """

    def _wrapped(x, *args, **kwargs):
        if (isinstance(x, float) and np.isnan(x)) or x is None:
            return np.nan
        return f(x, *args, **kwargs)

    return _wrapped


def skiperror(
    f: Callable, return_x: bool = False, return_val=np.nan
) -> Callable:
    """Decorator for escaping any error in a function.

    Examples:
        ```python
        df[column].apply(
            skiperror(transform, return_val=3, return_x=False))
        ```

        Can also be used as shown below

        ```python
        @skiperror(return_val=3, return_x=False)
        def transform(x):
            pass
        ```

    Args:
        f: The function to be wrapped.
        return_x: Whether or not the original value that caused error
            should be returned.
        return_val: The value to be returned when an error hits.
            Ignored if `return_x` is `True`.

    Returns:
        The wrapped function.
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
    """This is a helper function to check if the client
    is connected to the internet.

    Examples:

        >>> print(is_connected("www.google.com"))
        True

    Args:
        url: We take a test url to check if we are
            able to create a valid connection.

    Raises:
        OSError: If connection to `URL` cannot be established

    Returns:
        We return a boolean that signifies our connection to the internet
    """
    try:
        sock = socket.create_connection((url, 80))
        if sock is not None:
            sock.close()
            return True
    except OSError as e:
        warn(
            "There was an issue connecting to the internet. "
            "Please see original error below."
        )
        raise e
    return False


def find_stack_level() -> int:
    """Find the first place in the stack that is not inside janitor
    (tests notwithstanding).

    Adapted from Pandas repo.

    Returns:
        Stack level number
    """

    import inspect

    import janitor as jn

    pkg_dir = os.path.abspath(os.path.dirname(jn.__file__))
    test_dir = os.path.join(os.path.dirname(pkg_dir), "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir) and not fname.startswith(test_dir):
            frame = frame.f_back
            n += 1
        else:
            break
    return n


def dynamic_import(file_path: Path):
    """Dynamically import all modules in a directory.

    :param file_path: The path to the file
        containing the modules to import.
    """
    # Iterate through all files in the current directory
    for filename in file_path.glob("*.py"):
        # Check if the file is a Python file and it's not the current __init__.py
        if filename != "__init__.py":
            # Get the module name (without the .py extension)
            module_name = filename.name
            # Dynamically import the module
            module = importlib.import_module(
                f".{module_name}", package=__name__
            )
            # Import all symbols from the module into the current namespace
            for name in dir(module):
                if not name.startswith("_"):  # avoid importing private symbols
                    globals()[name] = getattr(module, name)
