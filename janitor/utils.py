"""Miscellaneous internal PyJanitor helper functions."""

import functools
import warnings
from typing import Callable

import pandas as pd


def import_message(submodule, package, installation):
    """Raise import missing message."""
    print(
        f"To use the janitor submodule {submodule}, you need to install \
        {package}."
    )
    print()
    print(f"To do so, use the following command:")
    print()
    print(f"    {installation}")


def idempotent(func: Callable, df: pd.DataFrame, *args, **kwargs):
    """
    Check if a function is idempotent, i.e., f(f(x))=f(x) is true for all x.

    :param func: a python method
    :param df: a pandas dataframe
    :param args: Arguments supplied to the method
    :param kwargs: Arguments supplied to the method
    :return:
    """
    assert func(df, *args, **kwargs) == func(
        func(df, *args, **kwargs), *args, **kwargs
    )


def deprecated_alias(**aliases):
    """
    Raise a warning when deprecating old function argument names.

    Used as a decorator when deprecating old function argument names, while
    keeping backwards compatibility.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias  # noqa: E501

    Functional usage example:

    .. code-block:: python

        @deprecated_alias(a='alpha', b='beta')
        def simple_sum(alpha, beta):
            return alpha + beta

    :param aliases: dictionary of aliases for a function's arguments
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def rename_kwargs(func_name, kwargs, aliases):
    """
    Rename a keyeword argument with new names.

    Used to update deprecated argument names with new names. Throws a TypeError
    if both arguments are provided, and warns if old alias is used.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias  # noqa: E501

    :param func_name: name of decorated function
    :param aliases: dictionary of aliases for a function's arguments
    :param kwargs: arguments supplied to the method
    :return:
    """
    for old_alias, new_alias in aliases.items():
        if old_alias in kwargs:
            if new_alias in kwargs:
                raise TypeError(
                    f"{func_name} received both {old_alias} and {new_alias}"
                )
            warnings.warn(
                "{old_alias} is deprecated; use {new_alias}",
                DeprecationWarning,
            )
            kwargs[new_alias] = kwargs.pop(old_alias)
