"""polars variant of pandas_flavor"""

from __future__ import annotations

from functools import wraps
from typing import Callable

from janitor.utils import import_message

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def register_dataframe_method(method: Callable) -> Callable:
    """Register a function as a method attached to the Polars DataFrame.

    Example:
        >>> @register_dataframe_method # doctest: +SKIP
        >>> def print_column(df, col): # doctest: +SKIP
        ...    '''Print the dataframe column given''' # doctest: +SKIP
        ...    print(df[col]) # doctest: +SKIP

    !!! info "New in version 0.28.0"

    Args:
        method: Function to be registered as a method on the DataFrame.

    Returns:
        A Callable.
    """

    def inner(*args, **kwargs):

        class AccessorMethod(object):

            def __init__(self, polars_obj):
                self._obj = polars_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        pl.api.register_dataframe_namespace(method.__name__)(AccessorMethod)
        return method

    return inner()


def register_lazyframe_method(method: Callable) -> Callable:
    """Register a function as a method attached to the Polars LazyFrame.

    Example:
        >>> @register_lazyframe_method # doctest: +SKIP
        >>> def print_column(df, col): # doctest: +SKIP
        ...    '''Print the dataframe column given''' # doctest: +SKIP
        ...    print(df[col]) # doctest: +SKIP

    !!! info "New in version 0.28.0"

    Args:
        method: Function to be registered as a method on the LazyFrame.

    Returns:
        A Callable.
    """

    def inner(*args, **kwargs):

        class AccessorMethod(object):

            def __init__(self, polars_obj):
                self._obj = polars_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        pl.api.register_lazyframe_namespace(method.__name__)(AccessorMethod)

        return method

    return inner()


def register_expr_method(method):
    """Register a function as a method attached to a Polars Expression."""

    def inner(*args, **kwargs):

        class AccessorMethod(object):
            __doc__ = method.__doc__

            def __init__(self, polars_expr):
                self._obj = polars_expr

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        pl.api.register_expr_namespace(method.__name__)(AccessorMethod)

        return method

    return inner()
