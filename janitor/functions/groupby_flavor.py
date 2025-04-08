# variant of pandas' accessor

# copied from pandas' accessor file - pandas/pandas/core/accessor.py
"""

accessor.py contains base classes for implementing accessor properties
that can be mixed into or pinned onto other pandas classes.

"""
from __future__ import annotations

import warnings
from functools import wraps
from typing import Callable

from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.util._exceptions import find_stack_level


class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : DataFrameGroupBy
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.
    """

    def __init__(self, name: str, accessor: DataFrameGroupBy) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # https://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # NDFrame
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def _register_accessor(name: str, cls: DataFrameGroupBy) -> Callable:
    """
    Register a custom accessor on {klass} objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    cls: DataFrameGroupBy

    Returns
    -------
    Callable
        A class decorator.
    """

    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        if not hasattr(cls, "_accessors"):
            cls._accessors = set()
        cls._accessors.add(name)
        return accessor

    return decorator


def register_groupby_accessor(name: str):

    return _register_accessor(name, DataFrameGroupBy)


def register_groupby_method(method: Callable) -> Callable:
    """Register a function as a method attached to the pandas DataFrameGroupBy.

    Example:
        >>> @register_groupby_method # doctest: +SKIP
        >>> def print_column(grp, col): # doctest: +SKIP
        ...    '''Print the dataframe column given''' # doctest: +SKIP
        ...    print(grp[col]) # doctest: +SKIP

    !!! info "New in version 0.32.0"

    Args:
        method: Function to be registered as a method on the DataFrame.

    Returns:
        A Callable.
    """

    def inner(*args: tuple, **kwargs: dict):

        class AccessorMethod(object):
            __doc__ = method.__doc__

            def __init__(self, obj):
                self._obj = obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        register_groupby_accessor(method.__name__)(AccessorMethod)
        return method

    return inner()
