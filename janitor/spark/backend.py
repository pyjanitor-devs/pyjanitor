""" Backend functions for pyspark."""

import warnings
from functools import wraps

from janitor.utils import import_message


class CachedAccessor:
    """
    Custom property-like object (descriptor) for caching accessors.

    Parameters
    ----------
    name : str
        The namespace this will be accessed under, e.g. ``df.foo``
    accessor : cls
        The class with the extension methods.

    NOTE
    ----
    Modified based on pandas.core.accessor.
    """

    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # http://www.pydanny.com/cached-property.html
        setattr(obj, self._name, accessor_obj)
        return accessor_obj


def _register_accessor(name, cls):
    """
    NOTE
    ----
    Modified based on pandas.core.accessor.
    """

    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                "registration of accessor {!r} under name {!r} for type "
                "{!r} is overriding a preexisting attribute with the same "
                "name.".format(accessor, name, cls),
                UserWarning,
                stacklevel=2,
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_dataframe_accessor(name):
    """
    NOTE
    ----
    Modified based on pandas.core.accessor.
    """
    try:
        from pyspark.sql import DataFrame
    except ImportError:
        import_message(
            submodule="spark",
            package="pyspark",
            conda_channel="conda-forge",
            pip_install=True,
        )

    return _register_accessor(name, DataFrame)


def register_dataframe_method(method):
    """Register a function as a method attached to the Pyspark DataFrame.

    NOTE
    ----
    Modified based on pandas_flavor.register.
    """

    def inner(*args, **kwargs):
        class AccessorMethod:
            def __init__(self, pyspark_obj):
                self._obj = pyspark_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        register_dataframe_accessor(method.__name__)(AccessorMethod)

        return method

    return inner()
