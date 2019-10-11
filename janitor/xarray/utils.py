"""
Helpers to facilitate creating new XArray methods
"""


import xarray as xr

from xarray import register_dataarray_accessor, register_dataset_accessor
from functools import wraps


def make_accessor_wrapper(method):
    """
    Makes an XArray-compatible accessor to wrap a method to be added to an
    xr.DataArray, xr.Dataset, or both.

    :param method: A method which takes an XArray object and needed parameters.
    :return: The result of calling ``method``.
    """

    class XRAccessor:
        def __init__(self, xr_obj):
            self._xr_obj = xr_obj

        @wraps(method)
        def __call__(self, *args, **kwargs):
            return method(self._xr_obj, *args, **kwargs)

    return XRAccessor


def register_dataarray_method(method: callable):
    accessor_wrapper = make_accessor_wrapper(method)
    register_dataarray_accessor(method.__name__)(accessor_wrapper)

    return method


def register_dataset_method(method: callable):
    accessor_wrapper = make_accessor_wrapper(method)
    register_dataset_accessor(method.__name__)(accessor_wrapper)

    return method
