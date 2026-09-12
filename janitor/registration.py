"""Register pandas methods while preserving the source frame's metadata."""

from copy import deepcopy
from functools import wraps
from inspect import signature
from typing import Callable

import pandas as pd
import pandas_flavor as pf
from pandas.core.groupby.generic import DataFrameGroupBy


def preserve_dataframe_metadata(func: Callable) -> Callable:
    """Preserve source metadata when a DataFrame transformation returns a frame.

    Plain DataFrame results regain the source constructor; an explicitly
    returned DataFrame subtype is retained. The first argument owns the metadata,
    including for grouped inputs and
    multi-frame operations. Non-DataFrame results and in-place returns are
    unchanged. Attributes explicitly produced by the operation take precedence
    over inherited ``attrs``. Both direct and registered method calls use the
    same boundary; registering a function for multiple receiver types wraps it
    only once.

    Args:
        func: Function whose first argument is the source DataFrame or groupby.

    Returns:
        Callable: The function with a metadata-preserving return boundary.
    """
    if getattr(func, "_janitor_preserves_metadata", False):
        return func
    first_parameter = next(iter(signature(func).parameters), None)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Apply the shared policy after the original function returns."""
        source = args[0] if args else kwargs.get(first_parameter)
        if isinstance(source, DataFrameGroupBy):
            source = source.obj
        result = func(*args, **kwargs)
        if (
            not isinstance(source, pd.DataFrame)
            or not isinstance(result, pd.DataFrame)
            or result is source
            or (type(source) is pd.DataFrame and not source.attrs)
        ):
            return result
        result_attrs = deepcopy(result.attrs)
        if type(result) is pd.DataFrame and type(source) is not pd.DataFrame:
            result = source._constructor(result)
        result = result.__finalize__(source)
        result.attrs.update(result_attrs)
        return result

    wrapper._janitor_preserves_metadata = True
    return wrapper


def register_dataframe_method(func: Callable) -> Callable:
    """Register a DataFrame method with the shared metadata boundary.

    Args:
        func: DataFrame transformation to register.

    Returns:
        Callable: The registered function.
    """
    return pf.register_dataframe_method(preserve_dataframe_metadata(func))


def register_dataframe_groupby_method(func: Callable) -> Callable:
    """Register a grouped DataFrame method with the shared metadata boundary.

    Args:
        func: Grouped DataFrame transformation to register.

    Returns:
        Callable: The registered function.
    """
    return pf.register_dataframe_groupby_method(preserve_dataframe_metadata(func))
