"""Implementation of `round_half_up` and `signif_half_up`."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd


def _prepare_numeric_input(
    x,
) -> Tuple[np.ndarray, Callable[[np.ndarray], object]]:
    """Convert input to a numeric array and build a wrapper for output."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)

        def wrap(result: np.ndarray):
            return pd.Series(result, index=x.index, name=x.name)

        return arr, wrap

    arr = np.asarray(x, dtype=float)
    is_scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)

    def wrap(result: np.ndarray):
        if is_scalar:
            return result[0]
        return result

    return arr, wrap


def round_half_up(x, digits: int = 0):
    """Round values using "half up" rounding (Excel-style).

    This differs from Python's built-in `round`, which uses bankers rounding.

    Examples:
        >>> import janitor
        >>> janitor.round_half_up(12.5)
        13.0
        >>> janitor.round_half_up(1.125, digits=2)
        1.13
        >>> janitor.round_half_up(-0.5)
        -1.0

    Args:
        x: Numeric scalar or array-like.
        digits: Number of decimal digits to round to.

    Returns:
        Rounded value(s).
    """
    arr, wrap = _prepare_numeric_input(x)
    factor = 10**digits
    posneg = np.sign(arr)
    z = np.abs(arr) * factor
    z = z + 0.5 + np.sqrt(np.finfo(float).eps)
    z = np.trunc(z)
    z = z / factor
    z = z * posneg
    return wrap(z)


def signif_half_up(x, digits: int = 6):
    """Round values to the specified number of significant digits.

    Uses "half up" rounding for midpoint values.

    Examples:
        >>> import janitor
        >>> janitor.signif_half_up(12.5, digits=2)
        13.0
        >>> janitor.signif_half_up(1.125, digits=3)
        1.13
        >>> janitor.signif_half_up(-2.5, digits=1)
        -3.0

    Args:
        x: Numeric scalar or array-like.
        digits: Number of significant digits.

    Returns:
        Rounded value(s).
    """
    arr, wrap = _prepare_numeric_input(x)
    z = arr.astype(float, copy=True)
    mask = (z != 0) & np.isfinite(z)
    if mask.any():
        y = np.zeros_like(z)
        y[mask] = 10 ** (digits - np.ceil(np.log10(np.abs(z[mask]))))
        z[mask] = round_half_up(z[mask] * y[mask]) / y[mask]
    return wrap(z)
