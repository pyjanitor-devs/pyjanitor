"""Implementation of `single_value`."""

from __future__ import annotations

from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like, is_scalar

from janitor.utils import find_stack_level


def _normalize_missing(missing) -> list:
    """Normalize missing values into a list."""
    if missing is None:
        return [None]
    if is_list_like(missing) and not is_scalar(missing):
        return list(missing)
    return [missing]


def single_value(
    x: Iterable,
    missing: Sequence | None = None,
    warn_if_all_missing: bool = False,
    info: str | None = None,
):
    """Ensure that a vector contains only a single unique value.

    Missing values are excluded from the uniqueness check. If all values
    are missing, the first entry in `missing` is returned.

    Examples:
        >>> import janitor
        >>> janitor.single_value([1, 1, 1, None])
        1
        >>> janitor.single_value([None, "a"], missing=["a", None])
        'a'
        >>> janitor.single_value([1, 2, 3])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: More than one (3) value found (1, 2, 3)

    Args:
        x: Vector-like input (list, tuple, numpy array, or pandas Series).
        missing: Values to consider missing in `x`. Defaults to None.
        warn_if_all_missing: Whether to warn if all values are missing.
        info: Extra context to append to error messages.

    Raises:
        ValueError: If more than one unique non-missing value is found.

    Returns:
        The single unique value, or the first missing value if all missing.
    """
    if isinstance(x, pd.Series):
        series = x.copy()
    elif is_list_like(x) and not is_scalar(x):
        series = pd.Series(list(x), dtype="object")
    else:
        series = pd.Series([x], dtype="object")

    missing_values = _normalize_missing(missing)
    contains_na = any(pd.isna(value) for value in missing_values)
    missing_values_clean = [value for value in missing_values if not pd.isna(value)]

    mask_missing = pd.Series(False, index=series.index)
    if contains_na:
        mask_missing = series.isna()
    if missing_values_clean:
        mask_missing = mask_missing | series.isin(missing_values_clean)

    mask_found = ~mask_missing
    if warn_if_all_missing and not mask_found.any():
        warnings.warn(
            "All values are missing",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    found_values = pd.unique(series[mask_found])
    if len(found_values) == 0:
        return missing_values[0] if missing_values else None
    if len(found_values) == 1:
        value = found_values[0]
        if isinstance(value, np.generic):
            return value.item()
        return value

    values_str = ", ".join(map(str, found_values))
    message = f"More than one ({len(found_values)}) value found ({values_str})"
    if info is not None:
        message = f"{message}: {info}"
    raise ValueError(message)
