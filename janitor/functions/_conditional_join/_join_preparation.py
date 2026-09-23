"""Shared physical-layout preparation for Rust conditional-join kernels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
    _sort_if_not_monotonic,
)


@dataclass(frozen=True)
class _NotEqualAnchor:
    """Prepared first-predicate data for a null-aware ``!=`` comparison.

    The value series contain only non-null rows. Their indexes are physical
    positions in the original full layouts, so they remain aligned with the
    position arrays and can be used to recover public index labels later.
    """

    left_values: pd.Series
    right_values: pd.Series
    left_index: np.ndarray
    right_index: np.ndarray
    left_positions: np.ndarray
    right_positions: np.ndarray
    left_null_positions: np.ndarray | None
    right_null_positions: np.ndarray | None
    right_index_is_ordered: bool
    is_extension_array: bool


def _null_positions(series: pd.Series) -> np.ndarray | None:
    """Return full-layout physical positions of null rows, if any."""
    nulls = series.isna().to_numpy(dtype=bool)
    if not nulls.any():
        return None
    return _convert_array_to_numpy(array=series.index[nulls]._values)


def _prepare_not_equal_anchor(left: pd.Series, right: pd.Series) -> _NotEqualAnchor:
    """Prepare filtered values and physical metadata for a ``!=`` anchor.

    PyJanitor owns dataframe index reset, value sorting, and dtype alignment.
    This helper only removes null values, sorts the non-null right values when
    both sides contain searchable values, and carries every physical mapping
    needed by Rust to handle null-generated candidates.

    Args:
        left: Full-layout left predicate series.
        right: Full-layout right predicate series.

    Returns:
        A named bundle containing non-null values, full index labels, filtered
        physical positions, null physical positions, and the ordering flags.
    """
    left_nulls = left.isna()
    right_nulls = right.isna()
    left_values = left.loc[~left_nulls]
    right_values = right.loc[~right_nulls]
    if left_values.empty or right_values.empty:
        right_sorted = right_values
        right_index_is_ordered = True
    else:
        right_sorted, right_index_is_ordered = _sort_if_not_monotonic(right_values)

    return _NotEqualAnchor(
        left_values=left_values,
        right_values=right_sorted,
        left_index=_convert_array_to_numpy(array=left.index._values),
        right_index=_convert_array_to_numpy(array=right.index._values),
        left_positions=_convert_array_to_numpy(array=left_values.index._values),
        right_positions=_convert_array_to_numpy(array=right_sorted.index._values),
        left_null_positions=_null_positions(left),
        right_null_positions=_null_positions(right),
        right_index_is_ordered=right_index_is_ordered,
        is_extension_array=bool(pd.api.types.is_extension_array_dtype(left.dtype)),
    )
