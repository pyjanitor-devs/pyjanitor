"""Compute indices for one conditional-join predicate."""

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
    _null_checks_cond_join,
    _sort_if_not_monotonic,
    greater_than_join_types,
    less_than_join_types,
)

_SINGLE_JOIN_KERNELS = {
    "int64": janitor_rs.single_join_indices_int64,
    "int32": janitor_rs.single_join_indices_int32,
    "int16": janitor_rs.single_join_indices_int16,
    "int8": janitor_rs.single_join_indices_int8,
    "uint64": janitor_rs.single_join_indices_uint64,
    "uint32": janitor_rs.single_join_indices_uint32,
    "uint16": janitor_rs.single_join_indices_uint16,
    "uint8": janitor_rs.single_join_indices_uint8,
    "float64": janitor_rs.single_join_indices_f64,
    "float32": janitor_rs.single_join_indices_f32,
}


def _index_array(index: pd.Index) -> np.ndarray:
    """Return the established int64 index representation for Rust."""
    return np.asarray(index.to_numpy(copy=False), dtype=np.int64)


def _kernel_for(array: pd.Series):
    values = _convert_array_to_numpy(array=array._values)
    try:
        return _SINGLE_JOIN_KERNELS[values.dtype.name], values
    except KeyError as error:
        raise TypeError(
            f"single non-equi join does not support dtype {values.dtype}"
        ) from error


def _rust_single_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    keep: str,
    return_matching_indices: bool,
    right_index_is_ordered: bool,
    left_index: np.ndarray | None = None,
    right_index: np.ndarray | None = None,
    left_positions: np.ndarray | None = None,
    left_null_positions: np.ndarray | None = None,
    right_positions: np.ndarray | None = None,
    right_null_positions: np.ndarray | None = None,
    is_extension_array: bool = False,
) -> dict:
    """Call a dtype-specific Rust single-join kernel.

    Range joins use the value arrays and their aligned index arrays directly.
    For ``!=``, ``left`` and ``right`` contain only non-null values, the
    explicit position maps point back to the full index arrays, and the
    optional null-position arrays contain original physical positions. Rust
    converts those positions into public index labels after candidate
    selection.
    """
    kernel, left_values = _kernel_for(left)
    right_values = _convert_array_to_numpy(array=right._values)
    if left_index is None:
        left_index = _index_array(left.index)
    if right_index is None:
        right_index = _index_array(right.index)
    result = kernel(
        left_values,
        left_index,
        right_values,
        right_index,
        right_index_is_ordered,
        op,
        keep,
        bool(return_matching_indices),
        left_positions,
        left_null_positions,
        right_positions,
        right_null_positions,
        is_extension_array,
    )
    if result is None:
        empty = np.array([], dtype=np.int64)
        return {"left_index": empty, "right_index": empty}
    return result


def _single_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    keep: str,
    return_matching_indices: bool,
) -> dict:
    """Compute indices for a single join using the fused Rust kernel."""
    left_on, right_on, op = condition
    left_series = df[left_on]
    right_series = right[right_on]

    if op in less_than_join_types or op in greater_than_join_types:
        left_outcome = _null_checks_cond_join(series=left_series)
        right_outcome = _null_checks_cond_join(series=right_series)
        if (left_outcome is None) or (right_outcome is None):
            empty = np.array([], dtype=np.int64)
            return {"left_index": empty, "right_index": empty}
        left_nonnull, _ = left_outcome
        right_nonnull, _ = right_outcome
        right_sorted, _ = _sort_if_not_monotonic(series=right_nonnull)
        right_index_is_ordered = right_sorted.index.is_monotonic_increasing
        return _rust_single_join(
            left=left_nonnull,
            right=right_sorted,
            op=op,
            keep=keep,
            return_matching_indices=return_matching_indices,
            right_index_is_ordered=right_index_is_ordered,
        )

    if op == "!=":
        left_is_null = left_series.isna().to_numpy(dtype=bool)
        right_is_null = right_series.isna().to_numpy(dtype=bool)
        left_nonnull = left_series.loc[~left_is_null]
        right_nonnull = right_series.loc[~right_is_null]

        if right_nonnull.empty:
            right_sorted = right_nonnull
        else:
            right_sorted, _ = _sort_if_not_monotonic(series=right_nonnull)
        right_index_is_ordered = right_sorted.index.is_monotonic_increasing
        left_index = _index_array(left_series.index)
        right_index = _index_array(right_series.index)
        left_positions = _index_array(left_nonnull.index)
        right_positions = _index_array(right_sorted.index)
        left_null_positions = _index_array(left_series.index[left_is_null])
        right_null_positions = _index_array(right_series.index[right_is_null])
        return _rust_single_join(
            left=left_nonnull,
            right=right_sorted,
            op=op,
            keep=keep,
            return_matching_indices=return_matching_indices,
            right_index_is_ordered=right_index_is_ordered,
            left_index=left_index,
            right_index=right_index,
            left_positions=left_positions,
            left_null_positions=left_null_positions
            if left_null_positions.size
            else None,
            right_positions=right_positions,
            right_null_positions=right_null_positions
            if right_null_positions.size
            else None,
            is_extension_array=pd.api.types.is_extension_array_dtype(left_series.dtype),
        )

    # Equality is dispatched through the equi-join paths upstream.
    raise ValueError(f"unsupported single-join operator: {op}")
