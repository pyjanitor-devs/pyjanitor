"""Range-led multiple conditional-join indices backed by janitor-rs."""

from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
    _get_boolean_args_for_ne,
    _sort_if_not_monotonic,
    greater_than_join_types,
    less_than_join_types,
)

_EXTENDED_KERNEL_NAMES = {
    "int64": "single_join_extended_indices_int64",
    "int32": "single_join_extended_indices_int32",
    "int16": "single_join_extended_indices_int16",
    "int8": "single_join_extended_indices_int8",
    "uint64": "single_join_extended_indices_uint64",
    "uint32": "single_join_extended_indices_uint32",
    "uint16": "single_join_extended_indices_uint16",
    "uint8": "single_join_extended_indices_uint8",
    "float64": "single_join_extended_indices_f64",
    "float32": "single_join_extended_indices_f32",
}


def available() -> bool:
    """Return whether the installed Rust extension has the new kernels."""
    return hasattr(janitor_rs, next(iter(_EXTENDED_KERNEL_NAMES.values())))


def _empty_indices() -> dict:
    empty = np.array([], dtype=np.int64)
    return {"left_index": empty, "right_index": empty}


def _array_for(series: pd.Series) -> np.ndarray:
    """Convert one already-aligned Series without changing its row order."""
    return _convert_array_to_numpy(array=series._values)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    keep: str,
    return_matching_indices: bool,
) -> dict:
    """Build range-led multi-condition indices with the Rust extended kernel.

    The first range condition establishes the filtered, sorted physical layout.
    Every residual condition is reordered to that same layout before Rust sees
    it. `return_matching_indices` means that pyjanitor needs all materialized
    pairs, so it overrides the requested selection with ``keep="all"``.
    """
    first_position = next(
        (
            position
            for position, (_, _, op) in enumerate(conditions)
            if op in less_than_join_types.union(greater_than_join_types)
        ),
        None,
    )
    if first_position is None:
        raise ValueError("extended multiple join requires a range predicate")

    first = conditions[first_position]
    left_on, right_on, first_op = first
    left_series = df[left_on]
    right_series = right[right_on]

    left_nonnull = left_series.loc[~left_series.isna()]
    right_nonnull = right_series.loc[~right_series.isna()]
    if left_nonnull.empty or right_nonnull.empty:
        return _empty_indices()

    right_sorted, _ = _sort_if_not_monotonic(series=right_nonnull)
    left_positions = left_nonnull.index
    right_positions = right_sorted.index

    first_left = _array_for(left_nonnull)
    first_right = _array_for(right_sorted)
    first_dtype = first_left.dtype.name
    try:
        kernel_name = _EXTENDED_KERNEL_NAMES[first_dtype]
        kernel = getattr(janitor_rs, kernel_name)
    except KeyError as error:
        raise TypeError(
            f"extended non-equi join does not support dtype {first_left.dtype}"
        ) from error

    predicates = [
        (
            first_left,
            np.asarray(left_positions, dtype=np.int64),
            first_right,
            np.asarray(right_positions, dtype=np.int64),
            bool(right_sorted.index.is_monotonic_increasing),
            first_op,
        )
    ]

    for position, (left_on, right_on, op) in enumerate(conditions):
        if position == first_position:
            continue
        left_aligned = df.loc[left_positions, left_on]
        right_aligned = right.loc[right_positions, right_on]
        left_array = _array_for(left_aligned)
        right_array = _array_for(right_aligned)
        if op == "!=":
            left_booleans, right_booleans, is_extension_array = (
                _get_boolean_args_for_ne(
                    op=op,
                    left=left_aligned,
                    right=right_aligned,
                )
            )
            if left_booleans is None and right_booleans is None:
                predicates.append((left_array, right_array, op))
            else:
                predicates.append(
                    (
                        left_array,
                        left_booleans,
                        right_array,
                        right_booleans,
                        bool(is_extension_array),
                        op,
                    )
                )
        else:
            predicates.append((left_array, right_array, op))

    effective_keep = "all" if return_matching_indices else keep
    result = kernel(predicates, effective_keep)
    if result is None:
        return _empty_indices()
    return result
