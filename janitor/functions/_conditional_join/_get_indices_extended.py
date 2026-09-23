"""Multiple conditional-join indices backed by janitor-rs.

Mixed joins are range-led: the first range predicate creates the candidate
layout and later predicates filter it. When every predicate is ``!=``, the
first predicate creates flat physical position pairs and later predicates
filter those pairs directly.

The Rust boundary uses two first-predicate tuple shapes:

* range: ``(left, left_index, right, right_index,
  right_index_is_ordered, comparator)``;
* all-``!=``: ``(left, left_index, left_positions, left_null_positions,
  right, right_index, right_positions, right_null_positions,
  right_index_is_ordered, is_extension_array, comparator)``.

Residual predicates use ``(left, right, comparator)`` or, for null-aware
``!=``, ``(left, left_null_mask, right, right_null_mask,
is_extension_array, comparator)``. Residual arrays retain the full physical
layout because candidate positions index them directly. Rust does not sort,
align, or infer nullness; pyjanitor owns those responsibilities.
"""

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


def _empty_indices() -> dict:
    empty = np.array([], dtype=np.int64)
    return {"left_index": empty, "right_index": empty}


def _array_for(series: pd.Series) -> np.ndarray:
    """Convert one already-aligned Series without changing its row order."""
    return _convert_array_to_numpy(array=series._values)


def _positions_for(index: pd.Index) -> np.ndarray:
    """Return the physical positions carried by pyjanitor's RangeIndex."""
    return np.asarray(index, dtype=np.int64)


def _null_positions(series: pd.Series) -> np.ndarray | None:
    """Return full-layout null positions, or ``None`` when there are none."""
    nulls = series.isna().to_numpy()
    if not nulls.any():
        return None
    return _positions_for(series.index[nulls])


def _get_all_not_equal_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    keep: str,
    return_matching_indices: bool,
    kernel,
) -> dict:
    """Build all-``!=`` candidates, then filter residual predicates.

    The first predicate uses filtered, value-sorted non-null arrays for the
    Rust binary-search kernel. Its position maps and null positions preserve
    the correspondence with the full physical arrays. Every later predicate
    uses full-layout arrays because its candidate pairs already contain
    physical positions that index those arrays directly.

    Pyjanitor has already reset both frames to unique ``RangeIndex`` values.
    Rust trusts that alignment, but receives the full indexes and position
    metadata so it can return the original physical labels.
    """
    first_left_on, first_right_on, first_op = conditions[0]
    if first_op != "!=":
        raise ValueError("all-!= joins require != as the first predicate")

    left_series = df[first_left_on]
    right_series = right[first_right_on]
    left_nulls = left_series.isna()
    right_nulls = right_series.isna()
    left_nonnull = left_series.loc[~left_nulls]
    right_nonnull = right_series.loc[~right_nulls]
    right_sorted, right_index_is_sorted = _sort_if_not_monotonic(series=right_nonnull)

    left_index = _convert_array_to_numpy(array=left_series.index._values)
    right_index = _convert_array_to_numpy(array=right_series.index._values)
    first_left = _convert_array_to_numpy(array=left_nonnull._values)
    first_right = _convert_array_to_numpy(array=right_sorted._values)

    first_predicate = (
        first_left,
        left_index,
        _positions_for(left_nonnull.index),
        _null_positions(left_series),
        first_right,
        right_index,
        _positions_for(right_sorted.index),
        _null_positions(right_series),
        right_index_is_sorted,
        bool(pd.api.types.is_extension_array_dtype(left_series.dtype)),
        first_op,
    )
    predicates = [first_predicate]

    # Residual predicates retain the full physical layout. Their null masks
    # cover those full arrays, so candidate physical positions can index them
    # directly without another filtered-to-original mapping.
    for left_on, right_on, op in conditions[1:]:
        left_aligned = df[left_on]
        right_aligned = right[right_on]
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


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    keep: str,
    return_matching_indices: bool,
) -> dict:
    """Build multiple-condition indices with the Rust extended kernel.

    Mixed joins use the first range condition to establish the filtered, sorted
    physical layout. Every residual condition is reordered to that same layout
    before Rust sees it. All-``!=`` joins use a separate first-predicate path:
    the first predicate creates flat physical pairs and later predicates use
    full-layout arrays to filter those pairs. ``return_matching_indices``
    means that pyjanitor needs all materialized pairs, so it overrides the
    requested selection with ``keep="all"``.
    """
    all_not_equal = all(op == "!=" for _, _, op in conditions)
    if all_not_equal:
        first_left_on = conditions[0][0]
        first_values = df[first_left_on]
        first_values = first_values.loc[~first_values.isna()]
        first_dtype = _array_for(first_values).dtype.name
        try:
            kernel_name = _EXTENDED_KERNEL_NAMES[first_dtype]
            kernel = getattr(janitor_rs, kernel_name)
        except KeyError as error:
            raise TypeError(
                f"extended non-equi join does not support dtype "
                f"{df[first_left_on].dtype}"
            ) from error
        return _get_all_not_equal_indices(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            return_matching_indices=return_matching_indices,
            kernel=kernel,
        )

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
