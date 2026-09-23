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
    _maybe_remove_nulls_from_dataframe,
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
    """Return the standard empty index result.

    The Rust wrappers use empty ``int64`` arrays instead of ``None`` at this
    internal boundary. The public conditional-join layer decides how that
    empty result is represented to callers.

    Returns:
        A dictionary containing empty ``left_index`` and ``right_index``
        arrays.
    """
    empty = np.array([], dtype=np.int64)
    return {"left_index": empty, "right_index": empty}


def _null_positions(series: pd.Series) -> np.ndarray | None:
    """Return null positions in the series' full physical layout.

    The returned values are positions in the frame after PyJanitor has reset
    its index to a unique ``RangeIndex``. They are not boolean masks and are
    not public output labels. Rust uses them to add null-generated ``!=``
    candidates without receiving the original full value array.

    Args:
        series: The full-layout predicate series whose null positions should
            be collected.

    Returns:
        An ``int64`` NumPy array of null positions, or ``None`` when the
        series contains no nulls.
    """
    nulls = series.isna().to_numpy()
    if not nulls.any():
        return None
    return _convert_array_to_numpy(array=series.index[nulls]._values)


def _get_all_not_equal_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    keep: str,
    return_materialized_indices: bool,
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

    Args:
        df: Left working dataframe containing the columns in ``conditions``.
            Its ``RangeIndex`` identifies physical left-row positions.
        right: Right working dataframe containing the columns in
            ``conditions``. Its ``RangeIndex`` identifies physical right-row
            positions before the non-null values are sorted.
        conditions: Join predicates in user-supplied order. The first
            predicate must be ``!=`` for this path; later predicates filter
            the candidate pairs produced by it.
        keep: Selection mode requested by the caller. It is used unless
            ``return_materialized_indices`` is true.
        return_materialized_indices: Whether every surviving pair must be
            returned. This overrides ``keep`` with ``"all"`` for building
            blocks and aggregation preparation.
        kernel: Dtype-specific Rust callable selected from
            ``_EXTENDED_KERNEL_NAMES``.

    Returns:
        A dictionary containing ``left_index`` and ``right_index`` arrays.
        Both arrays contain public index labels, not filtered-array offsets.
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
    # Without non-null values on either side, no binary-search candidate can
    # be built. Keep the original right layout so Rust can still use the full
    # index and null-position metadata for null semantics.
    if left_nonnull.empty or right_nonnull.empty:
        right_sorted = right_nonnull
        right_index_is_ordered = True
    else:
        right_sorted, right_index_is_ordered = _sort_if_not_monotonic(
            series=right_nonnull
        )

    left_index = _convert_array_to_numpy(array=left_series.index._values)
    right_index = _convert_array_to_numpy(array=right_series.index._values)
    first_left = _convert_array_to_numpy(array=left_nonnull._values)
    first_right = _convert_array_to_numpy(array=right_sorted._values)
    left_positions = _convert_array_to_numpy(array=left_nonnull.index._values)
    right_positions = _convert_array_to_numpy(array=right_sorted.index._values)
    left_null_positions = _null_positions(left_series)
    right_null_positions = _null_positions(right_series)

    first_predicate = (
        first_left,
        left_index,
        left_positions,
        left_null_positions,
        first_right,
        right_index,
        right_positions,
        right_null_positions,
        right_index_is_ordered,
        bool(pd.api.types.is_extension_array_dtype(left_series.dtype)),
        first_op,
    )
    predicates = [first_predicate]

    # Residual predicates retain the full physical layout. Their null masks
    # cover those full arrays, so candidate physical positions can index them
    # directly without another filtered-to-original mapping. The right side
    # is already in the seed predicate's value-sorted order, so every residual
    # right array must use that same order.
    for left_on, right_on, op in conditions[1:]:
        left_aligned = df[left_on]
        right_aligned = right[right_on]
        left_array = _convert_array_to_numpy(array=left_aligned._values)
        right_array = _convert_array_to_numpy(array=right_aligned._values)
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

    effective_keep = "all" if return_materialized_indices else keep
    result = kernel(predicates, effective_keep)
    if result is None:
        return _empty_indices()
    return result


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    keep: str,
    return_materialized_indices: bool,
) -> dict:
    """Build multiple-condition indices with the Rust extended kernel.

    Mixed joins use the first range condition to establish the filtered, sorted
    physical layout. Every residual condition is reordered to that same layout
    before Rust sees it. All-``!=`` joins use a separate first-predicate path:
    the first predicate creates flat physical pairs and later predicates use
    full-layout arrays to filter those pairs. ``return_materialized_indices``
    means that pyjanitor needs all materialized pairs, so it overrides the
    requested selection with ``keep="all"``.

    Args:
        df: Left working dataframe with the reset physical ``RangeIndex``.
        right: Right working dataframe with the reset physical ``RangeIndex``.
        conditions: Join predicates. An all-``!=`` join uses its first
            predicate to build flat candidate pairs. A mixed join uses the
            first range predicate (in condition order) to build candidate
            windows.
        keep: ``"all"``, ``"first"``, or ``"last"`` selection requested for
            the final indices.
        return_materialized_indices: Force all surviving pairs to be
            materialized. This is required when the caller needs building
            blocks or aggregation inputs and therefore overrides ``keep``.

    Returns:
        A dictionary with ``left_index`` and ``right_index`` arrays. Empty
        arrays represent no matches at this internal PyJanitor boundary.

    Raises:
        TypeError: If the seed predicate dtype has no registered Rust kernel.
        ValueError: If a mixed join has no range predicate or an all-``!=``
            join does not begin with ``!=``.
    """
    all_not_equal = all(op == "!=" for _, _, op in conditions)
    if all_not_equal:
        first_left_on = conditions[0][0]
        first_values = df[first_left_on]
        first_values = first_values.loc[~first_values.isna()]
        first_dtype = _convert_array_to_numpy(array=first_values._values).dtype.name
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
            return_materialized_indices=return_materialized_indices,
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

    # A null cannot satisfy any non-``!=`` predicate, including a residual
    # ``==`` predicate. Filter those rows once at the dataframe level so the
    # seed range predicate and every residual non-``!=`` predicate receive
    # null-free, positionally aligned arrays. Keep nulls in ``!=`` columns;
    # their masks are handled separately below.
    non_ne_left_columns = {left_on for left_on, _, op in conditions if op != "!="}
    non_ne_right_columns = {right_on for _, right_on, op in conditions if op != "!="}
    df = _maybe_remove_nulls_from_dataframe(
        df=df,
        columns=non_ne_left_columns,
    )
    if df is None:
        return _empty_indices()

    right = _maybe_remove_nulls_from_dataframe(
        df=right,
        columns=non_ne_right_columns,
    )
    if right is None:
        return _empty_indices()

    first = conditions[first_position]
    left_on, right_on, first_op = first
    left_series = df[left_on]
    right_series = right[right_on]

    if left_series.empty or right_series.empty:
        return _empty_indices()

    # The right values must be sorted before Rust performs binary searches.
    # The helper also reports whether the resulting right-index labels remain
    # monotonically ordered in that value-sorted layout.
    right_sorted, right_index_is_ordered = _sort_if_not_monotonic(series=right_series)
    left_positions = _convert_array_to_numpy(array=left_series.index._values)
    right_positions = _convert_array_to_numpy(array=right_sorted.index._values)
    first_left = _convert_array_to_numpy(array=left_series._values)
    first_right = _convert_array_to_numpy(array=right_sorted._values)
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
            left_positions,
            first_right,
            right_positions,
            right_index_is_ordered,
            first_op,
        )
    ]

    for position, (left_on, right_on, op) in enumerate(conditions):
        if position == first_position:
            continue
        # The left frame was filtered but never reordered. The right frame was
        # sorted for the seed range predicate, so only the right residual
        # series needs explicit positional reordering here.
        left_aligned = df[left_on]
        right_aligned = right.loc[right_positions, right_on]
        left_array = _convert_array_to_numpy(array=left_aligned._values)
        right_array = _convert_array_to_numpy(array=right_aligned._values)
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
                        is_extension_array,
                        op,
                    )
                )
        else:
            predicates.append((left_array, right_array, op))

    effective_keep = "all" if return_materialized_indices else keep
    result = kernel(predicates, effective_keep)
    if result is None:
        return _empty_indices()
    return result
