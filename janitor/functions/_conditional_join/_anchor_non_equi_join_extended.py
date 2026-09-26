"""Single-anchor and all-``!=`` conditional joins backed by janitor-rs.

This module is deliberately narrower than its historical “extended” name
suggests. A range-led call here has exactly one binary-search anchor; every
remaining predicate is a residual filter evaluated against that anchor's
candidate window. Dual-range calls are routed to ``_range_join``, which owns
the second sorted right array, window intersection, and range aggregation
dispatch. All-``!=`` calls use their null-aware flat candidate stream here.

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

The aggregation entry point uses the same anchor and residual traversal as
the index entry point, but updates Rust aggregation state for each surviving
candidate instead of materializing candidate pairs. It returns a complete
trimmed output domain with an explicit ``matched`` level when requested.
"""

from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._aggregation_helpers import (
    _aggregation_inputs,
    _empty_aggregation_result,
    _materialize_aggregation_result,
    _select_aggregation_kernel,
)
from janitor.functions._conditional_join._helpers import (
    _build_residual_predicate,
    _convert_array_to_numpy,
    _get_boolean_args_for_ne,
    _maybe_remove_nulls_from_dataframe,
    _not_equal_layout_positions,
    _prepare_not_equal_anchor,
    _prepare_range_anchor,
)
from janitor.functions.utils import greater_than_join_types, less_than_join_types

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

# Each entry is `(forward, reverse)`. Keep this registry separate from the
# single-join registry because the extended Rust kernels have a different
# predicate contract and different PyO3 functions.
_EXTENDED_AGGREGATION_KERNELS = {
    "int64": (
        janitor_rs.single_join_extended_aggregate_int64,
        janitor_rs.single_join_extended_aggregate_reverse_int64,
    ),
    "int32": (
        janitor_rs.single_join_extended_aggregate_int32,
        janitor_rs.single_join_extended_aggregate_reverse_int32,
    ),
    "int16": (
        janitor_rs.single_join_extended_aggregate_int16,
        janitor_rs.single_join_extended_aggregate_reverse_int16,
    ),
    "int8": (
        janitor_rs.single_join_extended_aggregate_int8,
        janitor_rs.single_join_extended_aggregate_reverse_int8,
    ),
    "uint64": (
        janitor_rs.single_join_extended_aggregate_uint64,
        janitor_rs.single_join_extended_aggregate_reverse_uint64,
    ),
    "uint32": (
        janitor_rs.single_join_extended_aggregate_uint32,
        janitor_rs.single_join_extended_aggregate_reverse_uint32,
    ),
    "uint16": (
        janitor_rs.single_join_extended_aggregate_uint16,
        janitor_rs.single_join_extended_aggregate_reverse_uint16,
    ),
    "uint8": (
        janitor_rs.single_join_extended_aggregate_uint8,
        janitor_rs.single_join_extended_aggregate_reverse_uint8,
    ),
    "float64": (
        janitor_rs.single_join_extended_aggregate_f64,
        janitor_rs.single_join_extended_aggregate_reverse_f64,
    ),
    "float32": (
        janitor_rs.single_join_extended_aggregate_f32,
        janitor_rs.single_join_extended_aggregate_reverse_f32,
    ),
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
    anchor = _prepare_not_equal_anchor(left=left_series, right=right_series)

    first_predicate = (
        _convert_array_to_numpy(array=anchor.left_values._values),
        anchor.left_index,
        anchor.left_positions,
        anchor.left_null_positions,
        _convert_array_to_numpy(array=anchor.right_values._values),
        anchor.right_index,
        anchor.right_positions,
        anchor.right_null_positions,
        anchor.right_index_is_ordered,
        anchor.is_extension_array,
        first_op,
    )
    predicates = [first_predicate]

    # Residual predicates retain the full physical layout. Their null masks
    # cover those full arrays, so candidate physical positions can index them
    # directly without another filtered-to-original mapping. The right side
    # is already in the seed predicate's value-sorted order, so every residual
    # right array must use that same order.
    for left_on, right_on, op in conditions[1:]:
        predicates.append(
            _build_residual_predicate(
                left=df[left_on],
                right=right[right_on],
                operation=op,
            )
        )

    effective_keep = "all" if return_materialized_indices else keep
    result = kernel(predicates, effective_keep)
    if result is None:
        return _empty_indices()
    return result


def _aggregate_extended(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    aggfunc: list[tuple],
    reverse: bool,
    return_matched: bool,
) -> pd.DataFrame:
    """Run fused Rust aggregation for one anchor plus residual predicates.

    If every predicate is ``!=``, the first predicate supplies the null-aware
    candidate stream and later ``!=`` predicates filter those candidates. In
    In a mixed call, the first range predicate supplies the single binary-
    search window and every remaining predicate filters candidates inside that
    window. Dual-range calls are dispatched to ``_range_join`` before this
    function is entered. If a pair cannot satisfy the dual-range sorted-layout
    contract, this function remains the correctness-preserving fallback.
    Aggregation occurs while candidates are evaluated; no flat pair index is
    materialized.

    The first predicate's arrays may be filtered and the right range array may
    be sorted, but every residual predicate is rebuilt in the same physical
    left/right layout. Aggregation sources use the corresponding trimmed
    layout; Rust maps physical candidate positions into those local slots.

    Rust returns ``None`` when no candidate survives all predicates. Otherwise
    it returns output positions and aggregation arrays, with the matched array
    included only when ``return_matched`` is true. The shared materializer
    preserves the trimmed calculation layout and chooses a plain index or a
    boolean ``matched`` MultiIndex level accordingly.

    Args:
        df: Left dataframe with a unique physical ``RangeIndex``.
        right: Right dataframe with a unique physical ``RangeIndex``.
        conditions: Join predicates in user order. An all-``!=`` call is
            handled by the null-aware branch. A mixed call must contain at
            least one range predicate; this function uses the first range
            predicate as its only anchor and evaluates all other predicates as
            residuals.
        aggfunc: Non-empty ``(column, operation)`` requests for ``sum``,
            ``prod``, ``min``, ``max``, ``count``, or ``size``.
        reverse: When false, aggregate right-side values into left output
            rows. When true, aggregate left-side values into right output
            rows.
        return_matched: Whether to request the per-output matched mask from
            Rust and expose it as a second MultiIndex level.

    Returns:
        A dataframe indexed by ``(output_index, matched)``. Rows with no
        surviving candidate remain present with neutral or missing values.
        If no candidate survives anywhere, an empty schema-only dataframe is
        returned.

    Raises:
        TypeError: If the anchor dtype has no registered Rust kernel.
        ValueError: If predicates are malformed, residual arrays are not
            aligned, or an unsupported operator combination is requested.
    """
    all_not_equal = all(operation == "!=" for _, _, operation in conditions)
    if all_not_equal:
        first_left_on, first_right_on, _ = conditions[0]
        left_series = df[first_left_on]
        right_series = right[first_right_on]
        anchor = _prepare_not_equal_anchor(left=left_series, right=right_series)
        left_array = _convert_array_to_numpy(array=anchor.left_values._values)
        right_array = _convert_array_to_numpy(array=anchor.right_values._values)
        left_output_positions = _not_equal_layout_positions(
            anchor.left_positions, anchor.left_null_positions
        )
        right_output_positions = _not_equal_layout_positions(
            anchor.right_positions, anchor.right_null_positions
        )
        predicates = [
            (
                left_array,
                anchor.left_index,
                anchor.left_positions,
                anchor.left_null_positions,
                right_array,
                anchor.right_index,
                anchor.right_positions,
                anchor.right_null_positions,
                anchor.right_index_is_ordered,
                anchor.is_extension_array,
                left_output_positions,
                right_output_positions,
                "!=",
            )
        ]
        for left_on, right_on, operation in conditions[1:]:
            predicates.append(
                _build_residual_predicate(
                    left=df[left_on],
                    right=right[right_on],
                    operation=operation,
                )
            )
        aggregation_source = (
            right.iloc[right_output_positions]
            if not reverse
            else df.iloc[left_output_positions]
        )
        kernel = _select_aggregation_kernel(
            registry=_EXTENDED_AGGREGATION_KERNELS,
            dtype=left_array.dtype.name,
            reverse=reverse,
        )
        result = kernel(
            predicates,
            _aggregation_inputs(
                source=aggregation_source,
                aggfunc=aggfunc,
            ),
            return_matched,
        )
        return _materialize_aggregation_result(
            result=result,
            output_index=(
                right.index.take(right_output_positions)
                if reverse
                else df.index.take(left_output_positions)
            ),
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    non_ne_left = {left for left, _, operation in conditions if operation != "!="}
    non_ne_right = {
        right_name for _, right_name, operation in conditions if operation != "!="
    }
    filtered_df = _maybe_remove_nulls_from_dataframe(df, non_ne_left)
    filtered_right = _maybe_remove_nulls_from_dataframe(right, non_ne_right)
    if filtered_df is None or filtered_right is None:
        return _empty_aggregation_result(
            source=right if not reverse else df,
            aggfunc=aggfunc,
        )

    # This module owns one range anchor plus residual predicates. Calls with
    # two compatible range anchors are dispatched by `_range_join` before this
    # function is entered.
    first_position = next(
        position
        for position, (_, _, operation) in enumerate(conditions)
        if operation in less_than_join_types.union(greater_than_join_types)
    )
    first_left_on, first_right_on, first_operation = conditions[first_position]
    anchor = _prepare_range_anchor(
        left=filtered_df[first_left_on],
        right=filtered_right[first_right_on],
    )
    residual_positions = [
        position for position in range(len(conditions)) if position != first_position
    ]
    if anchor is None:
        return _empty_aggregation_result(
            source=right if not reverse else df,
            aggfunc=aggfunc,
        )
    predicates = [
        (
            anchor.left_array,
            anchor.left_index,
            anchor.right_array,
            anchor.right_index,
            anchor.right_index_is_ordered,
            _convert_array_to_numpy(array=df.index._values),
            _convert_array_to_numpy(array=right.index._values),
            first_operation,
        )
    ]
    aggregation_source = (
        filtered_right.loc[anchor.right_index]
        if not reverse
        else filtered_df.loc[anchor.left_index]
    )
    output_index = (
        right.index.take(anchor.right_index)
        if reverse
        else df.index.take(anchor.left_index)
    )
    for position in residual_positions:
        left_on, right_on, operation = conditions[position]
        predicates.append(
            _build_residual_predicate(
                left=filtered_df[left_on],
                right=filtered_right.loc[anchor.right_index, right_on],
                operation=operation,
            )
        )

    kernel = _select_aggregation_kernel(
        registry=_EXTENDED_AGGREGATION_KERNELS,
        dtype=anchor.left_array.dtype.name,
        reverse=reverse,
    )
    result = kernel(
        predicates,
        _aggregation_inputs(
            source=aggregation_source,
            aggfunc=aggfunc,
        ),
        return_matched,
    )
    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    keep: str,
    return_materialized_indices: bool,
) -> dict:
    """Build one-anchor indices with the Rust extended kernel.

    Mixed joins use the first range predicate to establish the filtered,
    sorted physical layout. Every later condition is reordered to that same
    layout before Rust sees it and is passed as a residual filter. Dual-range
    window intersection is routed to the dedicated range-join path. All-``!=``
    joins use a separate first-predicate path:
    the first predicate creates flat physical pairs and later predicates use
    full-layout arrays to filter those pairs. ``return_materialized_indices``
    means that pyjanitor needs all materialized pairs, so it overrides the
    requested selection with ``keep="all"``.

    Args:
        df: Left working dataframe with the reset physical ``RangeIndex``.
        right: Right working dataframe with the reset physical ``RangeIndex``.
        conditions: Join predicates. An all-``!=`` join uses its first
            predicate to build flat candidate pairs. Otherwise the first
            range predicate builds the only candidate window here; every later
            predicate is a residual filter. Calls with two compatible range
            anchors are handled by ``_range_join`` before this function.
        keep: ``"all"``, ``"first"``, ``"last"``, or ``"any"`` selection
            requested for the final indices.
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

    # This is the single-anchor extended path. The first range predicate
    # creates the only binary-search window; every other predicate, including
    # a second range comparison, remains a residual filter. The dual-range
    # optimizer is selected separately through `range_join_extended`.
    left_on, right_on, first_op = conditions[first_position]
    anchor = _prepare_range_anchor(df[left_on], right[right_on])
    residual_positions = [
        position for position in range(len(conditions)) if position != first_position
    ]
    if anchor is None:
        return _empty_indices()
    first_dtype = anchor.left_array.dtype.name
    try:
        kernel_name = _EXTENDED_KERNEL_NAMES[first_dtype]
        kernel = getattr(janitor_rs, kernel_name)
    except KeyError as error:
        raise TypeError(
            f"extended non-equi join does not support dtype {anchor.left_array.dtype}"
        ) from error

    predicates = [
        (
            anchor.left_array,
            anchor.left_index,
            anchor.right_array,
            anchor.right_index,
            anchor.right_index_is_ordered,
            first_op,
        )
    ]

    for position in residual_positions:
        left_on, right_on, op = conditions[position]
        # The left frame was filtered but never reordered. The right frame was
        # sorted for the seed range predicate, so only the right residual
        # series needs explicit positional reordering here.
        left_aligned = df[left_on]
        right_aligned = right.loc[anchor.right_index, right_on]
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
