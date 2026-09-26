"""Compute indices and aggregations for one conditional-join predicate.

This module prepares one predicate for the dtype-specific Rust kernels. The
caller has already reset both input frames to unique ``RangeIndex`` values;
those positions are therefore safe to carry through filtering and sorting.
PyJanitor performs null filtering, stable right-value sorting, and index
alignment before Rust evaluates candidates. Aggregation calls retain full
source arrays and materialize Rust's explicit output positions and match mask
as a pandas ``MultiIndex``.
"""

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
    _convert_array_to_numpy,
    _not_equal_layout_positions,
    _prepare_not_equal_anchor,
    _prepare_range_anchor,
)
from janitor.functions.utils import greater_than_join_types, less_than_join_types

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

# Each entry is `(forward, reverse)`. The explicit suffixes mirror the PyO3
# exports in `anchor_non_equi_join_agg.rs`, including Rust's `f64`/`f32` spelling.
_SINGLE_AGGREGATION_KERNELS = {
    "int64": (
        janitor_rs.single_join_aggregate_int64,
        janitor_rs.single_join_aggregate_reverse_int64,
    ),
    "int32": (
        janitor_rs.single_join_aggregate_int32,
        janitor_rs.single_join_aggregate_reverse_int32,
    ),
    "int16": (
        janitor_rs.single_join_aggregate_int16,
        janitor_rs.single_join_aggregate_reverse_int16,
    ),
    "int8": (
        janitor_rs.single_join_aggregate_int8,
        janitor_rs.single_join_aggregate_reverse_int8,
    ),
    "uint64": (
        janitor_rs.single_join_aggregate_uint64,
        janitor_rs.single_join_aggregate_reverse_uint64,
    ),
    "uint32": (
        janitor_rs.single_join_aggregate_uint32,
        janitor_rs.single_join_aggregate_reverse_uint32,
    ),
    "uint16": (
        janitor_rs.single_join_aggregate_uint16,
        janitor_rs.single_join_aggregate_reverse_uint16,
    ),
    "uint8": (
        janitor_rs.single_join_aggregate_uint8,
        janitor_rs.single_join_aggregate_reverse_uint8,
    ),
    "float64": (
        janitor_rs.single_join_aggregate_f64,
        janitor_rs.single_join_aggregate_reverse_f64,
    ),
    "float32": (
        janitor_rs.single_join_aggregate_f32,
        janitor_rs.single_join_aggregate_reverse_f32,
    ),
}


def _rust_anchor_non_equi_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    keep: str,
    return_materialized_indices: bool,
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

    Args:
        left: Left predicate values. For range operators this is the
            null-filtered left series; for ``!=`` it contains only non-null
            values.
        right: Right predicate values in binary-search order. For range
            operators and ``!=``, PyJanitor supplies the value-sorted layout.
        op: Comparison operator understood by the Rust kernel.
        keep: Requested output selection (``"all"``, ``"first"``,
            ``"last"``, or ``"any"``).
        return_materialized_indices: Whether the Rust wrapper should return
            materialized matching index arrays rather than only internal
            range-building information.
        right_index_is_ordered: Whether right-index labels are monotonically
            increasing in the value-sorted right layout. This affects
            first/last selection when the right values were reordered.
        left_index: Full left index-label array. When omitted, it is derived
            from ``left.index``.
        right_index: Full right index-label array. When omitted, it is derived
            from ``right.index``.
        left_positions: Positions of ``left`` values in the full left layout,
            used by ``!=`` after null filtering.
        left_null_positions: Full-layout left null positions, or ``None``.
        right_positions: Positions of ``right`` values in the full right
            layout, used by ``!=`` after sorting and null filtering.
        right_null_positions: Full-layout right null positions, or ``None``.
        is_extension_array: Whether the comparison uses pandas nullable
            extension-array semantics for nulls.

    Returns:
        A dictionary containing ``left_index`` and ``right_index`` arrays.
        When Rust reports no matches, both arrays are empty ``int64`` arrays.
    """
    left_values = _convert_array_to_numpy(array=left._values)
    try:
        kernel = _SINGLE_JOIN_KERNELS[left_values.dtype.name]
    except KeyError as error:
        raise TypeError(
            f"single non-equi join does not support dtype {left_values.dtype}"
        ) from error
    right_values = _convert_array_to_numpy(array=right._values)
    if left_index is None:
        left_index = _convert_array_to_numpy(array=left.index._values)
    if right_index is None:
        right_index = _convert_array_to_numpy(array=right.index._values)
    result = kernel(
        left_values,
        left_index,
        right_values,
        right_index,
        right_index_is_ordered,
        op,
        keep,
        bool(return_materialized_indices),
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


def _anchor_non_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    keep: str,
    return_materialized_indices: bool,
) -> dict:
    """Compute indices for a single conditional-join predicate.

    Range operators discard null rows before binary search because nulls do
    not satisfy ordering comparisons. ``!=`` is handled separately: its
    non-null values are searched, while null positions are passed explicitly
    so Rust can apply NumPy and pandas-extension null semantics correctly.

    Equality is intentionally not handled here; PyJanitor dispatches equi
    joins through its upstream equality implementation.

    Args:
        df: Left dataframe whose index has been reset to physical positions.
        right: Right dataframe whose index has been reset to physical
            positions.
        condition: A ``(left_column, right_column, operator)`` tuple.
        keep: Requested selection mode for matching right rows.
        return_materialized_indices: Whether the Rust wrapper should return
            the materialized matching index arrays required by the caller.

    Returns:
        A dictionary containing ``left_index`` and ``right_index`` arrays.

    Raises:
        ValueError: If equality reaches this single non-equi dispatcher.
        TypeError: If the predicate dtype is unsupported by the Rust kernels.
    """
    left_on, right_on, op = condition
    left_series = df[left_on]
    right_series = right[right_on]

    if op in less_than_join_types.union(greater_than_join_types):
        anchor = _prepare_range_anchor(left=left_series, right=right_series)
        if anchor is None:
            empty = np.array([], dtype=np.int64)
            return {"left_index": empty, "right_index": empty}
        return _rust_anchor_non_equi_join(
            left=anchor.left_values,
            right=anchor.right_values,
            op=op,
            keep=keep,
            return_materialized_indices=return_materialized_indices,
            right_index_is_ordered=anchor.right_index_is_ordered,
        )

    if op == "!=":
        anchor = _prepare_not_equal_anchor(left=left_series, right=right_series)
        return _rust_anchor_non_equi_join(
            left=anchor.left_values,
            right=anchor.right_values,
            op=op,
            keep=keep,
            return_materialized_indices=return_materialized_indices,
            right_index_is_ordered=anchor.right_index_is_ordered,
            left_index=anchor.left_index,
            right_index=anchor.right_index,
            left_positions=anchor.left_positions,
            left_null_positions=anchor.left_null_positions,
            right_positions=anchor.right_positions,
            right_null_positions=anchor.right_null_positions,
            is_extension_array=anchor.is_extension_array,
        )

    # Equality is dispatched through the equi-join paths upstream.
    raise ValueError(f"unsupported single-join operator: {op}")


def _aggregate_single(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    aggfunc: list[tuple],
    reverse: bool,
    return_matched: bool,
) -> pd.DataFrame:
    """Run one fused Rust aggregation for a range or ``!=`` predicate.

    Predicate and aggregation arrays use the same trimmed calculation layout.
    Rust returns ``None`` when no pair satisfies the predicate; otherwise it
    returns output positions and aggregation arrays, with the matched array
    included only when ``return_matched`` is true. The materializer uses the
    trimmed output index directly and chooses either a plain index or a
    boolean ``matched`` MultiIndex level accordingly.

    Args:
        df: Left dataframe with its physical ``RangeIndex``.
        right: Right dataframe with its physical ``RangeIndex``.
        condition: ``(left_column, right_column, operator)``. Supported
            operators are ``<``, ``<=``, ``>``, ``>=``, and ``!=``. Equality
            is handled upstream by PyJanitor.
        aggfunc: Non-empty ``(column, operation)`` requests. Operations may
            be ``sum``, ``prod``, ``min``, ``max``, ``count``, or ``size``.
        reverse: When false, aggregate right-side values into left output
            rows. When true, aggregate left-side values into right output
            rows.
        return_matched: Whether to request the per-output matched mask from
            Rust and expose it as a second MultiIndex level.

    Returns:
        A dataframe with one row per trimmed output position. An empty
        schema-only dataframe is returned when the predicate has no possible
        matches.

    Raises:
        TypeError: If the predicate dtype has no registered Rust kernel.
        ValueError: If the operator or aggregation request violates the Rust
            kernel contract.
    """
    left_on, right_on, operation = condition
    left_series = df[left_on]
    right_series = right[right_on]
    left_positions = left_null_positions = right_positions = None
    right_null_positions = None
    is_extension_array = False
    left_output_positions = _convert_array_to_numpy(array=df.index._values)
    right_output_positions = _convert_array_to_numpy(array=right.index._values)
    aggregation_source = right if not reverse else df
    output_index = right.index if reverse else df.index

    if operation in less_than_join_types.union(greater_than_join_types):
        anchor = _prepare_range_anchor(left=left_series, right=right_series)
        if anchor is None:
            return _empty_aggregation_result(
                source=right if not reverse else df,
                aggfunc=aggfunc,
            )
        left_array = anchor.left_array
        right_array = anchor.right_array
        left_positions = anchor.left_index
        right_positions = anchor.right_index
        aggregation_source = (
            right.loc[anchor.right_index] if not reverse else df.loc[anchor.left_index]
        )
        output_index = (
            right.index.take(anchor.right_index)
            if reverse
            else df.index.take(anchor.left_index)
        )
        # The range arrays may be filtered and sorted. These maps tell Rust
        # where each calculation-order row belongs in the trimmed output
        # layout. Rust uses the same map for its returned positions.
    elif operation == "!=":
        anchor = _prepare_not_equal_anchor(left=left_series, right=right_series)
        left_array = _convert_array_to_numpy(array=anchor.left_values._values)
        right_array = _convert_array_to_numpy(array=anchor.right_values._values)
        left_positions = anchor.left_positions
        right_positions = anchor.right_positions
        left_null_positions = anchor.left_null_positions
        right_null_positions = anchor.right_null_positions
        is_extension_array = anchor.is_extension_array
        left_output_positions = _not_equal_layout_positions(
            anchor.left_positions, anchor.left_null_positions
        )
        right_output_positions = _not_equal_layout_positions(
            anchor.right_positions, anchor.right_null_positions
        )
        aggregation_source = (
            right.iloc[right_output_positions]
            if not reverse
            else df.iloc[left_output_positions]
        )
        output_index = (
            right.index.take(right_output_positions)
            if reverse
            else df.index.take(left_output_positions)
        )
    else:
        raise ValueError("single Rust aggregation requires a non-equality predicate")

    kernel = _select_aggregation_kernel(
        registry=_SINGLE_AGGREGATION_KERNELS,
        dtype=left_array.dtype.name,
        reverse=reverse,
    )
    result = kernel(
        left_array,
        right_array,
        operation,
        left_positions,
        left_null_positions,
        right_positions,
        right_null_positions,
        is_extension_array,
        _aggregation_inputs(
            source=aggregation_source,
            aggfunc=aggfunc,
        ),
        left_output_positions,
        right_output_positions,
        return_matched,
    )
    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )
