"""Compute indices for one conditional-join predicate.

This module prepares one predicate for the dtype-specific Rust kernels. The
caller has already reset both input frames to unique ``RangeIndex`` values;
those positions are therefore safe to carry through filtering and sorting.
PyJanitor performs null filtering, stable right-value sorting, and index
alignment before Rust evaluates candidates.
"""

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._get_join_aggs import _build_agg_label
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


def _aggregation_inputs(source: pd.DataFrame, aggfunc: list[tuple]) -> list[tuple]:
    """Prepare full-layout aggregation arrays and authoritative null masks."""
    result = []
    for column_name, operation in aggfunc:
        series = source[column_name]
        result.append(
            (
                _convert_array_to_numpy(array=series._values),
                series.isna().to_numpy(dtype=bool),
                operation,
            )
        )
    return result


def _null_positions(series: pd.Series) -> np.ndarray | None:
    """Return full-layout physical positions of null rows."""
    nulls = series.isna().to_numpy(dtype=bool)
    if not nulls.any():
        return None
    return _convert_array_to_numpy(array=series.index[nulls]._values)


def _empty_aggregation_result(
    source: pd.DataFrame, aggfunc: list[tuple]
) -> pd.DataFrame:
    """Return an empty aggregation result with the requested output dtypes."""
    result = {}
    for column_name, operation in aggfunc:
        dtype = "int64" if operation == "size" else source[column_name].dtype
        result[_build_agg_label(column_name, operation)] = pd.array([], dtype=dtype)
    return pd.DataFrame(result, copy=False)


def _materialize_aggregation_result(
    result,
    output_index: pd.Index,
    source: pd.DataFrame,
    aggfunc: list[tuple],
) -> pd.DataFrame:
    """Convert Rust aggregation slots into a pandas result frame."""
    if result is None:
        return _empty_aggregation_result(source, aggfunc)
    matched = np.asarray(result[0], dtype=bool)
    index = output_index[matched]
    arrays = result[1]
    output = {}
    for position, (column_name, operation) in enumerate(aggfunc):
        values = np.asarray(arrays[position])
        if operation == "size":
            output[_build_agg_label(column_name, operation)] = values[matched]
            continue

        series = source[column_name]
        if operation in {"min", "max"}:
            invalid = values == -1
            safe_values = values.copy()
            safe_values[invalid] = 0
            values = series.iloc[safe_values].copy()
            values.iloc[invalid] = pd.NA
            values = values.array
        elif operation in {"sum", "prod"} and pd.api.types.is_extension_array_dtype(
            series.dtype
        ):
            values = pd.array(values, dtype=series.dtype)
        output[_build_agg_label(column_name, operation)] = values[matched]
    return pd.DataFrame(output, copy=False, index=index)


def _aggregation_kernel(name: str):
    """Resolve a dtype-specific Rust aggregation function."""
    try:
        return getattr(janitor_rs, name)
    except AttributeError as error:
        raise TypeError(f"Rust aggregation does not support dtype {name}") from error


def _rust_single_join(
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
        keep: Requested output selection (``"all"``, ``"first"``, or
            ``"last"``).
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


def _single_join(
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
        left_outcome = _null_checks_cond_join(series=left_series)
        right_outcome = _null_checks_cond_join(series=right_series)
        if (left_outcome is None) or (right_outcome is None):
            empty = np.array([], dtype=np.int64)
            return {"left_index": empty, "right_index": empty}
        left_nonnull, _ = left_outcome
        right_nonnull, _ = right_outcome
        right_sorted, right_index_is_ordered = _sort_if_not_monotonic(
            series=right_nonnull
        )
        return _rust_single_join(
            left=left_nonnull,
            right=right_sorted,
            op=op,
            keep=keep,
            return_materialized_indices=return_materialized_indices,
            right_index_is_ordered=right_index_is_ordered,
        )

    if op == "!=":
        left_is_null = left_series.isna().to_numpy(dtype=bool)
        right_is_null = right_series.isna().to_numpy(dtype=bool)
        left_nonnull = left_series.loc[~left_is_null]
        right_nonnull = right_series.loc[~right_is_null]

        # Without non-null values on either side, no binary-search candidate
        # can be built. Keep the original right layout so Rust can still use
        # the full index and null-position metadata for null semantics.
        if left_nonnull.empty or right_nonnull.empty:
            right_sorted = right_nonnull
            right_index_is_ordered = True
        else:
            right_sorted, right_index_is_ordered = _sort_if_not_monotonic(
                series=right_nonnull
            )
        left_index = _convert_array_to_numpy(array=left_series.index._values)
        right_index = _convert_array_to_numpy(array=right_series.index._values)
        left_positions = _convert_array_to_numpy(array=left_nonnull.index._values)
        right_positions = _convert_array_to_numpy(array=right_sorted.index._values)
        left_null_positions = _convert_array_to_numpy(
            array=left_series.index[left_is_null]._values
        )
        right_null_positions = _convert_array_to_numpy(
            array=right_series.index[right_is_null]._values
        )
        return _rust_single_join(
            left=left_nonnull,
            right=right_sorted,
            op=op,
            keep=keep,
            return_materialized_indices=return_materialized_indices,
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
            is_extension_array=bool(
                pd.api.types.is_extension_array_dtype(left_series.dtype)
            ),
        )

    # Equality is dispatched through the equi-join paths upstream.
    raise ValueError(f"unsupported single-join operator: {op}")


def _aggregate_single(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    aggfunc: list[tuple],
    reverse: bool,
) -> pd.DataFrame:
    """Run a fused Rust aggregation for one range or ``!=`` predicate.

    Predicate arrays may be filtered and sorted, while aggregation arrays keep
    the full physical layout needed by the Rust position updates.
    """
    left_on, right_on, operation = condition
    left_series = df[left_on]
    right_series = right[right_on]
    left_positions = left_null_positions = right_positions = None
    right_null_positions = None
    is_extension_array = False

    if operation in less_than_join_types.union(greater_than_join_types):
        left_outcome = _null_checks_cond_join(left_series)
        right_outcome = _null_checks_cond_join(right_series)
        if left_outcome is None or right_outcome is None:
            return _empty_aggregation_result(df if reverse else right, aggfunc)
        left_values, _ = left_outcome
        right_values, _ = right_outcome
        right_values, _ = _sort_if_not_monotonic(right_values)
        left_work = df.loc[left_values.index]
        right_work = right.loc[right_values.index]
        left_array = _convert_array_to_numpy(left_values._values)
        right_array = _convert_array_to_numpy(right_values._values)
    elif operation == "!=":
        left_null = left_series.isna().to_numpy(dtype=bool)
        right_null = right_series.isna().to_numpy(dtype=bool)
        left_values = left_series.loc[~left_null]
        right_values = right_series.loc[~right_null]
        if left_values.empty or right_values.empty:
            right_sorted = right_values
        else:
            right_sorted, _ = _sort_if_not_monotonic(right_values)
        left_work = df
        right_work = right
        left_array = _convert_array_to_numpy(left_values._values)
        right_array = _convert_array_to_numpy(right_sorted._values)
        left_positions = _convert_array_to_numpy(left_values.index._values)
        right_positions = _convert_array_to_numpy(right_sorted.index._values)
        left_null_positions = _null_positions(left_series)
        right_null_positions = _null_positions(right_series)
        is_extension_array = bool(
            pd.api.types.is_extension_array_dtype(left_series.dtype)
        )
    else:
        raise ValueError("single Rust aggregation requires a non-equality predicate")

    dtype = left_array.dtype.name
    function_name = (
        "single_join_aggregate_reverse_" if reverse else "single_join_aggregate_"
    ) + dtype
    result = _aggregation_kernel(function_name)(
        left_array,
        right_array,
        operation,
        left_positions,
        left_null_positions,
        right_positions,
        right_null_positions,
        is_extension_array,
        _aggregation_inputs(right_work if not reverse else left_work, aggfunc),
    )
    return _materialize_aggregation_result(
        result,
        right_work.index if reverse else left_work.index,
        right_work if not reverse else left_work,
        aggfunc,
    )
