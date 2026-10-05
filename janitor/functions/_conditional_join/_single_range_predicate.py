"""Python boundary for single range and range-first conditional joins.

This module owns the Python preparation that sits immediately before the
Rust range kernels.  It deliberately keeps the pandas-facing concerns here:

* removing null anchor values, since ordinary range comparisons do not match
  nulls;
* sorting the right anchor values while retaining their original physical
  positions;
* selecting a dtype-specialised PyO3 function; and
* converting Rust's positional result into pandas indexes and aggregation
  columns.

The Rust kernels never receive pandas labels.  Every ``*_index`` array passed
through this module contains a zero-based physical position in the original
left or right dataframe.  A sorted right anchor therefore carries two pieces
of information: ``right`` is the sorted search layout, while ``right_index``
maps each sorted slot back to the original right row.

There are two related ABI shapes:

``single_range_predicate_indices_*``
    One range predicate.  The right values and right-position array are
    sorted together.  ``keep`` and ``right_index_is_ordered`` are handled by
    Rust after the binary-search window is built.

``range_anchor_extended_*``
    A range predicate followed by residual predicates.  The first tuple uses
    compact anchor values and the operator.  The physical position arrays are
    passed separately; aggregation receives arrays aligned to those compact
    layouts.
"""

from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers
from janitor.functions._conditional_join._aggregation_helpers import (
    _aggregation_inputs,
    _empty_aggregation_result,
    _materialize_aggregation_result,
    _unmatched_aggregation_result,
)

_SINGLE_RANGE_FUNCTIONS = {
    "int64": janitor_rs.single_range_predicate_indices_int64,
    "int32": janitor_rs.single_range_predicate_indices_int32,
    "int16": janitor_rs.single_range_predicate_indices_int16,
    "int8": janitor_rs.single_range_predicate_indices_int8,
    "uint64": janitor_rs.single_range_predicate_indices_uint64,
    "uint32": janitor_rs.single_range_predicate_indices_uint32,
    "uint16": janitor_rs.single_range_predicate_indices_uint16,
    "uint8": janitor_rs.single_range_predicate_indices_uint8,
    "float64": janitor_rs.single_range_predicate_indices_f64,
    "float32": janitor_rs.single_range_predicate_indices_f32,
}

_SINGLE_RANGE_AGGREGATE_FUNCTIONS = {
    "int64": (
        janitor_rs.single_range_aggregate_int64,
        janitor_rs.single_range_aggregate_reverse_int64,
    ),
    "int32": (
        janitor_rs.single_range_aggregate_int32,
        janitor_rs.single_range_aggregate_reverse_int32,
    ),
    "int16": (
        janitor_rs.single_range_aggregate_int16,
        janitor_rs.single_range_aggregate_reverse_int16,
    ),
    "int8": (
        janitor_rs.single_range_aggregate_int8,
        janitor_rs.single_range_aggregate_reverse_int8,
    ),
    "uint64": (
        janitor_rs.single_range_aggregate_uint64,
        janitor_rs.single_range_aggregate_reverse_uint64,
    ),
    "uint32": (
        janitor_rs.single_range_aggregate_uint32,
        janitor_rs.single_range_aggregate_reverse_uint32,
    ),
    "uint16": (
        janitor_rs.single_range_aggregate_uint16,
        janitor_rs.single_range_aggregate_reverse_uint16,
    ),
    "uint8": (
        janitor_rs.single_range_aggregate_uint8,
        janitor_rs.single_range_aggregate_reverse_uint8,
    ),
    "float64": (
        janitor_rs.single_range_aggregate_f64,
        janitor_rs.single_range_aggregate_reverse_f64,
    ),
    "float32": (
        janitor_rs.single_range_aggregate_f32,
        janitor_rs.single_range_aggregate_reverse_f32,
    ),
}

# Multi-predicate range-first kernels.  The first predicate owns the sorted
# right-hand layout; later predicates are residuals evaluated in that layout.
_MULTI_RANGE_FUNCTIONS = {
    "int64": janitor_rs.range_anchor_extended_indices_int64,
    "int32": janitor_rs.range_anchor_extended_indices_int32,
    "int16": janitor_rs.range_anchor_extended_indices_int16,
    "int8": janitor_rs.range_anchor_extended_indices_int8,
    "uint64": janitor_rs.range_anchor_extended_indices_uint64,
    "uint32": janitor_rs.range_anchor_extended_indices_uint32,
    "uint16": janitor_rs.range_anchor_extended_indices_uint16,
    "uint8": janitor_rs.range_anchor_extended_indices_uint8,
    "float64": janitor_rs.range_anchor_extended_indices_f64,
    "float32": janitor_rs.range_anchor_extended_indices_f32,
}

_MULTI_RANGE_AGGREGATE_FUNCTIONS = {
    False: janitor_rs.range_anchor_extended_aggregate,
    True: janitor_rs.range_anchor_extended_aggregate_reverse,
}


def _get_single_range_function(dtype: np.dtype) -> object:
    """Return the single-range index kernel for a NumPy dtype.

    Args:
        dtype: NumPy dtype of the prepared right anchor values.

    Returns:
        The dtype-specialised PyO3 function exported by ``janitor_rs``.

    Raises:
        KeyError: If the Rust extension has no kernel for ``dtype``.
    """
    dtype_name = dtype.name
    try:
        return _SINGLE_RANGE_FUNCTIONS[dtype_name]
    except KeyError as err:
        raise KeyError(f"Unsupported single-range dtype: {dtype_name!r}") from err


def _get_single_range_aggregation_function(dtype: np.dtype, reverse: bool) -> object:
    """Select the single-range aggregation kernel and direction.

    Args:
        dtype: NumPy dtype of the range anchor.
        reverse: When true, aggregate left values into right output slots;
            otherwise aggregate right values into left output slots.

    Returns:
        The forward or reverse dtype-specialised PyO3 function.

    Raises:
        KeyError: If ``dtype`` is unsupported by the Rust aggregation family.
    """
    dtype_name = dtype.name
    try:
        forward, reverse_function = _SINGLE_RANGE_AGGREGATE_FUNCTIONS[dtype_name]
    except KeyError as err:
        raise KeyError(
            f"Unsupported single-range aggregation dtype: {dtype_name!r}"
        ) from err
    return reverse_function if reverse else forward


def _get_multi_range_function(dtype: np.dtype) -> object:
    """Return the range-first residual index kernel for ``dtype``.

    The first predicate owns the sorted right search layout.  Rust evaluates
    every later predicate against that layout before applying ``keep``.

    Args:
        dtype: NumPy dtype of the range anchor.

    Returns:
        The matching ``range_anchor_extended_indices_*`` PyO3 function.
    """
    try:
        return _MULTI_RANGE_FUNCTIONS[dtype.name]
    except KeyError as err:
        raise KeyError(f"Unsupported multi-range dtype: {dtype.name!r}") from err


def _get_multi_range_aggregation_function(reverse: bool) -> object:
    """Return the range-first aggregation kernel for one join direction.

    Args:
        reverse: Select the reverse kernel when true.  Reverse aggregation
            produces one slot per right row and reads source values from the
            left dataframe; forward aggregation does the opposite.

    Returns:
        The registered range-first aggregation PyO3 function.
    """
    return _MULTI_RANGE_AGGREGATE_FUNCTIONS[bool(reverse)]


def _preparatory_work_single_join(
    df: pd.DataFrame, right: pd.DataFrame, condition: tuple[str, str, str]
) -> tuple | None:
    """Prepare one non-null range anchor for the Rust index kernel.

    The returned series retain their original dataframe indexes.  The right
    series is sorted by value, but its index travels with it, so Rust can
    return original physical right positions rather than sorted offsets.

    Args:
        df: Left dataframe.
        right: Right dataframe.
        condition: ``(left_column, right_column, operator)``.

    Returns:
        ``(left_values, right_values, right_index_is_ordered)`` or ``None``
        when either input is empty or one side contains only null anchors.
    """
    if df.empty or right.empty:
        return None
    left_column = condition.left
    left_column = df[left_column]
    booleans = left_column.isna()
    if booleans.all():
        return None
    if booleans.any():
        left_column = left_column[~booleans]
    right_column = condition.right
    right_column = right[right_column]
    booleans = right_column.isna()
    if booleans.all():
        return None
    if booleans.any():
        right_column = right_column[~booleans]

    right_column, right_index_is_ordered = _helpers._sort_if_not_monotonic(
        series=right_column
    )

    return left_column, right_column, right_index_is_ordered


def _compute_single_range_join(
    df,
    right,
    condition,
    keep,
    return_building_blocks,
    *,
    how="inner",
    df_columns=slice(None),
    right_columns=slice(None),
    indicator=False,
    include_join_positions=False,
    return_matching_indices=True,
):
    """Build index pairs for a single range predicate.

    Args:
        df: Left dataframe.
        right: Right dataframe.
        condition: Single range condition.
        keep: ``"all"``, ``"any"``, ``"first"``, or ``"last"``.
        return_building_blocks: Return ``starts``/``ends`` windows instead of
            materialised pairs when true.
        how: Join shape used when materializing dataframe output.
        df_columns: Left columns to retain when materializing output.
        right_columns: Right columns to retain when materializing output.
        indicator: Whether to add the merge indicator column.
        include_join_positions: Whether to include physical pair positions in
            the materialized result index.
        return_matching_indices: Return the physical index dictionary instead
            of a dataframe.

    Returns:
        A physical-index dictionary for index callers, or a materialized
        dataframe when ``return_matching_indices`` is false. No-match results
        use the corresponding empty form.
    """
    outcome = _preparatory_work_single_join(df=df, right=right, condition=condition)
    if outcome is None:
        result = _helpers._empty_indices()
        return _helpers._materialize_or_return_indices(
            result=result,
            df=df,
            right=right,
            how=how,
            df_columns=df_columns,
            right_columns=right_columns,
            indicator=indicator,
            include_join_positions=include_join_positions,
            return_matching_indices=return_matching_indices,
            return_building_blocks=return_building_blocks,
        )

    operator = condition.op
    left_column, right_column, right_index_is_ordered = outcome
    left_index = _helpers._convert_array_to_numpy(array=left_column.index._values)
    left_array = _helpers._convert_array_to_numpy(array=left_column._values)
    right_index = _helpers._convert_array_to_numpy(array=right_column.index._values)
    right_array = _helpers._convert_array_to_numpy(array=right_column._values)
    anchor_dtype = right_array.dtype
    function = _get_single_range_function(anchor_dtype)
    result = function(
        left_index=left_index,
        left=left_array,
        right_index=right_index,
        right=right_array,
        right_index_is_ordered=right_index_is_ordered,
        operator=operator,
        keep=keep,
        return_building_blocks=return_building_blocks,
    )
    result = _helpers._empty_indices() if result is None else result
    return _helpers._materialize_or_return_indices(
        result=result,
        df=df,
        right=right,
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        indicator=indicator,
        include_join_positions=include_join_positions,
        return_matching_indices=return_matching_indices,
        return_building_blocks=return_building_blocks,
    )


def _aggregate_single_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple[str, str, str],
    aggfunc: list[tuple],
    return_matched: bool,
    reverse: bool,
) -> pd.DataFrame:
    """Aggregate one range predicate through the Rust range kernel.

    Forward aggregation has one output slot per left row and reads values
    from the right source.  Reverse aggregation has one output slot per right
    row and reads values from the left source.  Python supplies aggregation
    arrays in the same prepared layout as the predicate arrays, allowing the
    specialised single-range Rust kernels to use their prefix/suffix paths.

    Args:
        df: Left dataframe.
        right: Right dataframe.
        condition: Single range condition.
        aggfunc: ``(column_name, operation)`` aggregation requests.
        return_matched: Include the match mask in the output index.
        reverse: Aggregate left values into right output slots when true.

    Returns:
        A pandas dataframe indexed by the participating side's physical
        positions, with one output column per aggregation request.
    """
    aggregation_source = df if reverse else right
    outcome = _preparatory_work_single_join(df=df, right=right, condition=condition)
    if outcome is None:
        return _empty_aggregation_result(
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    operator = condition.op
    left_column, right_column, _ = outcome
    source_indexer = left_column.index if reverse else right_column.index
    left_index = _helpers._convert_array_to_numpy(array=left_column.index._values)
    left_array = _helpers._convert_array_to_numpy(array=left_column._values)
    right_index = _helpers._convert_array_to_numpy(array=right_column.index._values)
    right_array = _helpers._convert_array_to_numpy(array=right_column._values)
    output_index = right_column.index if reverse else left_column.index
    anchor_dtype = right_array.dtype
    aggregation_inputs = _aggregation_inputs(
        source=aggregation_source,
        aggfunc=aggfunc,
        indexer=source_indexer,
    )
    function = _get_single_range_aggregation_function(anchor_dtype, reverse)
    result = function(
        left_index=left_index,
        left=left_array,
        right_index=right_index,
        right=right_array,
        operator=operator,
        aggregations=aggregation_inputs,
        return_matched=return_matched,
    )
    if result is None:
        return _unmatched_aggregation_result(
            output_index=output_index,
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )
    # for performance reasons the returned data may not be full length index
    # i.e output index len may be less than the original right index len
    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source_index=source_indexer,
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )


def _preparatory_work_multi_join(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list[tuple[str, str, str]]
) -> tuple | None:
    """Prepare a range-first multi-predicate join.

    Null rows are removed from ordinary range anchors.  The first range
    predicate is converted to compact value arrays and sorted on the right;
    each residual predicate is then indexed into that same physical layout.
    The residual tuple list is consumed unchanged by both the index and
    aggregation Rust entry points.

    Args:
        df: Left dataframe.
        right: Right dataframe.
        conditions: Ordered conditional-join predicates.  One range
            predicate must be available to act as the anchor.

    Returns:
        ``(left_index, right_index, anchor, residuals, anchor_dtype)``.
        The first predicate is a three-field value/operator tuple; the
        indexers are retained for residual alignment and converted to NumPy
        physical-position arrays only at the Rust call site.
    """
    if df.empty or right.empty:
        return None
    left_index = _helpers._get_indexer_for_non_null_rows(
        df=df,
        columns_and_ops=[(condition.left, condition.op) for condition in conditions],
    )
    if left_index is None:
        return None
    right_index = _helpers._get_indexer_for_non_null_rows(
        df=right,
        columns_and_ops=[(condition.right, condition.op) for condition in conditions],
    )
    if right_index is None:
        return None

    # range_predicate
    anchor_position = next(
        # Keep the condition's original position: duplicate predicates need a
        # unique occurrence identifier when the anchor is removed below.
        position
        for position, condition in enumerate(conditions)
        if condition.op
        in _helpers.less_than_join_types.union(_helpers.greater_than_join_types)
    )
    anchor_predicate = conditions[anchor_position]
    # Compare original positions so only the selected occurrence is omitted;
    # duplicate predicates remain distinguishable and residual order is kept.
    rest = [
        condition
        for position, condition in enumerate(conditions)
        if position != anchor_position
    ]

    left_column, right_column, op = (
        anchor_predicate.left,
        anchor_predicate.right,
        anchor_predicate.op,
    )
    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    left_column = df.loc[left_index, left_column]
    right_column = right.loc[right_index, right_column]
    right_column, right_index_is_ordered = _helpers._sort_if_not_monotonic(
        series=right_column
    )
    if not right_index_is_ordered:
        right_index = right_column.index
    right_array = _helpers._convert_array_to_numpy(array=right_column._values)
    anchor_dtype = right_array.dtype
    anchor_predicate = (
        _helpers._convert_array_to_numpy(array=left_column._values),
        right_array,
        op,
    )
    residual_predicates = []
    for condition in rest:
        residual_predicate = _helpers._build_residual_predicate(
            left=df.loc[left_index, condition.left],
            right=right.loc[right_index, condition.right],
            operation=condition.op,
            left_index=left_index,
            right_index=right_index,
        )
        residual_predicates.append(residual_predicate)

    return (
        left_index,
        right_index,
        anchor_predicate,
        residual_predicates,
        anchor_dtype,
    )


def _compute_multi_range_join(
    df,
    right,
    conditions,
    keep,
    *,
    how="inner",
    df_columns=slice(None),
    right_columns=slice(None),
    indicator=False,
    include_join_positions=False,
    return_matching_indices=True,
    return_building_blocks=False,
):
    """Build pairs for a range-first join with residual predicates.

    ``keep`` is applied in Rust only after the anchor window and every
    residual predicate have passed.  This matters for ``first``/``last``:
    choosing before residual filtering could return a rejected candidate.

    Args:
        df: Left dataframe.
        right: Right dataframe.
        conditions: One range anchor followed by residual predicates.
        keep: Output policy applied after residual filtering.
        how: Join shape used when materializing dataframe output.
        df_columns: Left columns to retain when materializing output.
        right_columns: Right columns to retain when materializing output.
        indicator: Whether to add the merge indicator column.
        include_join_positions: Whether to include physical pair positions in
            the materialized result index.
        return_matching_indices: Return the physical index dictionary instead
            of a dataframe.
        return_building_blocks: Preserve any returned building-block arrays.

    Returns:
        A physical-index dictionary for index callers, or a materialized
        dataframe when ``return_matching_indices`` is false. No-match results
        use the corresponding empty form.
    """
    outcome = _preparatory_work_multi_join(df=df, right=right, conditions=conditions)
    if outcome is None:
        result = _helpers._empty_indices()
        return _helpers._materialize_or_return_indices(
            result=result,
            df=df,
            right=right,
            how=how,
            df_columns=df_columns,
            right_columns=right_columns,
            indicator=indicator,
            include_join_positions=include_join_positions,
            return_matching_indices=return_matching_indices,
            return_building_blocks=return_building_blocks,
        )
    (
        left_index,
        right_index,
        anchor_predicate,
        residual_predicates,
        anchor_dtype,
    ) = outcome
    left_positions = _helpers._convert_array_to_numpy(array=left_index._values)
    right_positions = _helpers._convert_array_to_numpy(array=right_index._values)
    predicates = [anchor_predicate, *residual_predicates]
    function = _get_multi_range_function(anchor_dtype)
    result = function(
        predicates=predicates,
        left_index=left_positions,
        right_index=right_positions,
        keep=keep,
    )
    result = _helpers._empty_indices() if result is None else result
    return _helpers._materialize_or_return_indices(
        result=result,
        df=df,
        right=right,
        how=how,
        df_columns=df_columns,
        right_columns=right_columns,
        indicator=indicator,
        include_join_positions=include_join_positions,
        return_matching_indices=return_matching_indices,
        return_building_blocks=return_building_blocks,
    )


def _aggregate_multiple_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    aggfunc: list[tuple],
    return_matched: bool,
    reverse: bool,
) -> pd.DataFrame:
    """Aggregate a range-first join with residual predicates.

    The predicate and aggregation arrays use the same compact aligned layout.
    The first predicate remains a three-field value/operator tuple; its
    physical position arrays are passed separately to Rust.

    The position arrays are physical positions in the original dataframes;
    they are not compact offsets. Aggregation arrays are aligned to the
    compact anchor layout, so Rust can use window offsets directly while the
    physical maps identify output rows.

    Args:
        df: Left dataframe and source for reverse aggregation.
        right: Right dataframe and source for forward aggregation.
        conditions: Range-first predicate list with residual predicates.
        aggfunc: ``(column_name, operation)`` aggregation requests.
        return_matched: Include the match mask in the output index.
        reverse: Aggregate left values into right output slots when true.

    Returns:
        A pandas aggregation dataframe indexed by the prepared output side.
    """
    aggregation_source = df if reverse else right
    outcome = _preparatory_work_multi_join(df=df, right=right, conditions=conditions)
    if outcome is None:
        return _empty_aggregation_result(
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    left_index, right_index, anchor_predicate, residual_predicates, _ = outcome
    left_positions = _helpers._convert_array_to_numpy(array=left_index._values)
    right_positions = _helpers._convert_array_to_numpy(array=right_index._values)
    source_indexer = left_index if reverse else right_index
    output_index = right_index if reverse else left_index

    # The range-first index kernel receives the anchor in its compact
    # predicate layout. Its aggregation arrays use the same aligned compact
    # layout, so Rust can index them directly with window offsets.
    predicates = [anchor_predicate, *residual_predicates]

    aggregation_inputs = _aggregation_inputs(
        source=aggregation_source,
        aggfunc=aggfunc,
        indexer=source_indexer,
    )
    function = _get_multi_range_aggregation_function(reverse)
    result = function(
        predicates=predicates,
        left_index=left_positions,
        right_index=right_positions,
        aggregations=aggregation_inputs,
        return_matched=return_matched,
    )
    if result is None:
        return _unmatched_aggregation_result(
            output_index=output_index,
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source_index=source_indexer,
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )
