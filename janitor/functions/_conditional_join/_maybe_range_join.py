"""Preparation and dispatch for dual-range conditional joins.

This module owns every path whose first two predicates are range predicates:

* exactly two range predicates use the basic Rust range kernel;
* two compatible range predicates followed by residual predicates use the
  range-extended Rust kernel;
* range-led aggregations use the corresponding range aggregation kernels.

The first two range predicates are called the *dual-range anchors*. PyJanitor
removes null rows from non-``!=`` columns, chooses an anchor pair, and places
the right-hand search values and their physical positions in a shared layout.
The second anchor is retained as a dual anchor only when it is compatible with
that layout. Rust then performs the binary searches, intersects the two
half-open windows, and evaluates any remaining predicates. This module never
sorts inside Rust and never treats a compatible second range predicate as an
ordinary residual.

Physical-layout invariant:
    ``left_index`` and ``right_index`` identify rows in the reset-index
    working dataframes. When the first right anchor is sorted, its index is
    reordered with the values. Every residual predicate and aggregation input
    is then selected using that same right layout. Rust receives value arrays
    separately from physical maps, so binary-search offsets cannot be
    mistaken for dataframe positions.

Dispatch invariant:
    A compatible second anchor uses the dual-range kernel. If the second
    right column cannot share the first anchor's ordered layout, it is passed
    as a residual to the range-first extended kernel. Aggregations follow the
    same split and additionally choose forward or reverse kernels according to
    which side supplies the source values.

The neighboring ``_single_non_equi_join_extended`` module handles a single
range anchor plus residual predicates and all-``!=`` candidate streams. Keeping
these responsibilities separate is important: a single-anchor residual path
may use an arbitrary later predicate, while this module may use two sorted
right arrays to build one intersected window.
"""

from __future__ import annotations

import janitor_rs
import pandas as pd

from janitor.functions._conditional_join import _aggregation_helpers, _helpers
from janitor.functions._conditional_join._single_range_predicate import (
    _get_multi_range_aggregation_function,
    _get_multi_range_function,
)

_DUAL_RANGE_FUNCTION = janitor_rs.range_join_indices
_DUAL_RANGE_EXTENDED_FUNCTION = janitor_rs.range_join_extended_indices
_DUAL_RANGE_AGGREGATE_FUNCTIONS = {
    False: janitor_rs.range_join_aggregate,
    True: janitor_rs.range_join_aggregate_reverse,
}
_DUAL_RANGE_EXTENDED_AGGREGATE_FUNCTIONS = {
    False: janitor_rs.range_join_extended_aggregate,
    True: janitor_rs.range_join_extended_aggregate_reverse,
}


def _get_dual_range_function() -> object:
    """Return the basic dual-range index kernel.

    The Rust function expects two three-field anchor tuples and shared
    physical ``left_index`` and ``right_index`` arrays. The right-position
    ordering flag is used only for ``first`` and ``last`` selection.
    """
    return _DUAL_RANGE_FUNCTION


def _get_dual_range_extended_function() -> object:
    """Return the dual-range index kernel that applies residual predicates.

    Rust intersects the two anchor windows, evaluates residual predicates for
    each candidate, and applies ``keep`` only after all predicates pass.
    """
    return _DUAL_RANGE_EXTENDED_FUNCTION


def _get_dual_range_aggregation_function(reverse: bool, extended: bool) -> object:
    """Select the dual-range aggregation kernel.

    Args:
        reverse: Aggregate left source values into right output slots when
            true; otherwise aggregate right values into left output slots.
        extended: Select the residual-aware kernel when true.

    Returns:
        The registered Rust PyO3 callable for the requested shape.
    """
    functions = (
        _DUAL_RANGE_EXTENDED_AGGREGATE_FUNCTIONS
        if extended
        else _DUAL_RANGE_AGGREGATE_FUNCTIONS
    )
    return functions[bool(reverse)]


def _preparatory_work(
    df: pd.DataFrame, right: pd.DataFrame, conditions: list[tuple[str, str, str]]
) -> tuple | None:
    """Prepare dual range anchors and residual predicates.

    Args:
        df: Left dataframe; filtered rows retain left-row order.
        right: Right dataframe; the first anchor is sorted in ascending value
            order and its physical positions travel with that sort.
        conditions: ``(left_column, right_column, operator)`` triples.

    Returns:
        ``(left_index, right_index, anchors, residuals,
        right_index_is_ordered, anchor_dtype)`` or ``None`` when either side
        has no usable non-null rows. Anchor tuples contain values and the
        operator only; shared physical maps are passed separately to Rust.

    Notes:
        If a second range predicate cannot use the first anchor's sorted right
        layout, it remains a residual and the range-first extended kernel is
        selected instead of the two-anchor kernel.

    Layout details:
        The first anchor's left layout remains in caller order. Its right
        values are sorted ascending and its physical positions travel with
        that sort. Residual arrays use the resulting left/right layouts. The
        returned ordering flag describes only the first right anchor and is
        consumed only by the non-extended dual-range index kernel.
    """

    if df.empty or right.empty:
        return None
    left_columns_and_ops = [(condition.left, condition.op) for condition in conditions]
    left_index = _helpers._get_indexer_for_non_null_rows(
        df=df, columns_and_ops=left_columns_and_ops
    )
    if left_index is None:
        return None
    right_columns_and_ops = [
        (condition.right, condition.op) for condition in conditions
    ]
    right_index = _helpers._get_indexer_for_non_null_rows(
        df=right, columns_and_ops=right_columns_and_ops
    )
    if right_index is None:
        return None

    # Select up to two range predicates, preferring one from each orientation
    # when both orientations are present.
    range_positions = []
    le_lt_count = 0
    ge_gt_count = 0

    for position, condition in enumerate(conditions):
        if le_lt_count and ge_gt_count:
            break
        if condition.op in _helpers.less_than_join_types and not le_lt_count:
            range_positions.append(position)
            le_lt_count += 1
        elif condition.op in _helpers.greater_than_join_types and not ge_gt_count:
            range_positions.append(position)
            ge_gt_count += 1

    if (le_lt_count + ge_gt_count) < 2:
        range_positions = []
        for position, condition in enumerate(conditions):
            if len(range_positions) == 2:
                break
            if condition.op in _helpers.less_than_join_types.union(
                _helpers.greater_than_join_types
            ):
                range_positions.append(position)

    first_anchor, second_anchor = (conditions[position] for position in range_positions)
    left_column, right_column, op = (
        first_anchor.left,
        first_anchor.right,
        first_anchor.op,
    )
    left_column = df.loc[left_index, left_column]
    right_column = right.loc[right_index, right_column]
    right_column, right_index_is_ordered = _helpers._sort_if_not_monotonic(
        series=right_column
    )
    if not right_index_is_ordered:
        right_index = right_column.index
    right_array = _helpers._convert_array_to_numpy(array=right_column._values)
    anchor_dtype = right_array.dtype
    first_anchor = (
        _helpers._convert_array_to_numpy(array=left_column._values),
        right_array,
        op,
    )
    # let's see if the second_anchor is monotonic
    second_left_column, second_right_column, second_op = (
        second_anchor.left,
        second_anchor.right,
        second_anchor.op,
    )
    second_left_column = df.loc[left_index, second_left_column]
    second_right_column = right.loc[right_index, second_right_column]
    second_anchor = (
        _helpers._convert_array_to_numpy(array=second_left_column._values),
        _helpers._convert_array_to_numpy(array=second_right_column._values),
        second_op,
    )
    primary_positions = set(range_positions)
    rest = [
        condition
        for position, condition in enumerate(conditions)
        if position not in primary_positions
    ]
    residual_predicates = []
    if second_right_column.is_monotonic_increasing:
        anchor_predicates = [first_anchor, second_anchor]
    else:
        residual_predicates.append(second_anchor)
        anchor_predicates = [first_anchor]

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
        anchor_predicates,
        residual_predicates,
        right_index_is_ordered,
        anchor_dtype,
    )


def _compute_multi_range_join(
    df,
    right,
    conditions,
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
    """Build dual-range physical index pairs through the Rust boundary.

    Args:
        df: Left dataframe.
        right: Right dataframe.
        conditions: Complete conditional-join predicate list.
        keep: ``"all"``, ``"any"``, ``"first"``, or ``"last"``.
        return_building_blocks: Return half-open anchor windows when true.
        how: Join shape used when materializing dataframe output.
        df_columns: Left columns to retain when materializing output.
        right_columns: Right columns to retain when materializing output.
        indicator: Whether to add the merge indicator column.
        include_join_positions: Whether to include physical pair positions in
            the materialized result index.
        return_matching_indices: Return physical index arrays instead of a
            materialized dataframe.

    Returns:
        A physical-index dictionary for index callers, or a materialized
        dataframe when ``return_matching_indices`` is false. Building-block
        requests always retain the dictionary form.
    """

    outcome = _preparatory_work(df=df, right=right, conditions=conditions)
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
        anchor_predicates,
        residual_predicates,
        right_index_is_ordered,
        anchor_dtype,
    ) = outcome
    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    left_positions = _helpers._convert_array_to_numpy(array=left_index._values)
    right_positions = _helpers._convert_array_to_numpy(array=right_index._values)

    if len(anchor_predicates) == 1:
        if return_building_blocks:
            keep = "all"
        predicates = [*anchor_predicates, *residual_predicates]
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

    predicates = [*anchor_predicates, *residual_predicates]
    if residual_predicates:
        # The extended Rust entry point evaluates residuals against the
        # intersected dual-anchor candidates before applying ``keep``.
        if return_building_blocks:
            keep = "all"
        function = _get_dual_range_extended_function()
        result = function(
            predicates=predicates,
            left_index=left_positions,
            right_index=right_positions,
            keep=keep,
        )
    else:
        # Both anchors use the same physical layouts. Their tuples carry only
        # value arrays and operators; the shared maps and ordering flag are
        # passed once at the Rust boundary.
        function = _get_dual_range_function()
        result = function(
            predicates=predicates,
            left_index=left_positions,
            right_index=right_positions,
            right_index_is_ordered=right_index_is_ordered,
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


def _aggregate(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    aggfunc: list[tuple],
    return_matched: bool,
    reverse: bool,
) -> pd.DataFrame:
    """Aggregate a dual-range join and materialize the pandas result.

    Args:
        df: Left dataframe and reverse-aggregation source.
        right: Right dataframe and forward-aggregation source.
        conditions: Complete conditional-join predicate list.
        aggfunc: ``(column_name, operation)`` aggregation requests.
        return_matched: Include the per-output match flag in the result index.
        reverse: Aggregate left values into right output slots when true;
            otherwise aggregate right values into left output slots.

    Returns:
        A pandas dataframe indexed by the output side. Aggregation arrays are
        aligned with the filtered/sorted anchor arrays; Rust returns physical
        output positions for final materialization.
    """

    aggregation_source = df if reverse else right
    outcome = _preparatory_work(df=df, right=right, conditions=conditions)
    if outcome is None:
        return _aggregation_helpers._empty_aggregation_result(
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    (
        left_index,
        right_index,
        anchor_predicates,
        residual_predicates,
        *_,
    ) = outcome
    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    source_indexer = left_index if reverse else right_index
    output_index = right_index if reverse else left_index
    left_positions = _helpers._convert_array_to_numpy(array=left_index._values)
    right_positions = _helpers._convert_array_to_numpy(array=right_index._values)

    aggregation_inputs = _aggregation_helpers._aggregation_inputs(
        source=aggregation_source,
        aggfunc=aggfunc,
        indexer=source_indexer,
    )

    if len(anchor_predicates) == 1:
        predicates = [anchor_predicates[0], *residual_predicates]
        function = _get_multi_range_aggregation_function(reverse)
        result = function(
            predicates=predicates,
            left_index=left_positions,
            right_index=right_positions,
            aggregations=aggregation_inputs,
            return_matched=return_matched,
        )
    else:
        first_left, first_right, first_operator = anchor_predicates[0]
        first_anchor = (
            first_left,
            first_right,
            first_operator,
        )
        second_left, second_right, second_operator = anchor_predicates[1]
        second_anchor = (
            second_left,
            second_right,
            second_operator,
        )
        predicates = [first_anchor, second_anchor, *residual_predicates]
        function = _get_dual_range_aggregation_function(
            reverse=reverse,
            extended=bool(residual_predicates),
        )
        result = function(
            predicates=predicates,
            left_index=left_positions,
            right_index=right_positions,
            aggregations=aggregation_inputs,
            return_matched=return_matched,
        )
    if result is None:
        return _aggregation_helpers._empty_aggregation_result(
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )

    return _aggregation_helpers._materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source_index=source_indexer,
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
    )
