"""Preparation and dispatch for dual-range conditional joins.

This module owns every path whose first two predicates are range predicates:

* exactly two range predicates use the basic Rust range kernel;
* two compatible range predicates followed by residual predicates use the
  range-extended Rust kernel;
* range-led aggregations use the corresponding range aggregation kernels.

The first two range predicates are called the *dual-range anchors*. PyJanitor
removes null rows from non-``!=`` columns, chooses an anchor pair, and places
the right-hand search values and their physical positions in a shared layout.
The second anchor is retained as a dual anchor when it is compatible with that
layout. For opposing interval predicates, a non-monotonic second right array
is replaced for window construction by a cumulative envelope; the original
predicate remains as an exact residual for Rust to recheck. Rust then performs
the binary searches, intersects the two half-open windows, and evaluates any
remaining predicates. This module never sorts inside Rust.

Physical-layout invariant:
    ``left_index`` and ``right_index`` identify rows in the reset-index
    working dataframes. When the first right anchor is sorted, its index is
    reordered with the values. Every residual predicate and aggregation input
    is then selected using that same right layout. Rust receives value arrays
    separately from physical maps, so binary-search offsets cannot be
    mistaken for dataframe positions.

Dispatch invariant:
    A compatible second anchor uses the dual-range kernel. If an opposing
    second right column is non-monotonic in the first anchor's layout, a
    cumulative envelope supplies a monotonic superset window and the original
    predicate is rechecked by the range-extended Rust kernel. Aggregations
    follow the same split and additionally choose forward or reverse kernels
    according to which side supplies the source values.

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
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[_helpers.JoinCondition],
) -> tuple | None:
    """Prepare dual range anchors and residual predicates.

    Args:
        df: Left dataframe; filtered rows retain left-row order.
        right: Right dataframe; the first anchor is sorted in ascending value
            order and its physical positions travel with that sort.
        conditions: Validated, normalized :class:`JoinCondition` objects.
            Public condition tuples are converted before this internal
            preparation boundary is reached.

    Returns:
        ``(left_index, right_index, anchors, residuals,
        right_index_is_ordered, anchor_dtype)`` or ``None`` when either side
        has no usable non-null rows. Anchor tuples contain values and the
        operator only; shared physical maps are passed separately to Rust.

    Notes:
        If an opposing second range predicate cannot use the first anchor's
        sorted right layout directly, its cumulative envelope is used for
        candidate generation and the original predicate is retained for exact
        residual filtering. Other incompatible predicates remain residuals.

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

    # Every array below is indexed through these two compact maps. Null rows
    # have already been removed, so the maps may be shorter than the source
    # dataframes; retaining them is what lets Rust return original dataframe
    # positions after it searches the compact arrays.

    # For opposing interval predicates, make the lower-bound predicate the
    # primary anchor. Its sorted right array gives a compact prefix window;
    # the upper-bound predicate can then use a cumulative envelope in that
    # same layout, followed by an exact residual recheck.
    # A lower-bound search (`right.start <= left.end`) produces a prefix. An
    # upper-bound search (`left.start <= right.end`) can then be represented
    # by an envelope in that same right-side ordering. Choosing the predicates
    # in the opposite order loses this compact-window opportunity whenever
    # the right endpoints are not monotonic after sorting by right.start.
    # Maintainer note: this selects the layout that supports the cumulative
    # envelope optimization; it is not a selectivity estimator. On skewed
    # data, the selected anchor can admit substantially more candidates than
    # another valid range predicate, causing a large performance and memory
    # penalty before residual predicates are applied. The previous
    # implementation used a sampling heuristic for this choice, but sampling
    # can mis-rank predicates and is intentionally not restored here. Revisit
    # this only with a representative user workload or a deterministic cost
    # estimate that is cheaper than constructing the join candidates.
    range_positions = [
        position
        for position, condition in enumerate(conditions)
        if condition.op in _helpers.greater_than_join_types
    ][:1]
    range_positions.extend(
        position
        for position, condition in enumerate(conditions)
        if condition.op in _helpers.less_than_join_types
    )
    range_positions = range_positions[:2]

    if len(range_positions) < 2:
        range_positions = []
        # ``enumerate`` is needed because the fallback also records which
        # original condition slots became anchors; this remains unambiguous
        # when duplicate predicates are present.
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
        # Sorting changes search offsets, not row identity. Carry the sorted
        # index positions forward so every second anchor, residual predicate,
        # and aggregation source uses exactly the same physical layout.
        right_index = right_column.index
    right_array = _helpers._convert_array_to_numpy(array=right_column._values)
    anchor_dtype = right_array.dtype
    first_anchor = (
        _helpers._convert_array_to_numpy(array=left_column._values),
        right_array,
        op,
    )
    # The second right array is aligned to the primary anchor's sorted layout.
    # It may therefore be non-monotonic even when the first right array is.
    second_left_column, second_right_column, second_op = (
        second_anchor.left,
        second_anchor.right,
        second_anchor.op,
    )
    second_left_column = df.loc[left_index, second_left_column]
    second_right_column = right.loc[right_index, second_right_column]
    second_left_array = _helpers._convert_array_to_numpy(
        array=second_left_column._values
    )
    second_right_array = _helpers._convert_array_to_numpy(
        array=second_right_column._values
    )
    primary_positions = set(range_positions)
    # Compare original positions so duplicate anchor predicates are excluded
    # by their individual slots, while all remaining predicates stay ordered.
    rest = [
        condition
        for position, condition in enumerate(conditions)
        if position not in primary_positions
    ]
    residual_predicates = []
    if second_right_column.is_monotonic_increasing:
        # Both real predicates are monotonic in the shared sorted layout, so
        # Rust can intersect their two binary-search windows directly.
        anchor_predicates = [
            first_anchor,
            (second_left_array, second_right_array, second_op),
        ]
    else:
        cumulative_right = None
        if (
            first_anchor[2] in _helpers.greater_than_join_types
            and second_op in _helpers.less_than_join_types
        ):
            # For each position, cummax records the largest endpoint seen in
            # the prefix. It may widen a candidate window, but never removes
            # a genuinely matching row; the original endpoint comparison is
            # rechecked below before a result is emitted.
            #
            # Example, after sorting by the first right column:
            #
            #     second_right = [8, 2, 5, 3]
            #     cummax       = [8, 8, 8, 8]
            #
            # For ``left.start == 4``, the envelope admits all four positions
            # because ``4 <= [8, 8, 8, 8]``. The original values only admit
            # positions 0 and 2 (``4 <= [8, 2, 5, 3]``), so Rust's residual
            # filter removes positions 1 and 3.
            cumulative_right = second_right_column.cummax()
            cumulative_right = _helpers._convert_array_to_numpy(
                array=cumulative_right._values
            )
        elif (
            first_anchor[2] in _helpers.less_than_join_types
            and second_op in _helpers.greater_than_join_types
        ):
            # This is the mirrored layout: a reverse cumulative minimum keeps
            # the suffix boundary monotonic while remaining a safe superset.
            #
            # Example, with the second right column aligned to the same
            # primary layout:
            #
            #     second_right = [2, 8, 1, 5]
            #     reverse_min  = [1, 1, 1, 5]
            #
            # For ``left.end == 4``, the envelope admits positions 0, 1, and
            # 2 because ``4 >= [1, 1, 1, 5]``. The original values admit only
            # positions 0 and 2 (``4 >= [2, 8, 1, 5]``); Rust removes position
            # 1 during the exact residual check.
            cumulative_right = second_right_column.iloc[::-1].cummin().iloc[::-1]
            cumulative_right = _helpers._convert_array_to_numpy(
                array=cumulative_right._values
            )

        if cumulative_right is None:
            # Same-direction range predicates do not have a generally safe
            # opposing envelope. Keep the second predicate as a residual and
            # let the single-anchor extended kernel evaluate it exactly.
            residual_predicates.append(
                (second_left_array, second_right_array, second_op)
            )
            anchor_predicates = [first_anchor]
        else:
            # The cumulative envelope supplies a monotonic superset window;
            # retain the original predicate for an exact Rust recheck. This
            # two-stage contract is essential: the envelope is for speed, not
            # for changing the join's truth table.
            anchor_predicates = [
                first_anchor,
                (second_left_array, cumulative_right, second_op),
            ]
            residual_predicates.append(
                _helpers._build_residual_predicate(
                    left=second_left_column,
                    right=second_right_column,
                    operation=second_op,
                    left_index=left_index,
                    right_index=right_index,
                )
            )

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
        # There is no safe second monotonic window. The range-first extended
        # endpoint still avoids a Cartesian product by filtering candidates
        # inside the first anchor's bounded window.
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
        # intersected dual-anchor candidates before applying ``keep``. In the
        # cumulative-envelope case, this is where the original second range
        # predicate is restored exactly.
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
        # passed once at the Rust boundary. Since no residual can reject a
        # candidate, the cheaper basic dual-range endpoint is sufficient.
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
    # The source and output sides intentionally swap for reverse aggregation.
    # These indexers refer to original dataframe positions, while the anchor
    # arrays may be compact and sorted; mixing the two would aggregate values
    # into the wrong rows without necessarily raising an error.
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
        return _aggregation_helpers._unmatched_aggregation_result(
            output_index=output_index,
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
