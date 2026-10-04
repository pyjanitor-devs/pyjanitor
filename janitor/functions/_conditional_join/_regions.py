"""Python preparation and dispatch for the Rust region kernels.

The regions algorithm is selected for joins with at least two inequality
predicates when ``join_algorithm="regions"``. PyJanitor owns null filtering,
right-value sorting, and physical-position preservation. Rust then consumes
two five-field primary anchors, aligns them by their physical positions, and
performs the monotonic region sweep. Additional predicates remain residuals.

The first anchor establishes the canonical compact layout. Its left and right
physical position arrays are also used to align aggregation inputs, so region
coordinates never index an unrelated dataframe layout. The Rust index kernels
return fully materialized pairs; unlike the ordinary range kernels, they do
not expose ``starts``/``ends`` building blocks.

Preparation sequence:

1. Filter rows that are null for ordinary predicates, retaining their original
   physical row positions.
2. Select two inequality predicates as region anchors, preferring one
   less-than-family and one greater-than-family predicate when available.
3. Sort each anchor's right values while carrying its physical positions.
4. Let Rust align the independently prepared anchors by their unique physical
   positions and construct the region labels.
5. Keep every remaining predicate as a residual evaluated only after both
   region anchors pass.

The aggregation path uses the same preparation. It adds output maps to the
first anchor, selects forward or reverse Rust kernels, and passes aggregation
arrays in the corresponding aligned source layout. It does not use range
``starts``/``ends`` building blocks or a right-ordering flag.
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


def _preparatory_work(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple[pd.Index, pd.Index, list[tuple], list[tuple]] | None:
    """Prepare the two region anchors and residual predicates.

    Args:
        df: Reset-index left dataframe.
        right: Reset-index right dataframe.
        conditions: Complete conditional-join predicate list.

    Returns:
        ``(left_index, right_index, anchor_predicates,
        residual_predicates)``. The anchor list always contains the two
        region predicates; later predicates are residuals aligned to the
        first anchor. Returns ``None`` when either side is empty or null
        filtering leaves no usable rows.

        The returned index objects identify physical rows in the reset-index
        working dataframes. They are not sorted offsets. The first anchor's
        right values are sorted together with its physical positions; the
        second anchor is independently sorted and then reindexed to the first
        anchor's physical layouts.

    Raises:
        ValueError: If fewer than two inequality predicates are available.
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

    # grab possible range join combo
    range_positions = []
    le_lt_count = 0
    ge_gt_count = 0

    # Store condition positions along with the selected anchors. Positions
    # distinguish duplicate predicates and let residual construction exclude
    # exactly the selected occurrences later.
    for position, condition in enumerate(conditions):
        if le_lt_count and ge_gt_count:
            break
        if (condition.op in _helpers.less_than_join_types) and not le_lt_count:
            range_positions.append(position)
            le_lt_count += 1
        elif (condition.op in _helpers.greater_than_join_types) and not ge_gt_count:
            range_positions.append(position)
            ge_gt_count += 1

    if (le_lt_count + ge_gt_count) < 2:
        range_positions = []
        # The fallback still needs original positions so duplicate predicates
        # are selected and removed by occurrence, without reordering anything.
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
    first_anchor = (
        _helpers._convert_array_to_numpy(array=left_column._values),
        _helpers._convert_array_to_numpy(array=left_column.index._values),
        _helpers._convert_array_to_numpy(array=right_column._values),
        _helpers._convert_array_to_numpy(array=right_column.index._values),
        op,
    )
    second_left_column, second_right_column, second_op = (
        second_anchor.left,
        second_anchor.right,
        second_anchor.op,
    )
    second_left_column = df.loc[left_index, second_left_column]
    second_right_column = right.loc[right_index, second_right_column]
    second_right_column, _ = _helpers._sort_if_not_monotonic(series=second_right_column)
    second_anchor = (
        _helpers._convert_array_to_numpy(array=second_left_column._values),
        _helpers._convert_array_to_numpy(array=second_left_column.index._values),
        _helpers._convert_array_to_numpy(array=second_right_column._values),
        _helpers._convert_array_to_numpy(array=second_right_column.index._values),
        second_op,
    )

    anchor_predicates = [first_anchor, second_anchor]
    primary_positions = set(range_positions)
    # Filter by original condition position: duplicate anchor occurrences have
    # already been handled by the regions kernel and must not run as residuals.
    rest = [
        condition
        for position, condition in enumerate(conditions)
        if position not in primary_positions
    ]
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

    return left_index, right_index, anchor_predicates, residual_predicates


def _compute_regions_join(
    *,
    df,
    right,
    conditions,
    keep,
    how,
    df_columns,
    right_columns,
    indicator,
    include_join_positions,
    return_matching_indices,
    return_building_blocks,
):
    """Normalize Rust pairs and choose the public return representation.

    Rust returns ``None`` for no matches and otherwise returns a dictionary of
    physical position arrays. Index callers receive that dictionary unchanged;
    ordinary dataframe callers pass it through the shared materializer, which
    applies join shape, column selection, indicators, and join-position index
    options.

    Args:
        df: Reset-index left dataframe.
        right: Reset-index right dataframe.
        how: Requested join shape.
        df_columns: Left columns to retain.
        right_columns: Right columns to retain.
        indicator: Indicator-column setting.
        include_join_positions: Whether to include physical pair positions.
        return_matching_indices: Return physical arrays instead of a frame.
        return_building_blocks: Preserve dictionary output for experimental
            building-block callers.
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

    *_, anchor_predicates, residual_predicates = outcome
    predicates = [*anchor_predicates, *residual_predicates]
    function = (
        janitor_rs.region_indices
        if not residual_predicates
        else janitor_rs.region_indices_extended
    )
    result = function(predicates=predicates, keep=keep)
    if result is None:
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


def _aggregation_predicates(predicates: list[tuple]) -> list[tuple]:
    """Attach output maps required by the region aggregation ABI.

    The index kernels use five-field anchors. Aggregation additionally needs
    the output labels for the compact first-anchor layout: forward results are
    labeled by first-anchor left positions and reverse results by right
    positions. These maps label outputs only; Rust still uses its internal
    compact-to-source mappings for reading aggregation values.

    The first four fields remain the normal region anchor fields. The
    aggregation form inserts a boolean extension marker, then the left and
    right output maps, and finally the operator. Residual tuples are preserved
    unchanged after the augmented first anchor. The maps are copied from the
    first anchor because that anchor defines the canonical output layout.
    """
    first = predicates[0]
    left_output = np.asarray(first[1], dtype=np.int64)
    right_output = np.asarray(first[3], dtype=np.int64)
    return [(*first[:4], True, left_output, right_output, first[4]), *predicates[1:]]


def _aggregate(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple],
    aggfunc: list[tuple],
    return_matched: bool,
    reverse: bool,
) -> pd.DataFrame:
    """Compute region aggregation without materializing candidate pairs.

    Forward aggregation consumes right values and returns one output slot per
    first-anchor left position. Reverse aggregation consumes left values and
    returns one slot per first-anchor right position. Aggregation arrays are
    aligned with the same compact layouts passed to Rust.

    Args:
        df: Reset-index left dataframe and reverse aggregation source.
        right: Reset-index right dataframe and forward aggregation source.
        conditions: At least two inequality anchors followed by optional
            residual predicates.
        aggfunc: ``(column_name, operation)`` requests.
        return_matched: Include one match flag per output slot.
        reverse: Aggregate left values into right output slots when true.

    Returns:
        A dataframe with one row per first-anchor output position, including
        unmatched slots. If no candidate survives, every eligible output slot
        remains with identity values.

    Direction contract:
        Forward aggregation reads values from ``right`` and emits one output
        slot per first-anchor left position. Reverse aggregation reads values
        from ``df`` and emits one slot per first-anchor right position. The
        source indexer and output index are selected together so aggregation
        arrays, Rust output positions, and the final pandas index share one
        physical layout.
    """
    aggregation_source = df if reverse else right
    outcome = _preparatory_work(df=df, right=right, conditions=conditions)
    if outcome is None:
        return _empty_aggregation_result(
            source=aggregation_source,
            aggfunc=aggfunc,
            return_matched=return_matched,
        )
    left_index, right_index, anchor_predicates, residual_predicates = outcome
    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    predicates = _aggregation_predicates([*anchor_predicates, *residual_predicates])
    source_index = left_index if reverse else right_index
    output_index = right_index if reverse else left_index
    aggregation_inputs = _aggregation_inputs(
        source=aggregation_source,
        aggfunc=aggfunc,
        indexer=source_index,
    )
    extended = len(predicates) > 2
    function = {
        (False, False): janitor_rs.region_aggregate,
        (True, False): janitor_rs.region_aggregate_reverse,
        (False, True): janitor_rs.region_extended_aggregate,
        (True, True): janitor_rs.region_extended_aggregate_reverse,
    }[(reverse, extended)]
    result = function(
        predicates=predicates,
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
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
        source_index=source_index,
    )
