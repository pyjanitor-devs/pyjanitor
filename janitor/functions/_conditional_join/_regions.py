"""Python preparation and dispatch for the Rust region kernels.

The regions algorithm is selected for joins with at least two inequality
predicates when ``join_algorithm="regions"``. PyJanitor owns null filtering,
right-value sorting, and physical-position preservation. Rust consumes either
two primary range predicates or three-or-more range predicates, depending on
the number of prepared range conditions. Additional non-range predicates are
passed through the dual- or n-region residual path.

The first range establishes the canonical compact layout. Its left and right
physical position arrays are also used to align aggregation inputs, so region
coordinates never index an unrelated dataframe layout. Later right ranges are
aligned to that layout before Rust sees them. The Rust index kernels return
fully materialized pairs; unlike the ordinary range kernels, they do not
expose ``starts``/``ends`` building blocks.

Preparation sequence:

1. Filter rows that are null for ordinary predicates, retaining their original
   physical row positions.
2. Collect the inequality predicates in their prepared order.
3. Sort every right range column in the direction required by its operator,
   carrying its physical positions.
4. Keep the first range's right order as the global layout. For every later
   range, build an aligner where ``aligner[global_position]`` identifies the
   source position in that range's sorted right values.
5. Send exactly two ranges to the dual-region kernel. Send three-or-more
   ranges to the n-region kernel, which checks all range paths together.
6. Keep non-range predicates as residuals for the dual- or n-region path.

The aggregation path uses the same preparation and selects the matching
dual-region or n-region Rust kernel. Aggregation arrays are passed in the
corresponding aligned source layout. It does not use range ``starts``/``ends``
building blocks or a right-ordering flag.
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
) -> (
    tuple[
        pd.Index,
        pd.Index,
        list[np.ndarray],
        list[tuple],
        list[tuple],
    ]
    | None
):
    """Prepare range predicates, right aligners, and residual predicates.

    Args:
        df: Reset-index left dataframe.
        right: Reset-index right dataframe.
        conditions: Complete conditional-join predicate list.

    Returns:
        ``(left_index, right_index, alignments, range_predicates,
        residual_predicates)``. ``range_predicates`` contains every
        inequality predicate in input order. ``alignments`` has one entry for
        every range after the first, so its length is
        ``len(range_predicates) - 1``. Returns ``None`` when either side is
        empty or null filtering leaves no usable rows.

        The returned index objects identify physical rows in the reset-index
        working dataframes. They are not sorted offsets. The first range's
        right values establish the global right layout. Each later range is
        independently sorted and its aligner maps a global position to the
        corresponding position in that sorted array.

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

    range_predicates = []
    alignments = []
    first_range_condition = False
    for condition in conditions:
        if condition.op not in _helpers.less_than_join_types.union(
            _helpers.greater_than_join_types
        ):
            continue
        left_column = _helpers._take_column(df, condition.left, left_index)
        right_column = _helpers._take_column(right, condition.right, right_index)
        if condition.op in _helpers.greater_than_join_types:
            ascending = False
        else:
            ascending = True
        right_column, _ = _helpers._sort_if_not_monotonic(
            series=right_column, ascending=ascending
        )
        if not first_range_condition:
            first_range_condition = True
            right_index = right_column.index
        else:
            _align = right_column.index.get_indexer(right_index)
            alignments.append(_align)
        range_predicates.append(
            (
                _helpers._convert_array_to_numpy(array=left_column._values),
                _helpers._convert_array_to_numpy(array=right_column._values),
                condition.op,
            )
        )

    residual_predicates = []
    for condition in conditions:
        if condition.op in _helpers.less_than_join_types.union(
            _helpers.greater_than_join_types
        ):
            continue
        residual_predicate = _helpers._build_residual_predicate(
            left=_helpers._take_column(df, condition.left, left_index),
            right=_helpers._take_column(right, condition.right, right_index),
            operation=condition.op,
        )
        residual_predicates.append(residual_predicate)

    return left_index, right_index, alignments, range_predicates, residual_predicates


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
        conditions: Complete conditional-join predicate list.
        keep: Match-selection policy passed to Rust.
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

    left_index, right_index, alignments, range_predicates, residual_predicates = outcome

    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index

    if (len(range_predicates) == 2) and not residual_predicates:
        first_predicate, second_predicate = range_predicates

        result = janitor_rs.dual_regions_indices(
            left_index=_helpers._convert_array_to_numpy(array=left_index._values),
            right_index=_helpers._convert_array_to_numpy(array=right_index._values),
            first_predicate=first_predicate,
            second_predicate=second_predicate,
            aligner=_helpers._convert_array_to_numpy(array=alignments[0]),
            keep=keep,
        )
    elif len(range_predicates) == 2:
        first_predicate, second_predicate = range_predicates

        result = janitor_rs.dual_regions_and_residual_indices(
            left_index=_helpers._convert_array_to_numpy(array=left_index._values),
            right_index=_helpers._convert_array_to_numpy(array=right_index._values),
            first_predicate=first_predicate,
            second_predicate=second_predicate,
            aligner=_helpers._convert_array_to_numpy(array=alignments[0]),
            residual_predicates=residual_predicates,
            keep=keep,
        )
    elif (len(range_predicates) > 2) and not residual_predicates:
        # The first range defines the global right layout. Every later range
        # is paired with the aligner produced for that same range, so keep
        # these two lists parallel when crossing the Rust boundary.
        first_predicate = range_predicates[0]
        remaining_predicates = range_predicates[1:]
        aligner_arrays = [
            _helpers._convert_array_to_numpy(array=aligner) for aligner in alignments
        ]
        result = janitor_rs.n_region_indices(
            left_index=_helpers._convert_array_to_numpy(array=left_index._values),
            right_index=_helpers._convert_array_to_numpy(array=right_index._values),
            first_predicate=first_predicate,
            range_predicates=remaining_predicates,
            aligners=aligner_arrays,
            keep=keep,
        )
    else:
        first_predicate = range_predicates[0]
        remaining_predicates = range_predicates[1:]
        aligner_arrays = [
            _helpers._convert_array_to_numpy(array=aligner) for aligner in alignments
        ]
        result = janitor_rs.n_region_and_residual_indices(
            left_index=_helpers._convert_array_to_numpy(array=left_index._values),
            right_index=_helpers._convert_array_to_numpy(array=right_index._values),
            first_predicate=first_predicate,
            range_predicates=remaining_predicates,
            aligners=aligner_arrays,
            residual_predicates=residual_predicates,
            keep=keep,
        )

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
    (
        left_index,
        right_index,
        alignments,
        anchor_predicates,
        residual_predicates,
    ) = outcome
    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    source_index = left_index if reverse else right_index
    output_index = right_index if reverse else left_index
    aggregation_inputs = _aggregation_inputs(
        source=aggregation_source,
        aggfunc=aggfunc,
        indexer=source_index,
    )
    left_index_array = _helpers._convert_array_to_numpy(array=left_index._values)
    right_index_array = _helpers._convert_array_to_numpy(array=right_index._values)
    first_predicate = anchor_predicates[0]

    if len(anchor_predicates) == 2:
        second_predicate = anchor_predicates[1]
        aligner = _helpers._convert_array_to_numpy(array=alignments[0])
        if residual_predicates:
            function = (
                janitor_rs.dual_regions_and_residual_aggregate_reverse
                if reverse
                else janitor_rs.dual_regions_and_residual_aggregate
            )
            result = function(
                left_index=left_index_array,
                right_index=right_index_array,
                first_predicate=first_predicate,
                second_predicate=second_predicate,
                aligner=aligner,
                residual_predicates=residual_predicates,
                aggregations=aggregation_inputs,
                return_matched=return_matched,
            )
        else:
            function = (
                janitor_rs.dual_regions_aggregate_reverse
                if reverse
                else janitor_rs.dual_regions_aggregate
            )
            result = function(
                left_index=left_index_array,
                right_index=right_index_array,
                first_predicate=first_predicate,
                second_predicate=second_predicate,
                aligner=aligner,
                aggregations=aggregation_inputs,
                return_matched=return_matched,
            )
    else:
        remaining_predicates = anchor_predicates[1:]
        aligner_arrays = [
            _helpers._convert_array_to_numpy(array=aligner) for aligner in alignments
        ]
        if residual_predicates:
            function = (
                janitor_rs.n_region_and_residual_aggregate_reverse
                if reverse
                else janitor_rs.n_region_and_residual_aggregate
            )
            result = function(
                left_index=left_index_array,
                right_index=right_index_array,
                first_predicate=first_predicate,
                range_predicates=remaining_predicates,
                aligners=aligner_arrays,
                residual_predicates=residual_predicates,
                aggregations=aggregation_inputs,
                return_matched=return_matched,
            )
        else:
            function = (
                janitor_rs.n_region_aggregate_reverse
                if reverse
                else janitor_rs.n_region_aggregate
            )
            result = function(
                left_index=left_index_array,
                right_index=right_index_array,
                first_predicate=first_predicate,
                range_predicates=remaining_predicates,
                aligners=aligner_arrays,
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
