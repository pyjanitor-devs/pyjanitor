"""Preparation and dispatch for conditional joins with an equality anchor.

This module owns the Python-side layout for joins that contain one or more
``==`` predicates. Equality keys are converted into either direct right-row
positions (when the right keys are unique) or dense key codes (when they are
duplicated). Optional range and residual predicates are prepared in the same
physical layout before the final index or aggregation operation.

The first compatible range predicate may reorder the right dataframe. The
resulting right-position array travels with every prepared predicate, so the
equality mapping, residual filters, and aggregation source all refer to the
same physical rows. Public pandas labels are restored only after dispatch.
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

_EQUI_BUILDING_BLOCKS_FUNCTION = janitor_rs.equi_join_building_blocks
_EQUI_FUNCTION = janitor_rs.equi_join_indices
_EQUI_FILTERED_FUNCTION = janitor_rs.equi_join_filtered_indices
_EQUI_AGGREGATE_FUNCTION = janitor_rs.equi_join_aggregate


def _build_equi_predicate(
    left_keys: pd.Index,
    right_keys: pd.Index,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Build the left-to-right equi-key mapping.

    get_indexer is used when the right keys are unique. When duplicate right
    keys make get_indexer invalid, the right keys are factorized. In that
    case, left_indexer contains dense right-key codes and right_codes maps
    every physical right position back to its code.

    Args:
        left_keys: Equi-key values for the left rows.
        right_keys: Equi-key values for the right rows in their current
            physical layout.

    Returns:
        A pair of arrays for unique or duplicate right keys, or None when no
        left key matches any right key. Unmatched left rows have -1 in
        left_indexer.
    """
    try:
        left_indexer = right_keys.get_indexer(left_keys)
        return left_indexer, None
    except pd.errors.InvalidIndexError:
        right_codes, uniques = right_keys.factorize(sort=False)
        left_indexer = uniques.get_indexer(left_keys)
        return left_indexer, right_codes


def _build_equi_keys(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: pd.Index | slice,
    right_index: pd.Index | slice,
    equi_conditions: list[tuple[str, str, str]],
) -> tuple[pd.Index, pd.Index]:
    """Build aligned single- or multi-column equi keys.

    Args:
        df: Full left working dataframe.
        right: Full right working dataframe.
        left_index: Physical left positions that survive ordinary null
            filtering.
        right_index: Physical right positions that survive ordinary null
            filtering and any right-side sort.
        equi_conditions: Equi predicates as column/operator tuples.

    Returns:
        A pair of aligned pandas indexes. Multiple equi columns are
        represented as MultiIndex values.
    """
    l_cols = []
    r_cols = []
    for condition in equi_conditions:
        l_cols.append(df.loc[left_index, condition.left]._values)
        r_cols.append(right.loc[right_index, condition.right]._values)
    if len(l_cols) > 1:
        l_cols = pd.MultiIndex.from_arrays(l_cols)
        r_cols = pd.MultiIndex.from_arrays(r_cols)
    else:
        l_cols = pd.Index(l_cols[0])
        r_cols = pd.Index(r_cols[0])
    return l_cols, r_cols


def _preparatory_work(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> (
    tuple[
        pd.Index | slice,
        pd.Index | slice,
        np.ndarray,
        np.ndarray | None,
        list[tuple],
        list[tuple],
    ]
    | None
):
    """Prepare the shared positional representation for an equi-led join.

    This function performs the Python-side work required before either index
    generation or fused Rust aggregation. It establishes one physical layout
    for all predicate arrays and returns that same representation to both
    callers. It does not materialize matching pairs or aggregation values.

    Preparation occurs in this order:

    1. Physical indexers for rows containing no nulls in ordinary comparison
       columns are computed using PyJanitor's existing null policy. ``!=``
       columns are excluded because their null semantics are represented
       explicitly in their residual predicate tuples. The working dataframes
       remain full-length; the indexers carry the compact layout.
    2. The first suitable range predicate sorts the right dataframe when
       necessary. ``right_index`` records the resulting physical right-row
       layout.
    3. A second range predicate is retained only when its right values are
       monotonic in the first predicate's physical layout. Otherwise it is
       evaluated later as a residual predicate.
    4. Equality keys are built after any right-side layout change. Unique
       right keys use direct positions from ``get_indexer``; duplicate right
       keys use dense factorization codes and a code-to-right-position array.
    5. Every remaining non-equality predicate is converted to a Rust residual
       tuple. Its right values are aligned to ``right_index`` so Rust can use
       one physical coordinate system for all predicates.

    The returned coordinates are physical positions, not public dataframe
    labels. Public labels are restored later by the caller or by the Rust
    aggregation materializer. ``right_index`` is always populated in a
    successful return; when no range reorders the right side, it is the
    current right dataframe index.

    Args:
        df: Left dataframe in its current physical row layout.
        right: Right dataframe in its current physical row layout.
        conditions: Join predicates represented as
            ``(left_column, right_column, operator)`` tuples. At least one
            predicate must be an equality predicate.

    Returns:
        ``None`` when either side has no usable rows. A six-element tuple is
        returned even when no equality key matches; in that case
        ``left_indexer`` contains only ``-1`` sentinels so aggregation can
        retain the prepared output domain as unmatched rows.

        * the physical left index shared by all prepared left arrays;
        * the physical right index shared by all prepared right arrays;
        * ``left_indexer``, containing direct right positions for unique keys
          or dense right-key codes for duplicate keys, with ``-1`` for
          unmatched left rows;
        * ``right_codes``, or ``None`` when right equality keys are unique;
        * up to two aligned range predicate tuples; and
        * residual predicate tuples for all remaining conditions.
    """
    left_columns_and_ops = [(condition.left, condition.op) for condition in conditions]
    left_index = _helpers._get_indexer_for_non_null_rows(
        df=df,
        columns_and_ops=left_columns_and_ops,
    )
    if left_index is None:
        return None
    right_columns_and_ops = [
        (condition.right, condition.op) for condition in conditions
    ]
    right_index = _helpers._get_indexer_for_non_null_rows(
        df=right,
        columns_and_ops=right_columns_and_ops,
    )
    if right_index is None:
        return None
    range_positions = []
    le_lt_count = 0
    ge_gt_count = 0
    # Retain positions because duplicate range predicates need distinct anchor
    # occurrences, and the chosen slots are removed from residuals below.
    for position, condition in enumerate(conditions):
        operator = condition.op
        if operator in _helpers.less_than_join_types and not le_lt_count:
            range_positions.append(position)
            le_lt_count += 1
        elif operator in _helpers.greater_than_join_types and not ge_gt_count:
            range_positions.append(position)
            ge_gt_count += 1
        if len(range_positions) == 2:
            break
    if len(range_positions) < 2:
        range_positions = []
        # Record original slots as well as conditions; this distinguishes
        # duplicate predicates when reconstructing the residual list.
        for position, condition in enumerate(conditions):
            if len(range_positions) == 2:
                break
            if condition.op in _helpers.less_than_join_types.union(
                _helpers.greater_than_join_types
            ):
                range_positions.append(position)
    range_maybe = [conditions[position] for position in range_positions]
    selected_range_positions = set(range_positions)
    # Exclude anchors by their original positions, including duplicate
    # occurrences, and preserve the order of all remaining residuals.
    rest = [
        condition
        for position, condition in enumerate(conditions)
        if position not in selected_range_positions
        and condition.op != _helpers._JoinOperator.STRICTLY_EQUAL.value
    ]
    # Select at most two range predicates. A second range is retained only
    # when its right values share the first range's physical permutation.
    range_predicates = []
    if range_maybe:
        left_column, right_column, op = (
            range_maybe[0].left,
            range_maybe[0].right,
            range_maybe[0].op,
        )
        right_, _ = _helpers._sort_if_not_monotonic(
            series=right.loc[right_index, right_column]
        )
        left_array = _helpers._convert_array_to_numpy(
            array=df.loc[left_index, left_column]._values
        )
        right_array = _helpers._convert_array_to_numpy(array=right_._values)
        range_predicate = (
            left_array,
            right_array,
            op,
        )
        range_predicates.append(range_predicate)
        if len(range_maybe) > 1:
            left_column, right_column, op = (
                range_maybe[1].left,
                range_maybe[1].right,
                range_maybe[1].op,
            )
            right_ = right.loc[right_.index, right_column]
            if right_.is_monotonic_increasing:
                left_array = _helpers._convert_array_to_numpy(
                    array=df.loc[left_index, left_column]._values
                )
                right_array = _helpers._convert_array_to_numpy(array=right_._values)
                range_predicate = (
                    left_array,
                    right_array,
                    op,
                )
                range_predicates.append(range_predicate)
            else:
                range_positions = range_positions[:1]
                selected_range_positions = set(range_positions)
                # Track the fallback anchor by position so a duplicate of the
                # same condition can remain a residual when appropriate.
                rest = [
                    condition
                    for position, condition in enumerate(conditions)
                    if position not in selected_range_positions
                    and condition.op != _helpers._JoinOperator.STRICTLY_EQUAL.value
                ]
        right_index = right_.index
    left_keys, right_keys = _build_equi_keys(
        df=df,
        right=right,
        left_index=left_index,
        right_index=right_index,
        equi_conditions=[
            condition
            for condition in conditions
            if condition.op == _helpers._JoinOperator.STRICTLY_EQUAL.value
        ],
    )
    equi_predicates = _build_equi_predicate(
        left_keys=left_keys,
        right_keys=right_keys,
    )

    if equi_predicates is None:
        return None

    left_indexer, right_codes = equi_predicates

    residual_predicates = []
    for condition in rest:
        residual_predicate = _helpers._build_residual_predicate(
            left=df[condition.left],
            right=right[condition.right],
            operation=condition.op,
            left_index=left_index,
            right_index=right_index,
        )
        residual_predicates.append(residual_predicate)
    return (
        left_index,
        right_index,
        left_indexer,
        right_codes,
        range_predicates,
        residual_predicates,
    )


def _compute_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
    keep: str,
    return_building_blocks: bool = False,
    *,
    how: str = "inner",
    df_columns=slice(None),
    right_columns=slice(None),
    indicator: bool | str = False,
    include_join_positions: bool = False,
    return_matching_indices: bool = True,
) -> dict[str, np.ndarray] | pd.DataFrame:
    """Prepare and dispatch an equi-led conditional join.

    The conditions must contain at least one equi predicate. Normal operator
    columns use PyJanitor's existing null policy before key mapping; != keeps
    its existing special handling.

    Range preparation happens before equi mapping. Therefore, right_index
    maps Rust's physical right positions back to public right labels.

    Args:
        df: Left dataframe.
        right: Right dataframe.
        conditions: Join conditions as
            (left_column, right_column, operator) tuples.
        keep: Match-selection mode for materialized Rust paths.
        return_building_blocks: For duplicate-right, pure-equi joins, return
            Rust's building-block dictionary with left_index, right_index,
            left_indexer, offsets, and positions. It has no effect on unique
            or predicate-filtered paths.

    Returns:
        A physical-index dictionary for index callers or a materialized
        dataframe when ``return_matching_indices`` is false. Building-block
        requests always retain the dictionary form.
    """
    outcome = _preparatory_work(df, right, conditions)
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
        left_indexer,
        right_codes,
        range_predicates,
        residual_predicates,
    ) = outcome

    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    left_index = _helpers._convert_array_to_numpy(array=left_index._values)
    right_index = _helpers._convert_array_to_numpy(array=right_index._values)

    # A unique right equi key already gives one direct right position per
    # left row. There is no duplicate metadata for Rust to build here.
    if right_codes is None and not range_predicates and not residual_predicates:
        booleans = left_indexer != -1
        if not booleans.all():
            left_indexer = left_indexer[booleans]
            left_index = left_index[booleans]
        right_index = right_index[left_indexer]
        result = {"left_index": left_index, "right_index": right_index}
    elif (
        right_codes is not None
        and not range_predicates
        and not residual_predicates
        and return_building_blocks
    ):
        result = _EQUI_BUILDING_BLOCKS_FUNCTION(
            left_index,
            right_index,
            left_indexer,
            right_codes,
        )
    elif right_codes is not None and not range_predicates and not residual_predicates:
        result = _EQUI_FUNCTION(
            left_index,
            right_index,
            left_indexer,
            right_codes,
            keep,
        )
    else:
        result = _EQUI_FILTERED_FUNCTION(
            left_index,
            right_index,
            left_indexer,
            right_codes,
            range_predicates,
            residual_predicates,
            keep,
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
    conditions: list[tuple[str, str, str]],
    aggfunc: list[tuple],
    reverse: bool,
    return_matched: bool,
) -> pd.DataFrame:
    """Aggregate an equi-led conditional join in the Rust fused kernel.

    The Python side owns the physical layout. It computes non-null indexers
    for ordinary comparison operators, optionally sorts the right side for
    one or two compatible range anchors, builds the equi indexer, and sends
    the resulting positional representation to Rust. Rust then traverses
    equi candidates, applies range windows and residual predicates, and
    updates the aggregation state without materializing matching pairs.

    Args:
        df: Left dataframe in reset physical-row order.
        right: Right dataframe in reset physical-row order.
        conditions: Join predicates containing at least one equality. At most
            two range predicates are sent as windows; all remaining
            predicates are residual filters.
        aggfunc: Non-empty ``(column_name, operation)`` aggregation requests.
        reverse: Aggregate left values into right output rows when true;
            otherwise aggregate right values into left output rows.
        return_matched: Include the per-output matched mask in the result
            index when true.

    Returns:
        A dataframe using the shared conditional-join aggregation contract.
        Eligible output rows are retained with identity values when no
        candidate survives; only a missing eligible output domain is empty.
    """

    outcome = _preparatory_work(df, right, conditions)
    if outcome is None:
        return _empty_aggregation_result(
            source=df if reverse else right,
            return_matched=return_matched,
            aggfunc=aggfunc,
        )
    (
        left_index,
        right_index,
        left_indexer,
        right_codes,
        range_predicates,
        residual_predicates,
    ) = outcome

    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    left_positions = _helpers._convert_array_to_numpy(array=left_index._values)
    index_right = _helpers._convert_array_to_numpy(array=right_index._values)

    aggregation_source = df if reverse else right
    source_index = left_index if reverse else right_index
    output_index = right_index if reverse else left_index
    result = _EQUI_AGGREGATE_FUNCTION(
        left_positions,
        index_right,
        left_indexer,
        right_codes,
        range_predicates,
        residual_predicates,
        _aggregation_inputs(
            source=aggregation_source,
            aggfunc=aggfunc,
            indexer=source_index,
        ),
        return_matched,
        reverse,
    )
    if result is None:
        return _unmatched_aggregation_result(
            output_index=output_index,
            source=aggregation_source,
            return_matched=return_matched,
            aggfunc=aggfunc,
        )
    return _materialize_aggregation_result(
        result=result,
        output_index=output_index,
        source=aggregation_source,
        aggfunc=aggfunc,
        return_matched=return_matched,
        source_index=source_index,
    )
