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
    _materialize_aggregation_result,
    _unmatched_aggregation_result,
)


def _get_bounds_right(right_columns: pd.Index):
    """Build contiguous equality-key windows for the prepared right layout.

    ``right_columns`` must already be ordered so equal values occupy one
    contiguous block. Factorization assigns each distinct key a dense code;
    the returned ``starts`` and ``ends`` arrays map each key to its half-open
    block in ``right_columns``.

    Args:
        right_columns: Equality-key index for the prepared right rows. For
            multiple equality columns this is a ``MultiIndex``.

    Returns:
        A tuple ``(uniques, starts, ends)``. ``uniques`` contains one value
        per distinct key, while ``starts[i]:ends[i]`` selects that key's
        right-side rows.
    """
    positions, uniques = right_columns.factorize()
    counts = np.bincount(positions, minlength=len(uniques))
    starts = np.empty(counts.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = counts.cumsum()[:-1]
    ends = starts + counts
    return uniques, starts, ends


def _get_bounds_left(l_cols, uniques, starts, ends):
    """Align left equality keys with right-side equality windows.

    Left rows with no right key are removed from the compact layout. The
    returned ``starts`` and ``ends`` remain aligned with the surviving left
    rows; this is why filtering happens after indexing the window arrays.

    Args:
        l_cols: Equality keys for the prepared left rows.
        uniques: Distinct right equality keys returned by
            :func:`_get_bounds_right`.
        starts: Right-window starts indexed by ``uniques``.
        ends: Right-window ends indexed by ``uniques``.

    Returns:
        ``None`` when no left key matches. Otherwise returns either
        ``(None, starts, ends)`` when every left row matches or
        ``(booleans, starts, ends)`` when ``booleans`` identifies the
        surviving left rows.
    """
    indexers = uniques.get_indexer(l_cols)
    booleans = indexers != -1
    if not booleans.any():
        return None
    matched_starts = starts[indexers[booleans]]
    matched_ends = ends[indexers[booleans]]
    if booleans.all():
        return None, matched_starts, matched_ends
    return booleans, matched_starts, matched_ends


def _get_equi_indexer(
    df,
    right,
    left_index,
    right_index,
    equi_conditions,
    equi_positions,
    range_conditions,
):
    """Prepare the equality mapping and aligned range arrays.

    The right-side sorter is built in this level order:

    1. equality-key columns;
    2. selected range-key columns; and
    3. the original physical right index.

    A unique right equality key uses ``get_indexer``. Its matched right rows
    are projected with ``left_indexer`` so the left and right arrays become
    one-to-one; this path returns ``starts = ends = None``. Duplicate right
    equality keys use sorted contiguous windows and return ``starts``/``ends``
    for the surviving left rows.

    Args:
        df: Left dataframe in its current physical layout.
        right: Right dataframe in its current physical layout.
        left_index: Compact physical left index, or ``slice(None)`` when all
            left rows are eligible.
        right_index: Compact physical right index, or ``slice(None)`` when all
            right rows are eligible.
        equi_conditions: Equality predicates, in their original condition
            order.
        equi_positions: Positions of equality predicates in ``conditions``.
        range_conditions: At most two selected range predicates whose right
            values should be aligned to the prepared right layout.

    Returns:
        ``None`` when no left equality key matches. Otherwise returns
        ``(left_index, right_index, starts, ends, range_predicates,
        right_index_is_ordered)``. In the unique path, ``right_index`` is
        projected to the matched rows and ``starts``/``ends`` are ``None``.
    """
    # ELI5: build one table whose columns contain every value Rust will need.
    # The final column is the original right position, so sorting this table
    # never loses the mapping back to the caller's dataframe rows.
    sorter = []
    for condition in equi_conditions:
        series = right.loc[right_index, condition.right]._values
        sorter.append(series)
    if range_conditions:
        for condition in range_conditions:
            series = right.loc[right_index, condition.right]._values
            sorter.append(series)
    _right_index = right.index if isinstance(right_index, slice) else right_index
    sorter.append(_right_index._values)
    sorter = pd.MultiIndex.from_arrays(sorter)
    if len(equi_conditions) == 1:
        left_keys = df.loc[left_index, equi_conditions[0].left]._values
    else:
        left_keys = []
        for condition in equi_conditions:
            series = df.loc[left_index, condition.left]._values
            left_keys.append(series)
        left_keys = pd.MultiIndex.from_arrays(left_keys)
    uniqs = True
    try:
        # A successful get_indexer call proves that every right equality key
        # is unique. The returned positions point into the current sorter;
        # taking those positions makes the right arrays one-to-one with the
        # surviving left rows.
        if len(equi_positions) == 1:
            right_keys = sorter.get_level_values(0)
        else:
            n_equi = len(equi_conditions)
            right_keys = pd.MultiIndex(
                levels=sorter.levels[:n_equi],
                codes=sorter.codes[:n_equi],
                names=None,
                copy=False,
                verify_integrity=False,
            )
        left_indexer = right_keys.get_indexer(left_keys)
        booleans = left_indexer == -1
        if booleans.all():
            return None
        if booleans.any():
            booleans = ~booleans
            left_indexer = left_indexer[booleans]
            if isinstance(left_index, slice):
                left_index = df.index
            left_index = left_index[booleans]
        starts = None
        ends = None
    except pd.errors.InvalidIndexError:
        # Duplicate right keys cannot use get_indexer. Sort the complete row
        # record first, then derive equality windows from that same sorted
        # record so starts/ends and right_index share one coordinate system.
        uniqs = False
        if sorter.is_monotonic_decreasing:
            sorter = sorter[::-1]
        elif not sorter.is_monotonic_increasing:
            sorter = sorter.sort_values()
        if len(equi_positions) == 1:
            right_keys = sorter.get_level_values(0)
        else:
            n_equi = len(equi_conditions)
            right_keys = pd.MultiIndex(
                levels=sorter.levels[:n_equi],
                codes=sorter.codes[:n_equi],
                names=None,
                copy=False,
                verify_integrity=False,
            )
        uniques, starts, ends = _get_bounds_right(right_keys)
        outcome = _get_bounds_left(
            l_cols=left_keys, uniques=uniques, starts=starts, ends=ends
        )
        if outcome is None:
            return None
        booleans, starts, ends = outcome
        if booleans is not None:
            if isinstance(left_index, slice):
                left_index = df.index
            left_index = left_index[booleans]
    if uniqs:
        # Unique equality rows have no windows: every row's right candidate is
        # already selected by left_indexer.
        sorter = sorter.take(left_indexer)
    if range_conditions:
        range_arrays = []
        for level_number in range(len(equi_positions), sorter.nlevels):
            _index = sorter.get_level_values(level_number)
            range_arrays.append(_index)
        right_index = range_arrays[-1]
        left_arrays = [
            df.loc[left_index, condition.left] for condition in range_conditions
        ]
        range_ops = [condition.op for condition in range_conditions]
        range_predicates = zip(left_arrays, range_arrays[:-1], range_ops)
        range_predicates = [
            (
                _helpers._convert_array_to_numpy(array=left_array._values),
                _helpers._convert_array_to_numpy(array=right_array._values),
                operation,
            )
            for left_array, right_array, operation in range_predicates
        ]
    else:
        right_index = sorter.get_level_values(-1)
        range_predicates = []
    right_index_is_ordered = right_index.is_monotonic_increasing
    return (
        left_index,
        right_index,
        starts,
        ends,
        range_predicates,
        right_index_is_ordered,
    )


def _preparatory_work(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> (
    tuple[
        pd.Index | slice,
        pd.Index | slice,
        np.ndarray | None,
        np.ndarray | None,
        list[tuple],
        list[tuple],
        bool,
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
    2. The right dataframe's working rows are ordered lexicographically by
       equality keys, selected range keys, and their physical positions.
       Consequently, the first range key is monotonic within each equality
       group. ``right_index`` records the resulting physical right-row
       layout.
    3. A second range predicate is prepared as a candidate in that layout.
       Its usability relative to the first range predicate is decided by the
       downstream join kernel.
    4. Equality keys are built after any right-side layout change. Unique
       right keys use direct positions from ``get_indexer`` and project the
       right layout to aligned rows; duplicate right keys use contiguous
       windows from the sorted equality layout.
    5. Every remaining non-equality predicate is converted to a Rust residual
       tuple. Its right values are aligned to ``right_index`` so Rust can use
       one physical coordinate system for all predicates.

    The returned coordinates are physical positions, not public dataframe
    labels. Public labels are restored later by the caller or by the Rust
    aggregation materializer. ``right_index`` is always populated in a
    successful return; without range predicates, it is still the equality-
    sorted physical right-row layout.

    Args:
        df: Left dataframe in its current physical row layout.
        right: Right dataframe in its current physical row layout.
        conditions: Join predicates represented as
            ``(left_column, right_column, operator)`` tuples. At least one
            predicate must be an equality predicate.

    Returns:
        ``None`` when either side has no usable rows or when no left equality
        key matches a right equality key. Otherwise, a seven-element tuple is
        returned.

        * the physical left index shared by all prepared left arrays;
        * the physical right index shared by all prepared right arrays;
        * ``starts`` and ``ends``, containing right-row bounds for each
          matching left equality key, or both ``None`` when right equality
          keys are unique. In the unique case the right index and range
          arrays are already projected to one-to-one matching rows;
        * up to two aligned candidate range predicate tuples, represented as
          ``(left_values, right_values, operator)`` arrays;
        * residual predicate tuples for all remaining conditions; and
        * whether the prepared right index is monotonic increasing.
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
    equi_positions = [
        position
        for position, condition in enumerate(conditions)
        if condition.op == _helpers._JoinOperator.STRICTLY_EQUAL.value
    ]
    equi_conditions = [
        condition
        for condition in conditions
        if condition.op == _helpers._JoinOperator.STRICTLY_EQUAL.value
    ]
    range_positions = _helpers._get_range_positions_for_one_side(conditions=conditions)
    range_conditions = [conditions[position] for position in range_positions]
    outcome = _get_equi_indexer(
        df=df,
        right=right,
        left_index=left_index,
        right_index=right_index,
        equi_conditions=equi_conditions,
        equi_positions=equi_positions,
        range_conditions=range_conditions,
    )
    if outcome is None:
        return None
    (
        left_index,
        right_index,
        starts,
        ends,
        range_predicates,
        right_index_is_ordered,
    ) = outcome

    residual_predicates = []
    excluded = set(equi_positions).union(range_positions)
    for position, condition in enumerate(conditions):
        if position in excluded:
            continue
        residual_predicate = _helpers._build_residual_predicate(
            left=df.loc[left_index, condition.left],
            right=right.loc[right_index, condition.right],
            operation=condition.op,
        )
        residual_predicates.append(residual_predicate)
    return (
        left_index,
        right_index,
        starts,
        ends,
        range_predicates,
        residual_predicates,
        right_index_is_ordered,
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
        return_building_blocks: For pure-equi joins, return the prepared
            building-block dictionary with left_index, right_index, starts,
            and ends. It has no effect on predicate-filtered paths.

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
        starts,
        ends,
        range_predicates,
        residual_predicates,
        right_index_is_ordered,
    ) = outcome

    if not range_predicates and not residual_predicates and return_building_blocks:
        return {
            "left_index": left_index,
            "right_index": right_index,
            "starts": starts,
            "ends": ends,
        }
    if isinstance(left_index, slice):
        left_index = df.index
    if isinstance(right_index, slice):
        right_index = right.index
    left_index = _helpers._convert_array_to_numpy(
        array=left_index._values if hasattr(left_index, "_values") else left_index
    )
    right_index = _helpers._convert_array_to_numpy(
        array=right_index._values if hasattr(right_index, "_values") else right_index
    )
    if not range_predicates and not residual_predicates and (starts is not None):
        result = janitor_rs.equi_only_indices(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            right_index_is_ordered=right_index_is_ordered,
            keep=keep,
        )
    elif starts is None:
        predicates = [*range_predicates, *residual_predicates]
        result = janitor_rs.equi_uniq_residual_indices(
            left_index=left_index,
            right_index=right_index,
            residual_predicates=predicates,
        )
    elif not range_predicates and residual_predicates:
        result = janitor_rs.equi_ne_indices(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            residual_predicates=residual_predicates,
            keep=keep,
        )
    elif range_predicates and not residual_predicates:
        result = janitor_rs.equi_range_indices(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            range_predicates=range_predicates,
            keep=keep,
        )
    else:
        result = janitor_rs.equi_range_and_residual_indices(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            range_predicates=range_predicates,
            residual_predicates=residual_predicates,
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
        return _unmatched_aggregation_result(
            output_index=right.index if reverse else df.index,
            source=df if reverse else right,
            return_matched=return_matched,
            aggfunc=aggfunc,
        )
    (
        left_index,
        right_index,
        starts,
        ends,
        range_predicates,
        residual_predicates,
        _right_index_is_ordered,
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
    if reverse and starts is None:
        # Unique equality preparation aligns one right row with every left
        # row; reverse aggregation emits one result per distinct right row.
        output_index = pd.Index(pd.unique(right_index))
    result = janitor_rs.equi_aggregate(
        left_index=left_positions,
        right_index=index_right,
        starts=starts,
        ends=ends,
        range_predicates=range_predicates,
        residual_predicates=residual_predicates,
        aggregations=_aggregation_inputs(
            source=aggregation_source,
            aggfunc=aggfunc,
            indexer=source_index,
        ),
        return_matched=return_matched,
        reverse=reverse,
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
