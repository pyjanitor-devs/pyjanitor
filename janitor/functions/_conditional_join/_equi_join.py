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


def _sort_right_rows(key_codes, range_values, right_index):
    """Order right rows by equality-key codes, then raw range values.

    Equality keys only need to be contiguous, so they sort on their codes.
    Range values are already NumPy and orderable, so they sort as is. The
    lexsort is stable, so the right labels join the sort keys only when
    physical order is not already label order.

    Args:
        key_codes: One integer code array per equality key, primary first.
        range_values: Right range arrays in the selected predicate order.
        right_index: Physical right labels for the eligible rows.

    Returns:
        Positions that sort the eligible right rows.
    """
    sort_keys = [*key_codes, *range_values]
    if not right_index.is_monotonic_increasing:
        sort_keys.append(_helpers._convert_array_to_numpy(array=right_index._values))
    return np.lexsort(sort_keys[::-1])


def _drop_unmatched_left(df, left_index, left_positions):
    """Remove left rows whose equality key has no right match.

    Args:
        df: Left dataframe in its current physical layout.
        left_index: Compact physical left index, or ``slice(None)``.
        left_positions: Per-left-row lookup result; ``-1`` marks no match.

    Returns:
        ``None`` when nothing matches, else ``(left_index, left_positions)``
        restricted to the matched left rows.
    """
    booleans = left_positions != -1
    if not booleans.any():
        return None
    if not booleans.all():
        left_positions = left_positions[booleans]
        if isinstance(left_index, slice):
            left_index = df.index
        left_index = left_index[booleans]
    return left_index, left_positions


def _single_key_layout(df, right, left_index, right_index, condition, range_values):
    """Order the right rows for one equality key.

    Only the key is factorized. ``sort=False`` codes are assigned by hashing,
    so object keys with mixed types never need comparing.

    Args:
        df: Left dataframe in its current physical layout.
        right: Right dataframe in its current physical layout.
        left_index: Compact physical left index, or ``slice(None)``.
        right_index: Compact physical right index, or ``slice(None)``.
        condition: The equality predicate.
        range_values: Right range arrays, converted to NumPy.

    Returns:
        ``None`` when no left key matches, else
        ``(left_index, order, starts, ends)``. ``order`` selects the prepared
        right rows; ``starts``/``ends`` are ``None`` for unique right keys.
    """
    codes, uniques = pd.factorize(
        right.loc[right_index, condition.right]._values, sort=False
    )
    left_codes = pd.Index(uniques).get_indexer(
        df.loc[left_index, condition.left]._values
    )
    outcome = _drop_unmatched_left(df, left_index, left_codes)
    if outcome is None:
        return None
    left_index, left_codes = outcome
    if len(uniques) == len(codes):
        # Unique right keys: sort=False codes are right positions, so the
        # left codes select each left row's single right candidate.
        return left_index, left_codes, None, None
    _right_index = right.index if isinstance(right_index, slice) else right_index
    order = _sort_right_rows([codes], range_values, _right_index)
    counts = np.bincount(codes, minlength=len(uniques))
    ends = counts.cumsum()
    starts = ends - counts
    return left_index, order, starts[left_codes], ends[left_codes]


def _multi_key_layout(df, right, left_index, right_index, conditions, range_values):
    """Order the right rows for several equality keys.

    The MultiIndex holds only the equality keys; its level codes drive the
    sort, and the range columns stay raw.

    Args:
        df: Left dataframe in its current physical layout.
        right: Right dataframe in its current physical layout.
        left_index: Compact physical left index, or ``slice(None)``.
        right_index: Compact physical right index, or ``slice(None)``.
        conditions: The equality predicates, in their original order.
        range_values: Right range arrays, converted to NumPy.

    Returns:
        Same as :func:`_single_key_layout`.
    """
    right_keys = []
    left_keys = []
    for condition in conditions:
        right_keys.append(right.loc[right_index, condition.right]._values)
        left_keys.append(df.loc[left_index, condition.left]._values)
    right_keys = pd.MultiIndex.from_arrays(right_keys)
    left_keys = pd.MultiIndex.from_arrays(left_keys)
    if right_keys.is_unique:
        outcome = _drop_unmatched_left(
            df, left_index, right_keys.get_indexer(left_keys)
        )
        if outcome is None:
            return None
        left_index, left_indexer = outcome
        return left_index, left_indexer, None, None
    _right_index = right.index if isinstance(right_index, slice) else right_index
    order = _sort_right_rows(right_keys.codes, range_values, _right_index)
    # A new key block starts wherever any sorted level code changes.
    block_start = np.zeros(len(order), dtype=bool)
    block_start[0] = True
    for level_codes in right_keys.codes:
        sorted_codes = level_codes[order]
        block_start[1:] |= sorted_codes[1:] != sorted_codes[:-1]
    starts = np.flatnonzero(block_start)
    ends = np.append(starts[1:], len(order))
    # One right row per block names that block's key; those keys are unique.
    block_keys = right_keys.take(order[starts])
    outcome = _drop_unmatched_left(df, left_index, block_keys.get_indexer(left_keys))
    if outcome is None:
        return None
    left_index, left_blocks = outcome
    return left_index, order, starts[left_blocks], ends[left_blocks]


def _get_equi_indexer(
    df,
    right,
    left_index,
    right_index,
    equi_conditions,
    range_conditions,
):
    """Prepare the equality mapping and aligned range arrays.

    The right rows are ordered by equality key, then by the selected range
    columns, then by physical right label. Only the equality keys are
    factorized: the range columns are converted to NumPy once and sorted as
    raw values, so the same arrays feed both the sort and the Rust kernel.

    Unique right equality keys select one right row per left row; this path
    returns ``starts = ends = None``. Duplicate right equality keys use sorted
    contiguous windows and return ``starts``/``ends`` for the surviving left
    rows.

    Args:
        df: Left dataframe in its current physical layout.
        right: Right dataframe in its current physical layout.
        left_index: Compact physical left index, or ``slice(None)`` when all
            left rows are eligible.
        right_index: Compact physical right index, or ``slice(None)`` when all
            right rows are eligible.
        equi_conditions: Equality predicates, in their original condition
            order.
        range_conditions: At most two selected range predicates whose right
            values should be aligned to the prepared right layout.

    Returns:
        ``None`` when no left equality key matches. Otherwise returns
        ``(left_index, right_index, starts, ends, range_predicates,
        right_index_is_ordered)``. In the unique path, ``right_index`` is
        projected to the matched rows and ``starts``/``ends`` are ``None``.
    """
    range_values = []
    for condition in range_conditions:
        values = right.loc[right_index, condition.right]._values
        range_values.append(_helpers._convert_array_to_numpy(array=values))
    layout = _single_key_layout if len(equi_conditions) == 1 else _multi_key_layout
    outcome = layout(
        df,
        right,
        left_index,
        right_index,
        equi_conditions[0] if len(equi_conditions) == 1 else equi_conditions,
        range_values,
    )
    if outcome is None:
        return None
    left_index, order, starts, ends = outcome
    _right_index = right.index if isinstance(right_index, slice) else right_index
    if (
        isinstance(_right_index, pd.RangeIndex)
        and _right_index.start == 0
        and _right_index.step == 1
    ):
        # Labels equal positions, so order already holds the right labels.
        right_index = pd.Index(order, copy=False)
    else:
        right_index = _right_index[order]
    range_predicates = []
    for condition, values in zip(range_conditions, range_values):
        left_values = df.loc[left_index, condition.left]._values
        left_values = _helpers._convert_array_to_numpy(array=left_values)
        range_predicates.append((left_values, values[order], condition.op))
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
        Output rows whose equality key matched are retained with identity
        values when no range or residual candidate survives. The result is
        empty when no equality key matches.
    """

    outcome = _preparatory_work(df, right, conditions)
    if outcome is None:
        # No equality key matches: return an empty result, consistent with
        # get_join_indices, rather than every row (including null keys).
        output_index = right.index if reverse else df.index
        return _unmatched_aggregation_result(
            output_index=output_index[:0],
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
        output_index = right_index.unique()
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
