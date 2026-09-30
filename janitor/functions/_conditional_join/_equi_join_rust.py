"""Prepare and dispatch conditional joins with at least one equi predicate.

This module owns the Python-side physical layout used by the Rust equi-join
functions. Range predicates may reorder the right-hand rows first; after that
reordering, right_index maps physical right positions back to public labels.
"""

from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers


def _empty_indices() -> dict[str, np.ndarray]:
    """Return the standard empty conditional-join result.

    Returns:
        A dictionary containing empty left_index and right_index arrays,
        matching the other conditional-join paths.
    """
    empty = np.array([], dtype=np.intp)
    return {"left_index": empty, "right_index": empty}


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
        if np.all(left_indexer == -1):
            return None
        return left_indexer, None
    except pd.errors.InvalidIndexError:
        right_codes, uniques = right_keys.factorize(sort=False)
        left_indexer = uniques.get_indexer(left_keys)
        if np.all(left_indexer == -1):
            return None
        return left_indexer, right_codes


def _build_equi_keys(
    df: pd.DataFrame,
    right: pd.DataFrame,
    right_index: pd.Index | None,
    equi_conditions: list[tuple[str, str, str]],
) -> tuple[pd.Index, pd.Index]:
    """Build aligned single- or multi-column equi keys.

    Args:
        df: Left dataframe after null-row filtering.
        right: Right dataframe after null-row filtering.
        right_index: Optional physical right layout produced by range
            preparation. When present, right keys use this order.
        equi_conditions: Equi predicates as column/operator tuples.

    Returns:
        A pair of aligned pandas indexes. Multiple equi columns are
        represented as MultiIndex values.
    """
    l_cols = []
    r_cols = []
    if right_index is None:
        for left_col, right_col, _ in equi_conditions:
            l_cols.append(df[left_col]._values)
            r_cols.append(right[right_col]._values)
    else:
        for left_col, right_col, _ in equi_conditions:
            l_cols.append(df[left_col]._values)
            r_cols.append(right.loc[right_index, right_col]._values)
    if len(l_cols) > 1:
        l_cols = pd.MultiIndex.from_arrays(l_cols)
        r_cols = pd.MultiIndex.from_arrays(r_cols)
    else:
        l_cols = pd.Index(l_cols[0])
        r_cols = pd.Index(r_cols[0])
    return l_cols, r_cols


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
    keep: str,
    return_building_blocks: bool = False,
) -> dict[str, np.ndarray] | None:
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
        A final left_index/right_index dictionary, a Rust building-block
        dictionary when requested, or the standard empty dictionary when no
        matches exist.
    """
    left_columns = {
        left_column
        for left_column, _, operator in conditions
        if operator != _helpers._JoinOperator.NOT_EQUAL.value
    }
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=left_columns)
    if df is None:
        return None
    right_columns = {
        right_column
        for _, right_column, operator in conditions
        if operator != _helpers._JoinOperator.NOT_EQUAL.value
    }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=right_columns)
    if right is None:
        return None
    mapped_conditions = _helpers._separate_conditions_based_on_join_op(
        conditions=conditions
    )
    le_lt = mapped_conditions["le_lt"]
    ge_gt = mapped_conditions["ge_gt"]
    rest = []
    rest.extend(le_lt)
    rest.extend(ge_gt)
    rest.extend(mapped_conditions["not_equals"])
    range_maybe = []
    if le_lt:
        range_maybe.append(le_lt[0])
    if ge_gt:
        range_maybe.append(ge_gt[0])
    if len(range_maybe) == 1 and len(le_lt) > 1:
        range_maybe = le_lt[:2]
    elif len(range_maybe) == 1 and len(ge_gt) > 1:
        range_maybe = ge_gt[:2]
    # Select at most two range predicates. A second range is retained only
    # when its right values share the first range's physical permutation.
    range_predicates = []
    if range_maybe:
        left_column, right_column, op = range_maybe[0]
        right_, _ = _helpers._sort_if_not_monotonic(series=right[right_column])
        left_array = _helpers._convert_array_to_numpy(array=df[left_column]._values)
        right_array = _helpers._convert_array_to_numpy(array=right_._values)
        range_predicate = (
            left_array,
            right_array,
            op,
        )
        range_predicates.append(range_predicate)
        if len(range_maybe) > 1:
            left_column, right_column, op = range_maybe[1]
            right_ = right.loc[right_.index, right_column]
            if right_.is_monotonic_increasing:
                left_array = _helpers._convert_array_to_numpy(
                    array=df[left_column]._values
                )
                right_array = _helpers._convert_array_to_numpy(array=right_._values)
                range_predicate = (
                    left_array,
                    right_array,
                    op,
                )
                range_predicates.append(range_predicate)
            else:
                range_maybe = [range_maybe[0]]
        right_index = right_.index
        rest = [condition for condition in rest if condition not in range_maybe]
    else:
        right_index = None
    left_keys, right_keys = _build_equi_keys(
        df=df,
        right=right,
        right_index=right_index,
        equi_conditions=mapped_conditions["equals"],
    )
    equi_predicates = _build_equi_predicate(
        left_keys=left_keys,
        right_keys=right_keys,
    )

    if equi_predicates is None:
        return _empty_indices()

    left_indexer, right_codes = equi_predicates

    residual_predicates = []
    if rest:
        for left_column, right_column, operator in rest:
            residual_predicate = _helpers._build_residual_predicate(
                left=df[left_column],
                right=right[right_column],
                operation=operator,
                right_index=right_index,
            )
            residual_predicates.append(residual_predicate)
    left_index = _helpers._convert_array_to_numpy(array=df.index._values)
    if right_index is None:
        right_index = right.index
    right_index = _helpers._convert_array_to_numpy(array=right_index._values)

    # A unique right equi key already gives one direct right position per
    # left row. There is no duplicate metadata for Rust to build here.
    if right_codes is None and not range_predicates and not residual_predicates:
        booleans = left_indexer != -1
        if not booleans.all():
            left_indexer = left_indexer[booleans]
            left_index = left_index[booleans]
        right_index = right_index[left_indexer]
        return {"left_index": left_index, "right_index": right_index}
    if (
        right_codes is not None
        and not range_predicates
        and not residual_predicates
        and return_building_blocks
    ):
        blocks = janitor_rs.equi_join_building_blocks(
            left_index,
            right_index,
            left_indexer,
            right_codes,
        )
        if blocks is None:
            return _empty_indices()
        return blocks
    if (right_codes is not None) and not range_predicates and not residual_predicates:
        indices = janitor_rs.equi_join_indices(
            left_index,
            right_index,
            left_indexer,
            right_codes,
            keep,
        )
        if indices is None:
            return _empty_indices()
        return indices
    indices = janitor_rs.equi_join_filtered_indices(
        left_index,
        right_index,
        left_indexer,
        right_codes,
        range_predicates,
        residual_predicates,
        keep,
    )
    if indices is None:
        return _empty_indices()
    return indices
