from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _helpers,
    _range_indices,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    return_matching_indices: bool,
) -> tuple:
    """
    Get indices, or aggregates, for multiple conditions,
    where `==` is present
    """
    empty_array = np.array([], dtype=np.intp)
    mapping = _helpers._separate_conditions_based_on_op(conditions=conditions)
    _columns = (
        mapping["le_or_ge"],
        mapping["le_lt"],
        mapping["ge_gt"],
        mapping["equals"],
    )
    columns = []
    for entry in _columns:
        if not entry:
            continue
        if isinstance(entry, tuple):
            columns.append(entry)
        else:
            columns.extend(entry)
    left_columns = set()
    right_columns = set()
    for left_col, right_col, _ in columns:
        left_columns.add(left_col)
        right_columns.add(right_col)
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=left_columns)
    if df is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=right_columns)
    if right is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    try:
        l_cols = []
        r_cols = []
        for left_col, right_col, _ in mapping["equals"]:
            l_cols.append(df[left_col]._values)
            r_cols.append(right[right_col]._values)
        if len(l_cols) > 1:
            l_cols = pd.MultiIndex.from_arrays(l_cols)
            r_cols = pd.MultiIndex.from_arrays(r_cols)
        else:
            l_cols = pd.Index(l_cols[0])
            r_cols = pd.Index(r_cols[0])
        left_index = df.index._values
        right_index = right.index._values
        indexers = r_cols.get_indexer(l_cols)
        booleans = indexers != -1
        if not booleans.any():
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        if not booleans.all() and not any(
            (
                mapping["le_or_ge"],
                mapping["le_lt"],
                mapping["ge_gt"],
                mapping["not_equals"],
            )
        ):
            indexers = indexers[booleans]
            left_index = left_index[booleans]
            right_index = right_index[indexers]
            return {
                "left_index": left_index,
                "right_index": right_index,
            }
        if not any(
            (
                mapping["le_or_ge"],
                mapping["le_lt"],
                mapping["ge_gt"],
                mapping["not_equals"],
            )
        ):
            right_index = right_index[indexers]
            return {
                "left_index": left_index,
                "right_index": right_index,
            }
        rest = mapping["le_or_ge"]
        rest.append(mapping["le_lt"])
        rest.append(mapping["ge_gt"])
        rest.extend(mapping["not_equals"])
        rest = filter(None, rest)
        rest = dict.fromkeys(rest)
        outcome = _helpers._update_positions_no_range_(
            df=df, right=right, conditions=rest, positions=indexers
        )
        if outcome is None:
            return {
                "left_index": empty_array,
                "right_index": empty_array,
            }
        left_index = janitor_rs.index_trim_positions(
            index=left_index,
            positions=outcome["positions"],
            length=outcome["total"],
        )
        right_index = janitor_rs.build_positional_index(
            index=right_index,
            positions=outcome["positions"],
            length=outcome["total"],
        )
        return {
            "left_index": left_index,
            "right_index": right_index,
        }
    except pd.errors.InvalidIndexError:
        if mapping["is_range_join"]:
            return _get_indices_range_join(
                df=df,
                right=right,
                mapping=mapping,
                keep=keep,
                return_matching_indices=return_matching_indices,
            )
        if mapping["not_equals"] and not mapping["le_or_ge"]:
            return _get_indices_ne_only(df=df, right=right, mapping=mapping, keep=keep)
        if mapping["le_or_ge"]:
            return _get_indices_le_or_ge_and_not_range_join(
                df=df,
                right=right,
                mapping=mapping,
                keep=keep,
                return_matching_indices=return_matching_indices,
            )
        return _get_indices_equi_join_only(
            df=df,
            right=right,
            mapping=mapping,
            return_matching_indices=return_matching_indices,
            keep=keep,
        )


def _check_sorted_within_groups(
    equi_cols: list[tuple],
    right: pd.DataFrame,
    r_col: str | tuple,
) -> bool:
    """Check if r_col is sorted within each group of equi_cols."""
    r_cols = []
    for _, right_col, _ in equi_cols:
        r_cols.append(right_col)
    # handle duplicate keys in equi_cols
    # while still maintaining order
    grouper = dict.fromkeys(r_cols)
    grouper = [*grouper]
    grouped = right.groupby(grouper, sort=False, observed=True)
    check = grouped[r_col].is_monotonic_increasing.all()
    return check


def _maybe_sort_right(
    right: pd.DataFrame,
    r1_col: str | tuple,
    r2_col: str | tuple | None = None,
    is_sorted: bool = False,
) -> pd.DataFrame:
    """Sort the right DataFrame if needed."""
    if is_sorted:
        return right
    if r2_col is None:
        sorter = r1_col
    # possibly a range join
    else:
        # handle duplicates
        # while still maintaining order
        sorter = dict.fromkeys([r1_col, r2_col])
        sorter = [*sorter]
    right = right.sort_values(sorter, kind="stable", ignore_index=False)
    return right


def _build_equi_indices(df, right, mapping):
    l_cols = []
    r_cols = []
    for left_col, right_col, _ in mapping["equals"]:
        l_cols.append(df[left_col]._values)
        r_cols.append(right[right_col]._values)
    if len(l_cols) > 1:
        l_cols = pd.MultiIndex.from_arrays(l_cols)
        r_cols = pd.MultiIndex.from_arrays(r_cols)
    else:
        l_cols = pd.Index(l_cols[0])
        r_cols = pd.Index(r_cols[0])
    return l_cols, r_cols


def _get_positions_right(right_columns: pd.Index):
    positions, uniques = right_columns.factorize()
    counts = np.bincount(positions, minlength=len(uniques))
    starts = np.empty(counts.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = counts.cumsum()[:-1]
    ends = starts + counts
    return positions, uniques, starts, ends


def _get_indexers(l_cols, uniques, starts, ends):
    indexers = uniques.get_indexer(l_cols)
    booleans = indexers == -1
    if booleans.all():
        return None
    # align df to right
    starts = starts[indexers]
    ends = ends[indexers]
    if booleans.any():
        starts = np.where(booleans, -1, starts)
        ends = np.where(booleans, -1, ends)
    return indexers, starts, ends


def _update_positions_ge_gt(
    op: str,
    l_column: np.ndarray,
    r_column: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    if op == ">":
        return _helpers._binary_search_gt(
            left=l_column, right=r_column, starts=starts, ends=ends
        )
    return _helpers._binary_search_ge(
        left=l_column, right=r_column, starts=starts, ends=ends
    )


def _update_positions_le_lt(
    op: str,
    l_column: np.ndarray,
    r_column: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    if op == "<":
        return _helpers._binary_search_lt(
            left=l_column, right=r_column, starts=starts, ends=ends
        )
    return _helpers._binary_search_le(
        left=l_column, right=r_column, starts=starts, ends=ends
    )


def _get_indices_range_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    keep: str,
    return_matching_indices: bool,
) -> dict:
    """Get indices if '>/>=' and '</<=' present"""
    empty_array = np.array([], dtype=np.intp)
    (_, r1_col, _) = mapping["ge_gt"]
    (_, r2_col, _) = mapping["le_lt"]
    is_sorted = _check_sorted_within_groups(
        equi_cols=mapping["equals"],
        right=right,
        r_col=r1_col,
    )
    right = _maybe_sort_right(
        right=right, r1_col=r1_col, r2_col=r2_col, is_sorted=is_sorted
    )
    l_cols, r_cols = _build_equi_indices(df=df, right=right, mapping=mapping)
    positions, uniques, starts, ends = _get_positions_right(right_columns=r_cols)
    check = r_cols.is_monotonic_increasing
    if not check:
        reordered_positions = janitor_rs.reorder_index(
            positions=positions, starts=starts
        )
        right = right.iloc[reordered_positions]
    outcome = _get_indexers(l_cols=l_cols, uniques=uniques, starts=starts, ends=ends)
    if outcome is None:
        return {"left_index": empty_array, "right_index": empty_array}
    indexers, starts, ends = outcome
    (l1_col, r1_col, op) = mapping["ge_gt"]
    r_column = right[r1_col]._values
    r_column = _helpers._convert_array_to_numpy(array=r_column)
    l_column = df[l1_col]._values
    l_column = _helpers._convert_array_to_numpy(array=l_column)
    ends = _update_positions_ge_gt(
        op=op,
        l_column=l_column,
        r_column=r_column,
        starts=starts,
        ends=ends,
    )
    (l2_col, r2_col, op) = mapping["le_lt"]
    is_sorted = _check_sorted_within_groups(
        equi_cols=mapping["equals"],
        right=right,
        r_col=r2_col,
    )
    # if possible, run binary search on both sides
    #  - ge_gt and le_lt
    if is_sorted:
        r_column = right[r2_col]._values
        r_column = _helpers._convert_array_to_numpy(array=r_column)
        l_column = df[l2_col]._values
        l_column = _helpers._convert_array_to_numpy(array=l_column)
        starts = _update_positions_le_lt(
            op=op,
            l_column=l_column,
            r_column=r_column,
            starts=starts,
            ends=ends,
        )
        rest = []
    else:
        rest = [mapping["le_lt"]]
    rest.extend(mapping["le_or_ge"])
    rest.extend(mapping["not_equals"])
    rest = [entry for entry in rest if entry]
    booleans = (starts == -1) | (ends == -1) | (starts >= ends)
    if booleans.all():
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left_index = df.index._values
    if booleans.any():
        booleans = ~booleans
        starts = starts[booleans]
        ends = ends[booleans]
        left_index = left_index[booleans]
    if not rest:
        right_index = right.index._values
        return _range_indices._build_indices(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            keep=keep,
            right_is_sorted=check and is_sorted,
            return_matching_indices=return_matching_indices,
        )
    outcome = _helpers._get_positive_matches_conditions(
        df=df,
        right=right,
        conditions=rest,
        left_index=left_index,
        starts=starts,
        ends=ends,
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    return _helpers.build_indices_matches(
        left_index=left_index,
        right_index=right.index._values,
        counts_array=outcome["counts_array"],
        starts=starts,
        ends=ends,
        matches=outcome["matches"],
        total=outcome["total"],
        keep=keep,
    )


def _get_indices_ne_only(
    df: pd.DataFrame, right: pd.DataFrame, mapping: dict, keep: str
) -> dict:
    """
    Get indices for != only
    """
    empty_array = np.array([], dtype=np.intp)
    l_cols, r_cols = _build_equi_indices(df=df, right=right, mapping=mapping)
    positions, uniques, starts, ends = _get_positions_right(right_columns=r_cols)
    check = r_cols.is_monotonic_increasing
    if not check:
        reordered_positions = janitor_rs.reorder_index(
            positions=positions, starts=starts
        )
        right = right.iloc[reordered_positions]
    outcome = _get_indexers(l_cols=l_cols, uniques=uniques, starts=starts, ends=ends)
    if outcome is None:
        return {"left_index": empty_array, "right_index": empty_array}
    indexers, starts, ends = outcome
    booleans = indexers == -1
    if booleans.all():
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left_index = df.index._values
    if booleans.any():
        booleans = ~booleans
        indexers = indexers[booleans]
        starts = starts[booleans]
        ends = ends[booleans]
        left_index = left_index[booleans]
    outcome = _helpers._get_positive_matches_conditions(
        df=df,
        right=right,
        conditions=mapping["not_equals"],
        left_index=left_index,
        starts=starts,
        ends=ends,
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    return _helpers.build_indices_matches(
        left_index=left_index,
        right_index=right.index._values,
        counts_array=outcome["counts_array"],
        starts=starts,
        ends=ends,
        matches=outcome["matches"],
        total=outcome["total"],
        keep=keep,
    )


def _get_indices_le_or_ge_and_not_range_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    keep: str,
    return_matching_indices: bool,
) -> dict:
    """Get indices if </<=/>/>= but not range join"""
    empty_array = np.array([], dtype=np.intp)
    (_, r1_col, _), *_ = mapping["le_or_ge"]
    is_sorted = _check_sorted_within_groups(
        equi_cols=mapping["equals"],
        right=right,
        r_col=r1_col,
    )
    right = _maybe_sort_right(
        right=right, r1_col=r1_col, r2_col=None, is_sorted=is_sorted
    )
    l_cols, r_cols = _build_equi_indices(df=df, right=right, mapping=mapping)
    positions, uniques, starts, ends = _get_positions_right(right_columns=r_cols)
    check = r_cols.is_monotonic_increasing
    if not check:
        reordered_positions = janitor_rs.reorder_index(
            positions=positions, starts=starts
        )
        right = right.iloc[reordered_positions]
    outcome = _get_indexers(l_cols=l_cols, uniques=uniques, starts=starts, ends=ends)
    if outcome is None:
        return {"left_index": empty_array, "right_index": empty_array}
    indexers, starts, ends = outcome
    (l1_col, r1_col, op), *rest = mapping["le_or_ge"]
    r_column = right[r1_col]._values
    r_column = _helpers._convert_array_to_numpy(array=r_column)
    l_column = df[l1_col]._values
    l_column = _helpers._convert_array_to_numpy(array=l_column)
    if op in _helpers.greater_than_join_types:
        ends = _update_positions_ge_gt(
            op=op,
            l_column=l_column,
            r_column=r_column,
            starts=starts,
            ends=ends,
        )
    else:
        starts = _update_positions_le_lt(
            op=op,
            l_column=l_column,
            r_column=r_column,
            starts=starts,
            ends=ends,
        )
    rest.extend(mapping["not_equals"])
    rest = [entry for entry in rest if entry]
    booleans = (starts == -1) | (ends == -1) | (starts >= ends)
    if booleans.all():
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left_index = df.index._values
    if booleans.any():
        booleans = ~booleans
        indexers = indexers[booleans]
        starts = starts[booleans]
        ends = ends[booleans]
        left_index = left_index[booleans]
    if not rest:
        right_index = right.index._values
        return _range_indices._build_indices(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            keep=keep,
            right_is_sorted=check and is_sorted,
            return_matching_indices=return_matching_indices,
        )
    outcome = _helpers._get_positive_matches_conditions(
        df=df,
        right=right,
        conditions=rest,
        left_index=left_index,
        starts=starts,
        ends=ends,
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    return _helpers.build_indices_matches(
        left_index=left_index,
        right_index=right.index._values,
        counts_array=outcome["counts_array"],
        starts=starts,
        ends=ends,
        matches=outcome["matches"],
        total=outcome["total"],
        keep=keep,
    )


def _get_indices_equi_join_only(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    return_matching_indices: bool,
    keep: str,
) -> dict:
    """Get indices for an equi join only"""
    # purely equi join, and return_matching_indices is True
    l_cols, r_cols = _build_equi_indices(df=df, right=right, mapping=mapping)
    positions, uniques, starts, ends = _get_positions_right(right_columns=r_cols)
    check = r_cols.is_monotonic_increasing
    if not check:
        reordered_positions = janitor_rs.reorder_index(
            positions=positions, starts=starts
        )
        right = right.iloc[reordered_positions]
    indexers, starts, ends = _get_indexers(
        l_cols=l_cols, uniques=uniques, starts=starts, ends=ends
    )
    booleans = (starts == -1) | (ends == -1) | (starts >= ends)
    if booleans.all():
        return {
            "left_index": np.array([], dtype=np.intp),
            "right_index": np.array([], dtype=np.intp),
        }
    left_index = df.index._values
    if booleans.any():
        booleans = ~booleans
        indexers = indexers[booleans]
        starts = starts[booleans]
        ends = ends[booleans]
        left_index = left_index[booleans]
    right_index = right.index._values
    return _range_indices._build_indices(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        keep=keep,
        right_is_sorted=check,
        return_matching_indices=return_matching_indices,
    )
