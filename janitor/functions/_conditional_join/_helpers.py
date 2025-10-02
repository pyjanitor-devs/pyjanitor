# helper functions for conditional_join.py
from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.core.algorithms import take_nd
from pandas.core.construction import sanitize_array
from pandas.core.indexes.base import Index

from janitor.cython_functions import cond_join, cond_join_indices


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


operator_mapping = {">": 0, ">=": 1, "<": 2, "<=": 3, "==": 4, "!=": 5}

less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _multiple_conditions_get_indices(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
    sizes: np.ndarray,
    keep: str,
    matches: np.ndarray,
    total: int,
):
    """
    get indices for multiple conditions
    """
    if keep == "all":
        return cond_join_indices.build_indices_from_ranges_keep_all(
            starts=starts,
            ends=ends,
            sizes=sizes,
            matches=matches,
            left_index=left_index,
            right_index=right_index,
            left_indices=np.empty(total, dtype=np.intp),
            right_indices=np.empty(total, dtype=np.intp),
            booleans=booleans,
        )
    if keep == "first":
        return cond_join_indices.build_indices_from_ranges_keep_first(
            starts=starts,
            ends=ends,
            sizes=sizes,
            matches=matches,
            left_index=left_index,
            right_index=right_index,
            left_indices=np.empty(total, dtype=np.intp),
            right_indices=np.empty(total, dtype=np.intp),
            booleans=booleans,
        )

    return cond_join_indices.build_indices_from_ranges_keep_last(
        starts=starts,
        ends=ends,
        sizes=sizes,
        matches=matches,
        left_index=left_index,
        right_index=right_index,
        left_indices=np.empty(total, dtype=np.intp),
        right_indices=np.empty(total, dtype=np.intp),
        booleans=booleans,
    )


def _maybe_remove_nulls_from_dataframe(
    df: pd.DataFrame, columns: Sequence, return_bools: bool = False
):
    """
    Remove nulls if op is not !=;
    """
    any_nulls = df.loc[:, [*columns]].isna().any(axis=1)
    if any_nulls.all():
        return None
    if return_bools:
        any_nulls = ~any_nulls
        return any_nulls
    if any_nulls.any():
        df = df.loc[~any_nulls]
    return df


def _separate_conditions_based_on_op(
    conditions: Sequence, keep_equals_separate: bool = False
):
    """
    Create separate blocks (`equals`, `not_equals`, `others`)
    based on `op`
    """
    l_cols = set()
    r_cols = set()
    # check for possibility of a range join
    # keep the first match for le_lt or ge_gt
    le_lt = None
    ge_gt = None
    not_equals = []
    others = []
    equals = []
    for condition in conditions:
        left_on, right_on, op = condition
        if op == _JoinOperator.NOT_EQUAL.value:
            not_equals.append(condition)
            continue
        if op == _JoinOperator.STRICTLY_EQUAL.value:
            l_cols.add(left_on)
            r_cols.add(right_on)
            equals.append(condition)
            continue
        l_cols.add(left_on)
        r_cols.add(right_on)
        others.append(condition)
        if (op in less_than_join_types) and le_lt:
            continue
        elif op in less_than_join_types:
            le_lt = (left_on, right_on, op)
        elif (op in greater_than_join_types) and ge_gt:
            continue
        elif op in greater_than_join_types:
            ge_gt = (left_on, right_on, op)
    non_equi_count = len(others)
    equi_count = len(equals)
    ne_count = len(not_equals)
    if not keep_equals_separate:
        others.extend(equals)
    others.extend(not_equals)
    is_range_join = all((le_lt, ge_gt))
    if is_range_join:
        others = [
            condition
            for condition in others
            if condition not in (ge_gt, le_lt)
        ]
        others = [ge_gt, le_lt] + others
    return {
        "l_cols": l_cols,
        "r_cols": r_cols,
        "is_range_join": is_range_join,
        "conditions": others,
        "equals": equals,
        "equi_count": equi_count,
        "ne_count": ne_count,
        "non_equi_count": non_equi_count,
    }


def _get_positive_matches(
    indices: dict,
    conditions: tuple[tuple],
) -> dict | None:
    """
    Iterate through conditions
    and get positive matches
    """
    starts = indices["starts"]
    ends = indices["ends"]
    sizes = indices["sizes"]
    booleans = indices["booleans"]
    matches = np.empty(sizes.sum(), dtype=np.int8)
    counts_array = np.zeros(sizes.size, dtype=np.intp)
    for number, (
        left,
        right,
        op,
        is_extension_array,
        left_booleans,
        right_booleans,
        any_nulls,
    ) in enumerate(conditions):
        if not any_nulls:
            matches, booleans, counts_array, total, l_counts = (
                cond_join.get_positive_matches(
                    starts=starts,
                    ends=ends,
                    sizes=sizes,
                    op=op,
                    matches=matches,
                    left=left,
                    right=right,
                    counts_array=counts_array,
                    booleans=booleans,
                    first_time=number == 0,
                )
            )
        else:
            matches, booleans, counts_array, total, l_counts = (
                cond_join.get_positive_matches_ne(
                    starts=starts,
                    ends=ends,
                    sizes=sizes,
                    op=op,
                    matches=matches,
                    left=left,
                    right=right,
                    counts_array=counts_array,
                    booleans=booleans,
                    is_extension_array=is_extension_array,
                    first_time=number == 0,
                    left_booleans=left_booleans.astype(np.int8, copy=False),
                    right_booleans=right_booleans.astype(np.int8, copy=False),
                )
            )
        if total == 0:
            return None
    indices["matches"] = matches
    indices["counts_array"] = counts_array
    indices["booleans"] = booleans
    indices["total"] = total
    indices["l_counts"] = l_counts
    return indices


def _get_positive_matches_ranges_positions(
    indices: dict,
    conditions: tuple[tuple],
) -> dict | None:
    """
    Iterate through conditions
    and get positive matches
    """
    starts = indices["starts"]
    ends = indices["ends"]
    sizes = indices["sizes"]
    positions = indices["positions"]
    indexers = indices["indexers"]
    booleans = indices["booleans"]
    matches = np.empty(sizes.sum(), dtype=np.int8)
    counts_array = np.zeros(booleans.size, dtype=np.intp)
    for number, (
        left,
        right,
        op,
        is_extension_array,
        left_booleans,
        right_booleans,
        any_nulls,
    ) in enumerate(conditions):
        if not any_nulls:
            matches, booleans, counts_array, total, l_counts = (
                cond_join.get_positive_matches_ranges_positions(
                    starts=starts,
                    ends=ends,
                    sizes=sizes,
                    op=op,
                    matches=matches,
                    left=left,
                    right=right,
                    counts_array=counts_array,
                    booleans=booleans,
                    positions=positions,
                    indexers=indexers,
                    first_time=number == 0,
                )
            )
        else:
            matches, booleans, counts_array, total, l_counts = (
                cond_join.get_positive_matches_ranges_positions_ne(
                    starts=starts,
                    ends=ends,
                    sizes=sizes,
                    op=op,
                    is_extension_array=is_extension_array,
                    matches=matches,
                    left=left,
                    right=right,
                    counts_array=counts_array,
                    booleans=booleans,
                    left_booleans=left_booleans.astype(np.int8, copy=False),
                    right_booleans=right_booleans.astype(np.int8, copy=False),
                    positions=positions,
                    indexers=indexers,
                    first_time=number == 0,
                )
            )
        if total == 0:
            return None
    indices["matches"] = matches
    indices["counts_array"] = counts_array
    indices["booleans"] = booleans
    indices["total"] = total
    indices["l_counts"] = l_counts
    return indices


def _convert_to_numpy(
    left: np.ndarray, right: np.ndarray, na_value: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure array is a numpy array.
    """
    if pd.api.types.is_extension_array_dtype(left):
        array_dtype = left.dtype.numpy_dtype
        left = left.to_numpy(dtype=array_dtype, na_value=na_value, copy=False)
        right = right.to_numpy(
            dtype=array_dtype, na_value=na_value, copy=False
        )
    if pd.api.types.is_timedelta64_dtype(left):
        left = left.to_numpy(copy=False)
        right = right.to_numpy(copy=False)
    if pd.api.types.is_datetime64_dtype(
        left
    ) or pd.api.types.is_timedelta64_dtype(left):
        left = left.view(np.int64)
        right = right.view(np.int64)
    return left, right


def _convert_array_to_numpy(
    array: np.ndarray,
    na_value: int = 0,
) -> np.ndarray:
    """
    Ensure array is a numpy array.
    """
    if pd.api.types.is_extension_array_dtype(array):
        array_dtype = array.dtype.numpy_dtype
        array = array.to_numpy(
            dtype=array_dtype, na_value=na_value, copy=False
        )
    if pd.api.types.is_timedelta64_dtype(array):
        array = array.to_numpy(copy=False)
    if pd.api.types.is_datetime64_dtype(
        array
    ) or pd.api.types.is_timedelta64_dtype(array):
        array = array.view(np.int64)
    return array


def _build_indices_fast_path_range_join_only(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
    keep: str,
    total: int = None,
    matches: int = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build indices for a single equi join or a true range join
    """
    if keep == "all":
        return cond_join_indices.build_indices_fast_path_keep_all(
            left_index=left_index,
            right_index=right_index,
            left_indices=np.empty(total, dtype=np.intp),
            right_indices=np.empty(total, dtype=np.intp),
            starts=starts,
            ends=ends,
            booleans=booleans,
        )
    if keep == "first":
        return cond_join_indices.build_indices_fast_path_keep_first(
            left_index=left_index,
            right_index=right_index,
            left_indices=np.empty(matches, dtype=np.intp),
            right_indices=np.empty(matches, dtype=np.intp),
            starts=starts,
            ends=ends,
            booleans=booleans,
        )
    # keep=='last'
    return cond_join_indices.build_indices_fast_path_keep_last(
        left_index=left_index,
        right_index=right_index,
        left_indices=np.empty(matches, dtype=np.intp),
        right_indices=np.empty(matches, dtype=np.intp),
        starts=starts,
        ends=ends,
        booleans=booleans,
    )


def _get_positive_matches_no_ranges(
    right_index: np.ndarray,
    booleans: np.ndarray,
    conditions: tuple[tuple],
) -> np.ndarray | None:
    """
    Iterate through conditions
    and get positive matches.
    Applied to indices obtained from pd.merge
    or != only conditions
    """

    for (
        left,
        right,
        op,
        is_extension_array,
        left_booleans,
        right_booleans,
        any_nulls,
    ) in conditions:
        if not any_nulls:
            booleans, total = cond_join.get_positive_matches_no_ranges(
                op=op,
                left=left,
                right=right,
                right_index=right_index,
                booleans=booleans,
            )
        else:
            booleans, total = cond_join.get_positive_matches_no_ranges_ne(
                op=op,
                left=left,
                right=right,
                right_index=right_index,
                booleans=booleans,
                is_extension_array=is_extension_array,
                left_booleans=left_booleans.astype(np.int8, copy=False),
                right_booleans=right_booleans.astype(np.int8, copy=False),
            )

        if total == 0:
            return None

    return booleans


# copied from pandas/core/dtypes/missing.py
# seems function was introduced in 2.2.2
# we should support lesser versions - at least 2.0.0
def construct_1d_array_from_inferred_fill_value(
    value: object, length: int
) -> np.ndarray:
    # Find our empty_value dtype by constructing an array
    #  from our value and doing a .take on it
    arr = sanitize_array(value, Index(range(1)), copy=False)
    taker = -1 * np.ones(length, dtype=np.intp)
    return take_nd(arr, taker)


def _update_search_indices(
    left: np.ndarray,
    right: np.ndarray,
    indices: dict,
    op: str,
    first_time: bool = False,
):
    """
    Update `starts` or `ends` for non-equi
    """
    left, right = _convert_to_numpy(left=left, right=right)
    if (op == ">") and first_time:
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_greater_than_strict_init(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif op == ">":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_greater_than_strict(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif (op == ">=") and first_time:
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_greater_than_init(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif op == ">=":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_greater_than(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif (op == "<") and first_time:
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_less_than_strict_init(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif op == "<":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_less_than_strict(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif (op == "<=") and first_time:
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_less_than_init(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif op == "<=":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_less_than(
                left=left,
                right=right,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    if matches == 0:
        return None
    indices["ends"] = ends
    indices["starts"] = starts
    indices["booleans"] = booleans
    indices["total"] = total
    indices["matches"] = matches
    indices["sizes"] = sizes
    return indices


def _generate_tuples(df: pd.DataFrame, right: pd.DataFrame, conditions: list):
    """
    Build tuple of arrays to pass to numba/cython
    """
    if not conditions:
        return None
    tuples = []
    left_booleans = np.empty(1, dtype=np.bool_)
    right_booleans = np.empty(1, dtype=np.bool_)
    for left_on, right_on, op in conditions:
        left_arr = df[left_on]
        is_extension_array = pd.api.types.is_extension_array_dtype(left_arr)
        left_arr = left_arr._values
        right_arr = right[right_on]._values
        any_nulls = False
        if op == "!=":
            left_booleans = pd.isna(left_arr)
            right_booleans = pd.isna(right_arr)
            any_nulls = left_booleans.any() or right_booleans.any()
        left_arr, right_arr = _convert_to_numpy(left=left_arr, right=right_arr)
        condition = (
            left_arr,
            right_arr,
            operator_mapping[op],
            is_extension_array,
            left_booleans,
            right_booleans,
            any_nulls,
        )
        tuples.append(condition)
    return tuple(tuples)


def _build_indices_single_equi_or_true_range_join(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
    right_is_sorted: bool,
    return_ranges: bool,
    total: int,
    keep: str,
) -> tuple[np.ndarray, np.ndarray] | pd.Series | dict:
    """
    Build indices for a single equi join or a true range join
    """
    if return_ranges:
        if not booleans.all():
            booleans = booleans.astype(np.bool_, copy=False)
            starts = starts[booleans]
            ends = ends[booleans]
            left_index = left_index[booleans]
        return dict(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
        )
    if (keep == "first") and right_is_sorted:
        if not booleans.all():
            booleans = booleans.astype(np.bool_, copy=False)
            starts = starts[booleans]
            left_index = left_index[booleans]
        return left_index, right_index[starts]
    if (keep == "last") and right_is_sorted:
        if not booleans.all():
            booleans = booleans.astype(np.bool_, copy=False)
            ends = ends[booleans]
            left_index = left_index[booleans]
        return left_index, right_index[ends - 1]
    if keep == "all":
        return cond_join_indices.build_indices_fast_path_keep_all(
            left_index=left_index,
            right_index=right_index,
            left_indices=np.empty(total, dtype=np.intp),
            right_indices=np.empty(total, dtype=np.intp),
            starts=starts,
            ends=ends,
            booleans=booleans,
        )
    if keep == "first":
        return cond_join_indices.build_indices_fast_path_keep_first(
            left_index=left_index,
            right_index=right_index,
            left_indices=np.empty(total, dtype=np.intp),
            right_indices=np.empty(total, dtype=np.intp),
            starts=starts,
            ends=ends,
            booleans=booleans,
        )
    # keep=='last'
    return cond_join_indices.build_indices_fast_path_keep_last(
        left_index=left_index,
        right_index=right_index,
        left_indices=np.empty(total, dtype=np.intp),
        right_indices=np.empty(total, dtype=np.intp),
        starts=starts,
        ends=ends,
        booleans=booleans,
    )
