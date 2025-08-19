# helper functions for conditional_join.py
from __future__ import annotations

from enum import Enum
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from pandas.core.algorithms import take_nd
from pandas.core.construction import sanitize_array
from pandas.core.indexes.base import Index

from janitor.cython_functions import cond_join


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


def _null_checks_cond_join(series: pd.Series) -> tuple | None:
    """
    Checks for nulls in the pandas series before conducting binary search.
    """
    any_nulls = series.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        series = series[~any_nulls]
    return series, any_nulls.any()


def _sort_if_not_monotonic(series: pd.Series) -> pd.Series | None:
    """
    Sort the pandas `series` if it is not monotonic increasing
    """

    is_sorted = series.is_monotonic_increasing
    if not is_sorted:
        series = series.sort_values(kind="stable")

    return series, is_sorted


def _equal_indices(
    left: pd.Series,
    right: pd.Series,
) -> tuple:
    """
    Use binary search to get indices where left
    is equal to right.

    A tuple of integer indexes
    for left and right is returned.
    """

    outcome = _null_checks_cond_join(series=left)
    if outcome is None:
        return None
    left, _ = outcome
    outcome = _null_checks_cond_join(series=right)
    if outcome is None:
        return None
    right, _ = outcome
    right, _ = _sort_if_not_monotonic(series=right)
    # steal some perf here within the binary search
    # search for uniques
    # and later index them with left_positions
    # it is assumed that users will only reach for this
    # if the data is reasonably duplicated; if not
    # pd.merge is superb especially if it's a one-to-one
    # or one-to-many
    left_index = left.index._values
    right_index = right.index._values
    right = right.array
    positions, left = pd.factorize(left, sort=False)
    left = left.array
    starts = right.searchsorted(left, side="left")
    starts = starts[positions]
    ends = right.searchsorted(left, side="right")
    ends = ends[positions]
    booleans = starts < ends
    if not booleans.any():
        return None
    if not booleans.all():
        left_index = left_index[booleans]
        starts = starts[booleans]
        ends = ends[booleans]
    return left_index, right_index, starts, ends


def _less_than_indices(
    left: pd.array,
    left_index: np.ndarray,
    right: pd.array,
    strict: bool,
) -> tuple | None:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.

    Returns the left index and the binary search positions for left in right.
    """

    search_indices = right.searchsorted(left, side="left")
    # if any of the positions in `search_indices`
    # is equal to the length of `right_keys`
    # that means the respective position in `left`
    # has no values from `right` that are less than
    # or equal, and should therefore be discarded
    len_right = right.size
    booleans = search_indices < len_right
    if not booleans.any():
        return None
    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]

    # the idea here is that if there are any equal values
    # shift to the right to the immediate next position
    # that is not equal
    if strict:
        booleans = left == right[search_indices]
        # replace positions where rows are equal
        # with positions from searchsorted('right')
        # positions from searchsorted('right') will never
        # be equal and will be the furthermost in terms of position
        # example : right -> [2, 2, 2, 3], and we need
        # positions where values are not equal for 2;
        # the furthermost will be 3, and searchsorted('right')
        # will return position 3.
        if booleans.any():
            replacements = right.searchsorted(left, side="right")
            # now we can safely replace values
            # with strictly less than positions
            search_indices = np.where(booleans, replacements, search_indices)
        # check again if any of the values
        # have become equal to length of right
        # and get rid of them
        booleans = search_indices < len_right

        if not booleans.any():
            return None

        if not booleans.all():
            left_index = left_index[booleans]
            search_indices = search_indices[booleans]

    return left_index, search_indices


def _less_than_single_join(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    keep: str,
    return_ranges: bool,
    row_count: Hashable = None,
) -> tuple:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.

    A tuple of integer indexes
    for left and right is returned.
    """

    # no point going through all the hassle
    if left.min() > right.max():
        return None

    outcome = _null_checks_cond_join(series=left)
    if not outcome:
        return None
    left, _ = outcome
    outcome = _null_checks_cond_join(series=right)
    if not outcome:
        return None
    right, any_nulls = outcome
    right, right_is_sorted = _sort_if_not_monotonic(series=right)
    outcome = _less_than_indices(
        left=left.array,
        right=right.array,
        left_index=left.index._values,
        strict=strict,
    )

    if not outcome:
        return None
    left_index, search_indices = outcome
    len_right = right.size
    right_index = right.index._values
    if row_count:
        return pd.Series(
            index=left_index, data=len_right - search_indices, name=row_count
        )
    if right_is_sorted & (keep == "last"):
        indexer = np.empty_like(search_indices)
        indexer[:] = len_right - 1
        return left_index, right_index[indexer]
    if right_is_sorted & (keep == "first") & any_nulls:
        return left_index, right_index[search_indices]
    if right_is_sorted & (keep == "first"):
        return left_index, search_indices
    if return_ranges:
        return dict(
            left_index=left_index,
            right_index=right_index,
            starts=search_indices,
            ends=np.repeat(len_right, search_indices.size),
        )
    right = [right_index[ind:len_right] for ind in search_indices]
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(len_right - search_indices)
    return left, right


def _greater_than_indices(
    left: pd.array,
    left_index: np.ndarray,
    right: pd.array,
    strict: bool,
) -> tuple | None:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.

    if multiple_conditions is False, a tuple of integer indexes
    for left and right is returned;
    else a tuple of the index for left, right, as well
    as the positions of left in right is returned.
    """
    search_indices = right.searchsorted(left, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0 (less than 1), it implies that
    # left[position] is not greater than any value
    # in right
    booleans = search_indices > 0
    if not booleans.any():
        return None
    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]

    # the idea here is that if there are any equal values
    # shift downwards to the immediate next position
    # that is not equal
    if strict:
        booleans = left == right[search_indices - 1]
        # replace positions where rows are equal with
        # searchsorted('left');
        # this works fine since we will be using the value
        # as the right side of a slice, which is not included
        # in the final computed value
        if booleans.any():
            replacements = right.searchsorted(left, side="left")
            # now we can safely replace values
            # with strictly greater than positions
            search_indices = np.where(booleans, replacements, search_indices)
        # any value less than 1 should be discarded
        # since the lowest value for binary search
        # with side='right' should be 1
        booleans = search_indices > 0
        if not booleans.any():
            return None
        if not booleans.all():
            left_index = left_index[booleans]
            search_indices = search_indices[booleans]

    return left_index, search_indices


def _greater_than_single_join(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    keep: str,
    return_ranges: bool,
    row_count: Hashable = None,
) -> tuple:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.

    if multiple_conditions is False, a tuple of integer indexes
    for left and right is returned;
    else a tuple of the index for left, right, as well
    as the positions of left in right is returned.
    """

    # quick break, avoiding the hassle
    if left.max() < right.min():
        return None

    outcome = _null_checks_cond_join(series=left)
    if outcome is None:
        return None
    left, _ = outcome
    outcome = _null_checks_cond_join(series=right)
    if outcome is None:
        return None
    right, any_nulls = outcome
    right, right_is_sorted = _sort_if_not_monotonic(series=right)

    outcome = _greater_than_indices(
        left=left.array,
        right=right.array,
        left_index=left.index._values,
        strict=strict,
    )

    if outcome is None:
        return None
    left_index, search_indices = outcome
    if row_count:
        return pd.Series(index=left_index, data=search_indices, name=row_count)
    right_index = right.index._values
    if right_is_sorted & (keep == "first"):
        indexer = np.zeros_like(search_indices)
        return left_index, right_index[indexer]
    if right_is_sorted & (keep == "last") & any_nulls:
        return left_index, right_index[search_indices - 1]
    if right_is_sorted & (keep == "last"):
        return left_index, search_indices - 1
    if return_ranges:
        return dict(
            left_index=left_index,
            right_index=right_index,
            starts=np.repeat(0, search_indices.size),
            ends=search_indices,
        )
    right = [right_index[:ind] for ind in search_indices]
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(search_indices)
    return left, right


def _not_equal_indices(left: pd.Series, right: pd.Series, keep: str) -> tuple:
    """
    Use binary search to get indices where
    `left` is exactly  not equal to `right`.

    It is a combination of strictly less than
    and strictly greater than indices.

    A tuple of integer indexes for left and right
    is returned.
    """

    dummy = np.array([], dtype=int)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    lt_left = [dummy]
    lt_right = [dummy]
    gt_left = [dummy]
    gt_right = [dummy]
    any_left_nulls = left.isna()
    any_right_nulls = right.isna()
    if any_left_nulls.any():
        l1_nulls = left.index[any_left_nulls.array]
        l1_nulls = l1_nulls.to_numpy(copy=False)
        r1_nulls = right.index
        # avoid NAN duplicates
        if any_right_nulls.any():
            r1_nulls = r1_nulls[~any_right_nulls.array]
        r1_nulls = r1_nulls.to_numpy(copy=False)
        nulls_count = l1_nulls.size
        # blow up nulls to match length of right
        l1_nulls = np.tile(l1_nulls, r1_nulls.size)
        # ensure length of right matches left
        if nulls_count > 1:
            r1_nulls = np.repeat(r1_nulls, nulls_count)
    if any_right_nulls.any():
        r2_nulls = right.index[any_right_nulls.array]
        r2_nulls = r2_nulls.to_numpy(copy=False)
        l2_nulls = left.index
        right = right[~any_right_nulls]
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)

    l1_nulls = [l1_nulls, l2_nulls]
    r1_nulls = [r1_nulls, r2_nulls]
    check1 = _null_checks_cond_join(series=left)
    check2 = _null_checks_cond_join(series=right)
    if (check1 is None) or (check2 is None):
        lt_left = [dummy]
        lt_right = [dummy]
    else:
        left, _ = check1
        right, _ = check2
        right, _ = _sort_if_not_monotonic(series=right)
        right_index = right.index._values
        outcome = _less_than_indices(
            left=left.array,
            left_index=left.index._values,
            right=right.array,
            strict=True,
        )
        if outcome is not None:
            len_right = right.size
            lt_left, search_indices = outcome
            lt_right = [right_index[ind:len_right] for ind in search_indices]
            lt_left = [lt_left.repeat(len_right - search_indices)]
        outcome = _greater_than_indices(
            left=left.array,
            right=right.array,
            left_index=left.index._values,
            strict=True,
        )
        if outcome is not None:
            gt_left, search_indices = outcome
            gt_right = [right_index[:ind] for ind in search_indices]
            gt_left = [gt_left.repeat(search_indices)]
    lt_left.extend(gt_left)
    lt_left.extend(l1_nulls)
    lt_right.extend(gt_right)
    lt_right.extend(r1_nulls)
    left = np.concatenate(lt_left)
    right = np.concatenate(lt_right)
    if (not left.size) & (not right.size):
        return None
    return _keep_output(keep, left, right)


def _generic_func_cond_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    keep: str,
    row_count: Hashable = None,
    return_ranges: bool = False,
) -> tuple:
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices,
    or _not_equal_indices).
    """
    strict = False

    if op in {
        _JoinOperator.GREATER_THAN.value,
        _JoinOperator.LESS_THAN.value,
        _JoinOperator.NOT_EQUAL.value,
    }:
        strict = True

    if op in less_than_join_types:
        return _less_than_single_join(
            left=left,
            right=right,
            strict=strict,
            keep=keep,
            row_count=row_count,
            return_ranges=return_ranges,
        )
    if op in greater_than_join_types:
        return _greater_than_single_join(
            left=left,
            right=right,
            strict=strict,
            keep=keep,
            row_count=row_count,
            return_ranges=return_ranges,
        )
    if op == _JoinOperator.NOT_EQUAL.value:
        outcome = _not_equal_indices(left=left, right=right, keep=keep)
        if outcome is None:
            return None
        left_index, right_index = outcome
        if row_count:
            return (
                pd.Index(left_index).value_counts(sort=False).rename(row_count)
            )
        return left_index, right_index
    return _equal_indices(
        left=left,
        right=right,
        return_ranges=return_ranges,
    )


def _keep_output(keep: str, left: np.ndarray, right: np.ndarray):
    """return indices for left and right index based on the value of `keep`."""
    if keep == "all":
        return left, right
    grouped = pd.Series(right).groupby(left, sort=False)
    if keep == "first":
        grouped = grouped.min()
        return grouped.index, grouped._values
    grouped = grouped.max()
    return grouped.index, grouped._values


def _multiple_conditions_get_indices(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    start_indices: np.ndarray,
    booleans: np.ndarray,
    sizes: np.ndarray,
    counts_array: np.ndarray,
    keep: str,
    matches: np.ndarray,
):
    """
    get indices for multiple conditions
    """
    if keep == "all":
        total = counts_array.sum()
        return cond_join.build_indices_keep_all(
            starts=starts,
            sizes=sizes,
            matches=matches,
            starts_indices=start_indices,
            left_index=left_index,
            right_index=right_index,
            left_array=np.empty(total, dtype=np.intp),
            right_array=np.empty(total, dtype=np.intp),
            counts_array=counts_array,
            booleans=booleans,
        )
    total = booleans.sum()
    if keep == "first":
        return cond_join.build_indices_keep_first(
            starts=starts,
            sizes=sizes,
            matches=matches,
            starts_indices=start_indices,
            left_index=left_index,
            right_index=right_index,
            left_array=np.empty(total, dtype=np.intp),
            right_array=np.empty(total, dtype=np.intp),
            counts_array=counts_array,
            booleans=booleans,
        )

    return cond_join.build_indices_keep_last(
        starts=starts,
        sizes=sizes,
        matches=matches,
        starts_indices=start_indices,
        left_index=left_index,
        right_index=right_index,
        left_array=np.empty(total, dtype=np.intp),
        right_array=np.empty(total, dtype=np.intp),
        counts_array=counts_array,
        booleans=booleans,
    )


def _remove_nulls_multiple_conditions(df: pd.DataFrame, columns: Sequence):
    """
    Remove nulls if op is not !=;
    applies to multiple join conditions
    """
    any_nulls = df.loc[:, [*columns]].isna().any(axis=1)
    if any_nulls.all():
        return None
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
        "less_than_or_greater_than": any((le_lt, ge_gt)),
        "conditions": others,
        "equals": equals,
        "non_equi_count": non_equi_count,
    }


def _can_pass_ne_to_cython(left: np.ndarray, right=np.ndarray) -> bool:
    """
    Check if != condition can be passed to cython
    without extra work
    for possibly faster computation
    """
    check1 = pd.isna(left).any()
    check2 = pd.isna(right).any()
    check = any((check1, check2))
    check = not check
    return check


def _get_positive_matches(
    df: pd.DataFrame,
    right: pd.DataFrame,
    indices: dict,
    conditions: list[tuple],
    booleans: np.ndarray,
) -> dict | None:
    """
    Iterate through conditions
    and get positive matches
    """
    starts = indices["starts"]
    ends = indices["ends"]
    sizes = indices["sizes"]
    start_indices = indices["start_indices"]
    matches = np.ones(sizes.sum(), dtype=np.int8)
    counts_array = np.zeros(sizes.size, dtype=np.intp)
    for left_on, right_on, op in conditions:
        left_arr = df[left_on]
        is_extension_array = pd.api.types.is_extension_array_dtype(left_arr)
        left_arr = left_arr._values
        right_arr = right[right_on]._values
        check = True
        if op == "!=":
            check = _can_pass_ne_to_cython(left=left_arr, right=right_arr)
            if not check:
                left_booleans = pd.isna(left_arr).astype(np.int8, copy=False)
                right_booleans = pd.isna(right_arr).astype(np.int8, copy=False)
            else:
                left_booleans = None
                right_booleans = None
        left_arr, right_arr = _convert_to_numpy(left=left_arr, right=right_arr)
        if check:
            matches, booleans, counts_array, any_match = (
                cond_join.get_positive_matches(
                    start_indices=start_indices,
                    starts=starts,
                    ends=ends,
                    left_array=left_arr,
                    right_array=right_arr,
                    op=operator_mapping[op],
                    matches=matches,
                    booleans=booleans,
                    counts_array=counts_array,
                )
            )
        # respect pandas' rules when dealing with NA
        elif not check and is_extension_array:
            matches, booleans, counts_array, any_match = (
                cond_join.get_positive_matches_ne_pandas_array(
                    start_indices=start_indices,
                    starts=starts,
                    ends=ends,
                    left_array=left_arr,
                    right_array=right_arr,
                    op=operator_mapping[op],
                    matches=matches,
                    booleans=booleans,
                    counts_array=counts_array,
                    left_booleans=left_booleans,
                    right_booleans=right_booleans,
                )
            )
        else:
            matches, booleans, counts_array, any_match = (
                cond_join.get_positive_matches_ne(
                    start_indices=start_indices,
                    starts=starts,
                    ends=ends,
                    left_array=left_arr,
                    right_array=right_arr,
                    op=operator_mapping[op],
                    matches=matches,
                    booleans=booleans,
                    counts_array=counts_array,
                    left_booleans=left_booleans,
                    right_booleans=right_booleans,
                )
            )
        if not any_match:
            return None
    indices["matches"] = matches
    indices["counts_array"] = counts_array
    indices["booleans"] = booleans
    return indices


def _build_start_indices(indices: dict) -> dict:
    """
    Update indices with start_indices and sizes
    """
    sizes = indices["sizes"]
    start_indices = np.empty(sizes.size, dtype=np.intp)
    start_indices[0] = 0
    start_indices[1:] = sizes.cumsum()[:-1]
    indices["start_indices"] = start_indices
    return indices


def _convert_to_numpy(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure array is a numpy array.
    """
    if pd.api.types.is_extension_array_dtype(left):
        array_dtype = left.dtype.numpy_dtype
        left = left.to_numpy(dtype=array_dtype, na_value=-1, copy=False)
        right = right.to_numpy(dtype=array_dtype, na_value=-1, copy=False)
    if pd.api.types.is_timedelta64_dtype(left):
        left = left.to_numpy(copy=False)
        right = right.to_numpy(copy=False)
    if pd.api.types.is_datetime64_dtype(
        left
    ) or pd.api.types.is_timedelta64_dtype(left):
        left = left.view(np.int64)
        right = right.view(np.int64)
    return left, right


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
    Build indices for a single equi join and a true range join
    """
    if keep == "all":
        return cond_join.build_indices_equi_range_join_only_fast_path_keep_all(
            left_index=left_index,
            right_index=right_index,
            left_indices=np.empty(total, dtype=np.intp),
            right_indices=np.empty(total, dtype=np.intp),
            starts=starts,
            ends=ends,
            booleans=booleans,
        )
    if keep == "first":
        return (
            cond_join.build_indices_equi_range_join_only_fast_path_keep_first(
                left_index=left_index,
                right_index=right_index,
                left_indices=np.empty(matches, dtype=np.intp),
                right_indices=np.empty(matches, dtype=np.intp),
                starts=starts,
                ends=ends,
                booleans=booleans,
            )
        )
    # keep=='last'
    return cond_join.build_indices_equi_range_join_only_fast_path_keep_last(
        left_index=left_index,
        right_index=right_index,
        left_indices=np.empty(matches, dtype=np.intp),
        right_indices=np.empty(matches, dtype=np.intp),
        starts=starts,
        ends=ends,
        booleans=booleans,
    )


def _get_positive_matches_no_ranges(
    df: pd.DataFrame,
    right: pd.DataFrame,
    left_index: np.ndarray,
    right_index: np.ndarray,
    conditions: list,
):
    """
    Iterate through conditions
    and get positive matches.
    Applied to indices obtained from pd.merge
    or != only conditions
    """
    booleans = np.ones(left_index.size, dtype=np.int8)
    for left_on, right_on, op in conditions:
        # rethink this
        # let's not index here
        # but within cython
        left_arr = df.loc[left_index, left_on]
        is_extension_array = pd.api.types.is_extension_array_dtype(left_arr)
        left_arr = left_arr._values
        right_arr = right.loc[right_index, right_on]._values
        check = True
        if op == "!=":
            check = _can_pass_ne_to_cython(left=left_arr, right=right_arr)
            if not check:
                left_booleans = pd.isna(left_arr).astype(np.int8, copy=False)
                right_booleans = pd.isna(right_arr).astype(np.int8, copy=False)
            else:
                left_booleans = None
                right_booleans = None
        left_arr, right_arr = _convert_to_numpy(left=left_arr, right=right_arr)
        if check:
            booleans, count_exact_matches = (
                cond_join.get_positive_matches_no_ranges(
                    left_array=left_arr,
                    right_array=right_arr,
                    op=operator_mapping[op],
                    booleans=booleans,
                )
            )
        elif not check and is_extension_array:
            booleans, count_exact_matches = (
                cond_join.get_positive_matches_no_ranges_ne_pandas_array(
                    left_array=left_arr,
                    right_array=right_arr,
                    op=operator_mapping[op],
                    booleans=booleans,
                    left_booleans=left_booleans,
                    right_booleans=right_booleans,
                )
            )
        else:
            booleans, count_exact_matches = (
                cond_join.get_positive_matches_no_ranges_ne(
                    left_array=left_arr,
                    right_array=right_arr,
                    op=operator_mapping[op],
                    booleans=booleans,
                    left_booleans=left_booleans,
                    right_booleans=right_booleans,
                )
            )
        if not count_exact_matches:
            return None
    return left_index, right_index, booleans, count_exact_matches


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
    left_array: np.ndarray,
    right_array: np.ndarray,
    indices: dict,
    op: str,
):
    """
    Update `starts` or `ends` for non-equi
    """
    left_array, right_array = _convert_to_numpy(
        left=left_array, right=right_array
    )
    if op == ">":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_greater_than_strict(
                left_array=left_array,
                right_array=right_array,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif op == ">=":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_greater_than(
                left_array=left_array,
                right_array=right_array,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif op == "<":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_less_than_strict(
                left_array=left_array,
                right_array=right_array,
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                sizes=indices["sizes"],
            )
        )
    elif op == "<=":
        starts, ends, booleans, sizes, total, matches = (
            cond_join.update_search_indices_less_than(
                left_array=left_array,
                right_array=right_array,
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


def _update_search_indices_equi(
    left_array: np.ndarray,
    right_array: np.ndarray,
):
    """
    Update `starts` or `ends` for equi
    """
    length = left_array.size
    booleans = np.ones(length, dtype=np.int8)
    sizes = np.zeros(length, dtype=np.intp)
    starts = np.zeros(length, dtype=np.intp)
    ends = np.empty(length, dtype=np.intp)
    ends[:] = right_array.size
    left_array, right_array = _convert_to_numpy(
        left=left_array, right=right_array
    )
    starts, ends, booleans, sizes, total, matches = (
        cond_join.update_search_indices_strictly_equal_min(
            left_array=left_array,
            right_array=right_array,
            starts=starts,
            ends=ends,
            booleans=booleans,
            sizes=sizes,
        )
    )
    if matches == 0:
        return None
    starts, ends, booleans, sizes, total, matches = (
        cond_join.update_search_indices_strictly_equal_max(
            left_array=left_array,
            right_array=right_array,
            starts=starts,
            ends=ends,
            booleans=booleans,
            sizes=sizes,
        )
    )
    if matches == 0:
        return None
    indices = {
        "starts": starts,
        "ends": ends,
        "booleans": booleans,
        "total": total,
        "matches": matches,
        "sizes": sizes,
    }
    return indices


def _generate_tuples(df: pd.DataFrame, right: pd.DataFrame, conditions: list):
    """
    Build tuple of arrays to pass to numba
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
        if op == "!=":
            left_booleans = pd.isna(left_arr)
            right_booleans = pd.isna(right_arr)
        left_arr, right_arr = _convert_to_numpy(left=left_arr, right=right_arr)
        condition = (
            left_arr,
            right_arr,
            operator_mapping[op],
            is_extension_array,
            left_booleans,
            right_booleans,
        )
        tuples.append(condition)
    return tuple(tuples)
