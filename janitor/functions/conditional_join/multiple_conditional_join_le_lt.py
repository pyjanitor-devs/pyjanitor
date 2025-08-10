import itertools
from typing import Hashable

import numpy as np
import pandas as pd

from . import helpers
from .numba_non_equi_join_multiple import _numba_multiple_non_equi_join
from .numba_non_equi_join_single import _numba_single_non_equi_join


def _multiple_conditional_join_le_lt(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    use_numba: bool,
    return_ranges: bool,
    row_count: Hashable = None,
) -> tuple:
    """
    Get indices for multiple conditions,
    where `>/>=` or `</<=` is present,
    and there is no `==` operator.

    Returns a tuple of (df_index, right_index)
    """
    if use_numba:
        gt_lt = [
            condition
            for condition in conditions
            if condition[-1]
            in helpers.less_than_join_types.union(
                helpers.greater_than_join_types
            )
        ]
        conditions = [
            condition for condition in conditions if condition not in gt_lt
        ]
        if len(gt_lt) > 1:
            first_two = [op for *_, op in gt_lt[:2]]
            range_join_ops = itertools.product(
                helpers.less_than_join_types, helpers.greater_than_join_types
            )
            range_join_ops = map(set, range_join_ops)
            is_range_join = set(first_two) in range_join_ops
            if is_range_join and (
                first_two[0] in helpers.less_than_join_types
            ):
                gt_lt = [gt_lt[1], gt_lt[0], *gt_lt[2:]]
            gt_lt.extend(conditions)
            indices = _numba_multiple_non_equi_join(
                df,
                right,
                gt_lt,
                keep=keep,
                is_range_join=is_range_join,
                row_count=(row_count if row_count else None),
            )
            return indices
        else:
            left_on, right_on, op = gt_lt[0]
            indices = _numba_single_non_equi_join(
                left=df[left_on],
                right=right[right_on],
                op=op,
                keep="all",
                row_count=None,
            )
        if indices is None:
            return None
    # there is an opportunity for optimization for range joins
    # which is usually `lower_value < value < upper_value`
    # or `lower_value < a` and `b < upper_value`
    # intervalindex is not used here, as there are scenarios
    # where there will be overlapping intervals;
    # intervalindex does not offer an efficient way to get
    # the indices for overlaps
    # also, intervalindex covers only the first option
    # i.e => `lower_value < value < upper_value`
    # it does not extend to range joins for different columns
    # i.e => `lower_value < a` and `b < upper_value`
    # the option used for range joins is a simple form
    # dependent on sorting and extensible to overlaps
    # as well as the second option:
    # i.e =>`lower_value < a` and `b < upper_value`
    # range joins are also the more common types of non-equi joins
    # the other joins do not have an optimisation opportunity
    # within this space, as far as I know,
    # so a blowup of all the rows is unavoidable.

    # first step is to get two conditions, if possible
    # where one has a less than operator
    # and the other has a greater than operator
    # get the indices from that
    # and then build the remaining indices,
    # using _generate_indices function
    # the aim of this for loop is to see if there is
    # the possibility of a range join, and if there is,
    # then use the optimised path
    outcome = helpers._separate_conditions_based_on_op(conditions=conditions)
    # get rid of nulls, if any
    df = helpers._remove_nulls_multiple_conditions(
        df=df, columns=outcome["l_cols"]
    )
    if df is None:
        return None
    right = helpers._remove_nulls_multiple_conditions(
        df=right, columns=outcome["r_cols"]
    )
    if right is None:
        return None
    if not outcome.get("is_range_join"):
        indices = _get_indices_no_range_join(
            df=df,
            right=right,
            conditions=outcome["conditions"],
        )
        if indices is None:
            return None
        indices["sizes"] = indices["ends"] - indices["starts"]
        indices = helpers._build_start_indices(indices=indices)
        booleans = np.ones(indices["sizes"].size, dtype=np.int8)
        indices = helpers._get_positive_matches(
            df=df,
            right=right,
            indices=indices,
            conditions=indices["conditions"],
            booleans=booleans,
        )
        if indices is None:
            return None
        if row_count:
            return pd.Series(
                index=indices["left_index"],
                data=indices["counts_array"],
                name=row_count,
            )
        return helpers._multiple_conditions_get_indices(
            left_index=indices["left_index"],
            right_index=indices["right_index"],
            starts=indices["starts"],
            start_indices=indices["start_indices"],
            booleans=indices["booleans"],
            sizes=indices["sizes"],
            counts_array=indices["counts_array"],
            matches=indices["matches"],
            keep=keep,
        )
    # range join
    ge_gt, le_lt, *conditions = outcome["conditions"]
    right_is_sorted = False
    check1 = right[ge_gt[1]].is_monotonic_increasing
    if check1:
        right_is_sorted = True
    check2 = right[le_lt[1]].is_monotonic_increasing
    if not all((check1, check2)):
        sorter = {}
        sorter[ge_gt[1]] = 1
        sorter[le_lt[1]] = 1
        sorter = [*sorter]
        right = right.sort_values(by=sorter, ignore_index=False, kind="stable")
    indices = _range_indices(
        df=df,
        right=right,
        first=ge_gt,
        second=le_lt,
    )
    if indices is None:
        return None
    indices["right_is_sorted"] = right_is_sorted
    if condition := indices.get("condition"):
        conditions = [condition] + conditions
    if indices.get("fastpath") and not conditions:
        return _get_indices_fastpath_range_joins_dual(
            left_index=indices["left_index"],
            right_index=indices["right_index"],
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            booleans=indices["booleans"],
            right_is_sorted=indices.get("right_is_sorted"),
            row_count=row_count,
            return_ranges=return_ranges,
            total=indices["total"],
            matches=indices["matches"],
            keep=keep,
        )
    indices = helpers._build_start_indices(indices=indices)
    booleans = indices["booleans"].astype(np.int8, copy=False)
    indices = helpers._get_positive_matches(
        df=df,
        right=right,
        indices=indices,
        conditions=conditions,
        booleans=booleans,
    )
    if indices is None:
        return None
    if row_count:
        return pd.Series(
            index=indices["left_index"],
            data=indices["counts_array"],
            name=row_count,
        )
    return helpers._multiple_conditions_get_indices(
        left_index=indices["left_index"],
        right_index=indices["right_index"],
        starts=indices["starts"],
        start_indices=indices["start_indices"],
        booleans=indices["booleans"],
        sizes=indices["sizes"],
        counts_array=indices["counts_array"],
        matches=indices["matches"],
        keep=keep,
    )


def _get_indices_no_range_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Get outcome when there is no range join
    """
    (left_on, right_on, op), *conditions = conditions
    left_c = df[left_on]
    left_index = left_c.index._values
    right_c = right[right_on]
    right_c, _ = helpers._sort_if_not_monotonic(series=right[right_on])
    if op in helpers.less_than_join_types:
        outcome = helpers._less_than_indices(
            left=left_c.array,
            left_index=left_index,
            right=right_c.array,
            strict=op == "<",
        )
        if outcome is None:
            return None
        left_index, starts = outcome
        ends = np.empty(left_index.size, dtype=np.intp)
        ends[:] = right_c.size
    else:
        outcome = helpers._greater_than_indices(
            left=left_c.array,
            left_index=left_index,
            right=right_c.array,
            strict=op == ">",
        )
        if outcome is None:
            return None
        left_index, ends = outcome
        starts = np.zeros(left_index.size, dtype=np.intp)
    return {
        "left_index": left_index,
        "right_index": right_c.index._values,
        "starts": starts,
        "ends": ends,
        "conditions": conditions,
    }


def _get_indices_fastpath_range_joins_dual(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    sizes: np.ndarray,
    booleans: np.ndarray,
    right_is_sorted: bool,
    row_count: Hashable,
    return_ranges: bool,
    total: int,
    matches: int,
    keep: str,
) -> tuple[np.ndarray, np.ndarray] | pd.Series | dict:
    """
    Get indices if both right columns are sorted,
    and the number of join conditions is 2
    """
    if row_count:
        return pd.Series(
            index=left_index,
            data=sizes,
            name=row_count,
        )
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
    return helpers._build_indices_fast_path_range_join_only(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        booleans=booleans,
        keep=keep,
        total=total,
        matches=matches,
    )


def _range_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    first: tuple,
    second: tuple,
) -> dict | None:
    """
    Retrieve index positions for range/interval joins.

    Idea inspired by article:
    https://www.vertica.com/blog/what-is-a-range-join-and-why-is-it-so-fastba-p223413/

    Returns a tuple of (left_index, right_index)
    """
    # summary of code for range join:
    # get the positions where start_left is >/>= start_right
    # then within the positions,
    # get the positions where end_left is </<= end_right
    # this should reduce the search space
    left_on, right_on, op = first
    left_c = df[left_on]
    right_c = right[right_on]
    length = len(df)
    starts = np.zeros(length, dtype=np.intp)
    ends = np.empty(length, dtype=np.intp)
    ends[:] = len(right)
    booleans = np.ones(length, dtype=np.int8)
    sizes = np.zeros(length, dtype=np.intp)
    indices = dict(
        left_index=df.index._values,
        right_index=right.index._values,
        starts=starts,
        ends=ends,
        sizes=sizes,
        booleans=booleans,
    )
    indices = helpers._update_search_indices(
        left_array=df[left_on]._values,
        right_array=right[right_on]._values,
        indices=indices,
        op=op,
    )
    if indices is None:
        return None
    left_on, right_on, op = second
    right_c = right[right_on]
    left_c = df[left_on]
    # if True, we can use a binary search
    # for more performance, instead of a linear search
    fastpath = right_c.is_monotonic_increasing
    if not fastpath:
        right_c = right_c.cummax()
    indices = helpers._update_search_indices(
        left_array=left_c._values,
        right_array=right_c._values,
        indices=indices,
        op=op,
    )
    if indices is None:
        return None
    indices["fastpath"] = fastpath
    if not fastpath:
        indices["condition"] = second
    return indices
