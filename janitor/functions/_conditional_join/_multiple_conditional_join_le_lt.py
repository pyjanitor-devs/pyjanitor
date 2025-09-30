from __future__ import annotations

import numpy as np
import pandas as pd

from . import _aggs, _helpers
from ._numba_non_equi_join_multiple import _numba_multiple_non_equi_join


def _multiple_conditional_join_le_lt(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    use_numba: bool,
    return_ranges: bool,
    aggfunc: list[tuple] = None,
) -> tuple:
    """
    Get indices, or aggregates, for multiple conditions,
    where `>/>=` or `</<=` is present
    """
    if use_numba:
        return _numba_multiple_non_equi_join(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
        )
    outcome = _helpers._separate_conditions_based_on_op(conditions=conditions)
    booleans = _helpers._maybe_remove_nulls_from_dataframe(
        df=df, columns=outcome["l_cols"], return_bools=True
    )
    if booleans is None:
        return None
    # get rid of nulls, if any
    right = _helpers._maybe_remove_nulls_from_dataframe(
        df=right, columns=outcome["r_cols"]
    )
    if right is None:
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
    len_df = len(df)
    starts = np.empty(len_df, dtype=np.intp)
    ends = np.empty(len_df, dtype=np.intp)
    sizes = np.empty(len_df, dtype=np.intp)
    indices = {
        "left_index": df.index._values,
        "starts": starts,
        "ends": ends,
        "booleans": booleans.to_numpy(dtype=np.int8, copy=False),
        "sizes": sizes,
    }
    if not outcome.get("is_range_join"):
        right_on = outcome["conditions"][0][1]
        right_is_sorted = right[right_on].is_monotonic_increasing
        indices["right_is_sorted"] = right_is_sorted
        if not right_is_sorted:
            right = right.sort_values(
                right_on, kind="stable", ignore_index=False
            )
        (left_on, right_on, op), *conditions = outcome["conditions"]
        indices["conditions"] = conditions
        indices["right_index"] = right.index._values
        indices = _helpers._update_search_indices(
            left=df[left_on]._values,
            right=right[right_on]._values,
            indices=indices,
            op=op,
            first_time=True,
        )
        if indices is None:
            return None
        conditions = _helpers._generate_tuples(
            df=df, right=right, conditions=indices["conditions"]
        )
        indices = _helpers._get_positive_matches(
            indices=indices,
            conditions=conditions,
        )
        if indices is None:
            return None
        if aggfunc:
            df_index = indices["left_index"]
            # TODO: compute this only if `size` or `count` is in aggfunc
            # possibly limit if nulls?
            if not indices["booleans"].all():
                booleans = indices["booleans"].astype(np.bool_, copy=False)
                indices["counts_array"] = indices["counts_array"][booleans]
                df_index = df_index[booleans]
            results = _aggs.compute_aggfunc_result(
                aggfunc=aggfunc,
                agg_frame=right,
                indices=indices,
                total=indices["l_counts"],
            )
            return {"aggregates": results, "df_index": df_index}
        if keep == "all":
            total = indices["total"]
        else:
            total = indices["l_counts"]
        return _helpers._multiple_conditions_get_indices(
            left_index=indices["left_index"],
            right_index=indices["right_index"],
            starts=indices["starts"],
            ends=indices["ends"],
            booleans=indices["booleans"],
            sizes=indices["sizes"],
            matches=indices["matches"],
            keep=keep,
            total=total,
        )
    # range join
    ge_gt, le_lt, *conditions = outcome["conditions"]
    col1 = ge_gt[1]
    col2 = le_lt[1]
    check1 = right[col1].is_monotonic_increasing
    check2 = right[col2].is_monotonic_increasing
    right_is_sorted = all((check1, check2))
    if not right_is_sorted:
        sorter = {}
        sorter[col1] = 1
        sorter[col2] = 1
        sorter = [*sorter]
        right = right.sort_values(by=sorter, ignore_index=False, kind="stable")
    indices["right_index"] = right.index._values
    indices = _range_indices(
        df=df, right=right, first=ge_gt, second=le_lt, indices=indices
    )
    if indices is None:
        return None
    indices["right_is_sorted"] = right_is_sorted
    if condition := indices.get("condition"):
        conditions = [condition] + conditions
    if indices.get("fastpath") and not conditions and aggfunc:
        indices["counts_array"] = indices["sizes"]
        df_index = indices["left_index"]
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
            indices["counts_array"] = indices["counts_array"][booleans]
            df_index = df_index[booleans]
        results = _aggs.compute_aggfunc_result(
            aggfunc=aggfunc,
            agg_frame=right,
            indices=indices,
            total=indices["matches"],
        )
        return {"aggregates": results, "df_index": df_index}
    if indices.get("fastpath") and not conditions:
        if keep == "all":
            total = indices["total"]
        else:
            total = indices["matches"]
        return _helpers._build_indices_single_equi_or_true_range_join(
            left_index=indices["left_index"],
            right_index=indices["right_index"],
            starts=indices["starts"],
            ends=indices["ends"],
            right_is_sorted=indices.get("right_is_sorted"),
            return_ranges=return_ranges,
            total=total,
            keep=keep,
            booleans=indices["booleans"],
        )
    conditions = _helpers._generate_tuples(
        df=df, right=right, conditions=conditions
    )
    indices = _helpers._get_positive_matches(
        indices=indices,
        conditions=conditions,
    )
    if indices is None:
        return None
    if aggfunc:
        df_index = indices["left_index"]
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
            indices["counts_array"] = indices["counts_array"][booleans]
            df_index = df_index[booleans]
        results = _aggs.compute_aggfunc_result(
            aggfunc=aggfunc,
            agg_frame=right,
            indices=indices,
            total=indices["l_counts"],
        )
        return {"aggregates": results, "df_index": df_index}
    if keep == "all":
        total = indices["total"]
    else:
        total = indices["l_counts"]
    return _helpers._multiple_conditions_get_indices(
        left_index=indices["left_index"],
        right_index=indices["right_index"],
        starts=indices["starts"],
        ends=indices["ends"],
        booleans=indices["booleans"],
        sizes=indices["sizes"],
        matches=indices["matches"],
        keep=keep,
        total=total,
    )


def _range_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    first: tuple,
    second: tuple,
    indices: dict,
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
    indices = _helpers._update_search_indices(
        left=df[left_on]._values,
        right=right[right_on]._values,
        indices=indices,
        op=op,
        first_time=True,
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
    indices = _helpers._update_search_indices(
        left=left_c._values,
        right=right_c._values,
        indices=indices,
        op=op,
    )
    if indices is None:
        return None
    indices["fastpath"] = fastpath
    if not fastpath:
        indices["condition"] = second
    return indices
