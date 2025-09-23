from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd

from janitor.cython_functions import (
    cond_join,
    cond_join_aggs,
    cond_join_indices,
)

from . import aggs, helpers
from .multiple_conditional_join_le_lt import _multiple_conditional_join_le_lt
from .numba_equi_join import _numba_equi_join


def _multiple_conditional_join_eq(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    use_numba: bool,
    force: bool,
    return_ranges: bool,
    row_count: Hashable | None,
    use_pandas_merge_for_equi_join: bool,
    aggfunc: list[tuple],
) -> tuple:
    """
    Get indices for multiple conditions,
    if any of the conditions has an `==` operator.

    Returns a tuple of (left_index, right_index)
    """

    if force:
        return _multiple_conditional_join_le_lt(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            use_numba=use_numba,
            return_ranges=False,
            row_count=row_count,
        )

    if use_numba:
        return _numba_equi_join(
            df=df,
            right=right,
            conditions=conditions,
            row_count=row_count,
            keep=keep,
        )
    outcome = helpers._separate_conditions_based_on_op(
        conditions=conditions, keep_equals_separate=True
    )
    booleans = helpers._maybe_remove_nulls_from_dataframe(
        df=df, columns=outcome.get("l_cols"), return_bools=True
    )
    if booleans is None:
        return None
    right = helpers._maybe_remove_nulls_from_dataframe(
        df=right, columns=outcome.get("r_cols")
    )
    if right is None:
        return None
    equals = outcome["equals"]
    if use_pandas_merge_for_equi_join:
        left_on = []
        right_on = []
        for l_col, r_col, _ in equals:
            l_val = df[l_col].array
            r_val = right[r_col].array
            left_on.append(l_val)
            right_on.append(r_val)
        if len(left_on) > 1:
            left_on = pd.MultiIndex.from_arrays(left_on)
            right_on = pd.MultiIndex.from_arrays(right_on)
        else:
            left_on = pd.Index(left_on[0])
            right_on = pd.Index(right_on[0])
        try:
            indexers = right_on.get_indexer(left_on)
            bools = indexers == -1
            if bools.all():
                return None
            if (not booleans.all()) or bools.any():
                bools = ~bools
                booleans = booleans.astype(np.bool_, copy=False)
                booleans = booleans & bools
            if not outcome["conditions"] and aggfunc:
                if not booleans.all():
                    booleans = booleans.astype(np.bool_, copy=False)
                    df_index = df.index._values[booleans]
                    indexers = indexers[booleans]
                    booleans = booleans[booleans]
                else:
                    df_index = df.index._values
                results = aggs.compute_aggfunc_result_no_ranges(
                    aggfunc=aggfunc,
                    agg_frame=right,
                    booleans=booleans,
                    indexers=indexers,
                )
                return {"aggregates": results, "df_index": df_index}
            if not outcome["conditions"] and not booleans.all():
                left_index = df.index._values[booleans]
                indexers = indexers[booleans]
                right_index = right.index._values[indexers]
                return left_index, right_index
            if not outcome["conditions"]:
                left_index = df.index._values
                right_index = right.index._values[indexers]
                return left_index, right_index

            booleans = booleans.astype(np.int8, copy=False)
            conditions = helpers._generate_tuples(
                df=df, right=right, conditions=outcome["conditions"]
            )
            booleans = helpers._get_positive_matches_no_ranges(
                right_index=indexers, conditions=conditions, booleans=booleans
            )
            if booleans is None:
                return None
            if aggfunc:
                if not booleans.all():
                    booleans = booleans.astype(np.bool_, copy=False)
                    df_index = df.index._values[booleans]
                    indexers = indexers[booleans]
                    booleans = booleans[booleans]
                else:
                    df_index = df.index._values
                results = aggs.compute_aggfunc_result_no_ranges(
                    aggfunc=aggfunc,
                    agg_frame=right,
                    booleans=booleans,
                    indexers=indexers,
                )
                return {"aggregates": results, "df_index": df_index}
            if (keep == "all") and not booleans.all():
                booleans = booleans.astype(np.bool_, copy=False)
                left_index = df.index._values[booleans]
                indexers = indexers[booleans]
                right_index = right.index._values[indexers]
                return left_index, right_index
            if keep == "all":
                left_index = df.index._values
                right_index = right.index._values[indexers]
                return left_index, right_index
            if keep == "first":
                return cond_join_indices.build_indices_no_ranges_keep_first(
                    left_index=df.index._values,
                    right_index=right.index._values,
                    indexers=indexers,
                    matches=booleans,
                )
            return cond_join_indices.build_indices_no_ranges_keep_last(
                left_index=df.index._values,
                right_index=right.index._values,
                indexers=indexers,
                matches=booleans,
            )
        except pd.errors.InvalidIndexError:
            positions, right_on = right_on.factorize()
            indexers = right_on.get_indexer(left_on)
            bools = indexers == -1
            if bools.all():
                return None
            bools = None
            starts, ends, r_sizes, positions = cond_join.reorder_positions(
                len_uniques=right_on.size, positions=positions
            )
            sizes, booleans = (
                cond_join_aggs.get_row_counts_from_ranges_positions(
                    booleans=booleans, indexers=indexers, sizes=r_sizes
                )
            )
            if not outcome["conditions"] and aggfunc:
                indices = {
                    "starts": starts,
                    "ends": ends,
                    "sizes": sizes,
                    "positions": positions,
                    "booleans": booleans,
                    "indexers": indexers,
                    "counts_array": sizes,
                }
                if not booleans.all():
                    booleans = booleans.astype(np.bool_, copy=False)
                    indices["counts_array"] = indices["sizes"][booleans]
                    df_index = df.index._values[booleans]
                else:
                    indices["counts_array"] = indices["sizes"]
                    df_index = df.index._values
                results = aggs.compute_aggfunc_result(
                    aggfunc=aggfunc,
                    agg_frame=right,
                    indices=indices,
                    total=booleans.sum(),
                )
                return {"aggregates": results, "df_index": df_index}
            if not outcome["conditions"]:
                total = sizes.sum()
                return cond_join_indices.build_indices_from_ranges_positions(
                    booleans=booleans,
                    indexers=indexers,
                    starts=starts,
                    ends=ends,
                    positions=positions,
                    index_right=right.index._values,
                    left_index=np.empty(total, dtype=np.intp),
                    right_index=np.empty(total, dtype=np.intp),
                )
            conditions = helpers._generate_tuples(
                df=df, right=right, conditions=outcome["conditions"]
            )
            indices = {
                "starts": starts,
                "ends": ends,
                "sizes": sizes,
                "positions": positions,
                "booleans": booleans,
                "indexers": indexers,
            }
            indices = helpers._get_positive_matches_ranges_positions(
                indices=indices, conditions=conditions
            )
            if indices is None:
                return None
            if aggfunc:
                if not booleans.all():
                    booleans = indices["booleans"].astype(np.bool_, copy=False)
                    indices["counts_array"] = indices["counts_array"][booleans]
                    df_index = df.index._values[booleans]
                else:
                    df_index = df.index._values
                results = aggs.compute_aggfunc_result(
                    aggfunc=aggfunc,
                    agg_frame=right,
                    indices=indices,
                    total=indices["l_counts"],
                )
                return {"aggregates": results, "df_index": df_index}
            if keep == "all":
                return cond_join_indices.build_indices_matches_positions_all(
                    booleans=indices["booleans"],
                    matches=indices["matches"],
                    indexers=indices["indexers"],
                    sizes=indices["sizes"],
                    positions=indices["positions"],
                    starts=indices["starts"],
                    ends=indices["ends"],
                    index_right=right.index._values,
                    left_index=np.empty(indices["total"], dtype=np.intp),
                    right_index=np.empty(indices["total"], dtype=np.intp),
                )
            if keep == "first":
                return cond_join_indices.build_indices_matches_positions_first(
                    booleans=indices["booleans"],
                    matches=indices["matches"],
                    indexers=indices["indexers"],
                    sizes=indices["sizes"],
                    positions=indices["positions"],
                    starts=indices["starts"],
                    ends=indices["ends"],
                    index_right=right.index._values,
                    left_index=np.empty(indices["l_counts"], dtype=np.intp),
                    right_index=np.empty(indices["l_counts"], dtype=np.intp),
                )
            return cond_join_indices.build_indices_matches_positions_last(
                booleans=indices["booleans"],
                matches=indices["matches"],
                indexers=indices["indexers"],
                sizes=indices["sizes"],
                positions=indices["positions"],
                starts=indices["starts"],
                ends=indices["ends"],
                index_right=right.index._values,
                left_index=np.empty(indices["l_counts"], dtype=np.intp),
                right_index=np.empty(indices["l_counts"], dtype=np.intp),
            )
    _is_binary_search_appropriate(df=df, equals=equals)
    equals, *rest = equals
    rest.extend(outcome["conditions"])
    outcome["conditions"] = rest
    outcome["equals"] = equals
    _, col, _ = equals
    sorter = {col: 1}
    if outcome.get("is_range_join") and (outcome.get("equi_count") == 1):
        ge_gt, le_lt, *conditions = outcome["conditions"]
        _, col, _ = ge_gt
        sorter[col] = 1
        _, col, _ = le_lt
        sorter[col] = 1
        sorter = [*sorter]
        right = right.sort_values(by=sorter, ignore_index=False, kind="stable")
    # is there any >/>=/</<=?
    elif outcome.get("non_equi_count") and (outcome.get("equi_count") == 1):
        (_, col, _), *conditions = outcome["conditions"]
        sorter[col] = 1
        right = right.sort_values(
            by=[*sorter], ignore_index=False, kind="stable"
        )
    else:
        sorter = [*sorter]
        sorter = sorter[0]
        if not right[sorter].is_monotonic_increasing:
            right = right.sort_values(
                by=sorter, ignore_index=False, kind="stable"
            )
    left_on, right_on, op = equals
    indices = helpers._equal_indices(left=df[left_on], right=right[right_on])
    if indices is None:
        return None
    booleans = indices["booleans"] & booleans.astype(np.bool_, copy=False)
    indices["sizes"] = indices["ends"] - indices["starts"]
    indices["booleans"] = booleans.astype(np.int8, copy=False)
    if aggfunc and not outcome.get("conditions"):
        if not booleans.all():
            indices["counts_array"] = indices["sizes"][booleans]
            df_index = indices["left_index"][booleans]
        else:
            indices["counts_array"] = indices["sizes"]
            df_index = indices["left_index"]
        results = aggs.compute_aggfunc_result(
            aggfunc=aggfunc,
            agg_frame=right,
            indices=indices,
            total=booleans.sum(),
        )
        return {"aggregates": results, "df_index": df_index}
    if return_ranges and not outcome.get("conditions"):
        starts = indices["starts"]
        ends = indices["ends"]
        left_index = indices["left_index"]
        right_index = indices["right_index"]
        if not booleans.all():
            starts = starts[booleans]
            ends = ends[booleans]
            left_index = left_index[booleans]
        return {
            "left_index": left_index,
            "right_index": right_index,
            "starts": starts,
            "ends": ends,
        }
    if not outcome.get("conditions"):
        sizes = indices["sizes"]
        if not booleans.all():
            sizes = np.where(booleans, sizes, 0)
        indices["total"] = sizes.sum()
        indices["matches"] = np.count_nonzero(booleans)
        return helpers._build_indices_fast_path_range_join_only(
            left_index=indices["left_index"],
            right_index=indices["right_index"],
            starts=indices["starts"],
            ends=indices["ends"],
            booleans=indices["booleans"],
            keep=keep,
            total=indices["total"],
            matches=indices["matches"],
        )
    # != only
    if not outcome.get("non_equi_count") or (outcome.get("equi_count") > 1):
        conditions = outcome["conditions"]
        conditions = helpers._generate_tuples(
            df=df, right=right, conditions=conditions
        )
        indices = helpers._get_positive_matches(
            indices=indices,
            conditions=conditions,
        )
        if indices is None:
            return None
        if aggfunc:
            if not indices["booleans"].all():
                booleans = indices["booleans"].astype(np.bool_, copy=False)
                indices["counts_array"] = indices["counts_array"][booleans]
                df_index = indices["left_index"][booleans]
            else:
                df_index = indices["left_index"]
            results = aggs.compute_aggfunc_result(
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
        return helpers._multiple_conditions_get_indices(
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
    # range join only
    is_fastpath_range_join = False
    if outcome.get("is_range_join"):
        # we already know that ge_gt is increasing monotonic,
        # (we sorted on both eq and ge_gt)
        # we do need to check le_lt though and see if
        # we can steal some perf. there for a true range join
        # if it is, then we can use a binary search
        # to skip non matched entries
        left_on, right_on, _ = outcome["conditions"][1]
        _, arr = helpers._convert_to_numpy(
            left=df[left_on]._values, right=right[right_on]._values
        )
        is_fastpath_range_join = cond_join.check_monotonicity_per_range(
            starts=indices["starts"],
            ends=indices["ends"],
            arr=arr,
            booleans=indices["booleans"],
        )
        is_fastpath_range_join = bool(is_fastpath_range_join)
    if is_fastpath_range_join:
        ge_gt, le_lt, *conditions = outcome["conditions"]
        left_on, right_on, op = ge_gt
        left_array = df[left_on]._values
        right_array = right[right_on]._values
        indices = helpers._update_search_indices(
            left=df[left_on]._values,
            right=right[right_on]._values,
            indices=indices,
            op=op,
        )
        if indices is None:
            return None
        left_on, right_on, op = le_lt
        indices = helpers._update_search_indices(
            left=df[left_on]._values,
            right=right[right_on]._values,
            indices=indices,
            op=op,
        )
        if indices is None:
            return None
        if aggfunc and not conditions:
            if not indices["booleans"].all():
                booleans = indices["booleans"].astype(np.bool_, copy=False)
                indices["counts_array"] = indices["sizes"][booleans]
                df_index = indices["left_index"][booleans]
            else:
                indices["counts_array"] = indices["sizes"]
                df_index = indices["left_index"]
            results = aggs.compute_aggfunc_result(
                aggfunc=aggfunc,
                agg_frame=right,
                indices=indices,
                total=indices["matches"],
            )
            return {"aggregates": results, "df_index": df_index}
        if return_ranges and not conditions:
            starts = indices["starts"]
            ends = indices["ends"]
            left_index = indices["left_index"]
            right_index = indices["right_index"]
            booleans = indices["booleans"]
            if not booleans.all():
                booleans = booleans.astype(np.bool_, copy=False)
                starts = starts[booleans]
                ends = ends[booleans]
                left_index = left_index[booleans]
            return {
                "left_index": left_index,
                "right_index": right_index,
                "starts": starts,
                "ends": ends,
            }
        if not conditions:
            return helpers._build_indices_fast_path_range_join_only(
                left_index=indices["left_index"],
                right_index=indices["right_index"],
                starts=indices["starts"],
                ends=indices["ends"],
                booleans=indices["booleans"],
                keep=keep,
                total=indices["total"],
                matches=indices["matches"],
            )
        conditions = outcome["conditions"]
        conditions = helpers._generate_tuples(
            df=df, right=right, conditions=conditions
        )
        indices = helpers._get_positive_matches(
            indices=indices,
            conditions=conditions,
        )
        if indices is None:
            return None
        if aggfunc:
            if not indices["booleans"].all():
                booleans = indices["booleans"].astype(np.bool_, copy=False)
                indices["counts_array"] = indices["counts_array"][booleans]
                df_index = indices["left_index"][booleans]
            else:
                df_index = indices["left_index"]
            results = aggs.compute_aggfunc_result(
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
        return helpers._multiple_conditions_get_indices(
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
    # no range join, but at least one </<=/>/>= is present
    (left_on, right_on, op), *conditions = outcome["conditions"]
    left_array = df[left_on]._values
    right_array = right[right_on]._values
    indices = helpers._update_search_indices(
        left=left_array, right=right_array, indices=indices, op=op
    )
    if indices is None:
        return None
    if aggfunc and not conditions:
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
            indices["counts_array"] = indices["sizes"][booleans]
            df_index = df.index[booleans]
        else:
            df_index = df.index
        return helpers.compute_aggfunc_result(
            aggfunc=aggfunc,
            agg_frame=right,
            indices=indices,
            df_index=df_index,
            total=indices["matches"],
        )
    if not conditions:
        if keep == "all":
            total = indices["total"]
        else:
            total = indices["l_counts"]
        return helpers._build_indices_fast_path_range_join_only(
            left_index=indices["left_index"],
            right_index=indices["right_index"],
            starts=indices["starts"],
            ends=indices["ends"],
            booleans=indices["booleans"],
            keep=keep,
            total=indices["total"],
            matches=indices["matches"],
        )
    conditions = helpers._generate_tuples(
        df=df, right=right, conditions=conditions
    )
    indices = helpers._get_positive_matches(
        indices=indices,
        conditions=conditions,
    )
    if indices is None:
        return None
    if aggfunc:
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
            indices["counts_array"] = indices["counts_array"][booleans]
            df_index = indices["left_index"][booleans]
        else:
            df_index = indices["left_index"]
        results = aggs.compute_aggfunc_result(
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
    return helpers._multiple_conditions_get_indices(
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


def _is_binary_search_appropriate(df: pd.DataFrame, equals: list) -> bool:
    """
    Check if it is appropriate
    to use a binary search approach
    on the equality condition
    """
    for left_on, _, _ in equals:
        series = df[left_on]
        if (
            not pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_datetime64_dtype(series)
            and not pd.api.types.is_timedelta64_dtype(series)
        ):
            raise ValueError(
                "binary search is supported only "
                "for numeric, datetime and timedelta dtypes."
            )
    return True
