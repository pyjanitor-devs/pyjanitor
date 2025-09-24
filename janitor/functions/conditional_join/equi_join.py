from __future__ import annotations

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
    sort_equi_join: bool,
    aggfunc: list[tuple],
) -> tuple | dict | None:
    """
    Get indices, or aggregates, for multiple conditions,
    if any of the conditions has an `==` operator.
    """

    if force:
        return _multiple_conditional_join_le_lt(
            df=df,
            right=right,
            conditions=conditions,
            keep=keep,
            use_numba=use_numba,
            return_ranges=False,
        )

    if use_numba:
        return _numba_equi_join(
            df=df,
            right=right,
            conditions=conditions,
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
    if not sort_equi_join:
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
            booleans = booleans & (indexers != -1)
            if not booleans.any():
                return None
            if outcome["conditions"]:
                booleans = booleans.astype(np.int8, copy=False)
                conditions = helpers._generate_tuples(
                    df=df, right=right, conditions=outcome["conditions"]
                )
                booleans = helpers._get_positive_matches_no_ranges(
                    right_index=indexers,
                    conditions=conditions,
                    booleans=booleans,
                )
                if booleans is None:
                    return None
            if aggfunc:
                df_index = df.index._values
                if not booleans.all():
                    booleans = booleans.astype(np.bool_, copy=False)
                    df_index = df_index[booleans]
                    indexers = indexers[booleans]
                results = aggs.compute_aggfunc_result_no_ranges(
                    aggfunc=aggfunc,
                    agg_frame=right,
                    indexers=indexers,
                )
                return {"aggregates": results, "df_index": df_index}
            left_index = df.index._values
            if not booleans.all():
                booleans = booleans.astype(np.bool_, copy=False)
                left_index = left_index[booleans]
                indexers = indexers[booleans]
                right_index = right.index._values[indexers]
                return left_index, right_index
            right_index = right.index._values[indexers]
            return left_index, right_index
        except pd.errors.InvalidIndexError:
            positions, right_on = right_on.factorize()
            indexers = right_on.get_indexer(left_on)
            booleans = booleans & (indexers != -1)
            if not booleans.any():
                return None
            starts, ends, r_sizes, positions = cond_join.reorder_positions(
                len_uniques=right_on.size, positions=positions
            )
            booleans = booleans.astype(np.int8, copy=False)
            sizes, booleans = (
                cond_join_aggs.get_row_counts_from_ranges_positions(
                    booleans=booleans, indexers=indexers, sizes=r_sizes
                )
            )
            indices = {
                "starts": starts,
                "ends": ends,
                "sizes": sizes,
                "positions": positions,
                "booleans": booleans,
                "indexers": indexers,
                "counts_array": sizes,
                "total": sizes.sum(),
                "l_counts": booleans.sum(),
            }
            if outcome["conditions"]:
                conditions = helpers._generate_tuples(
                    df=df, right=right, conditions=outcome["conditions"]
                )
                indices = helpers._get_positive_matches_ranges_positions(
                    indices=indices, conditions=conditions
                )
                if indices is None:
                    return None
            if aggfunc:
                df_index = df.index._values
                if not booleans.all():
                    booleans = indices["booleans"].astype(np.bool_, copy=False)
                    indices["counts_array"] = indices["counts_array"][booleans]
                    df_index = df_index[booleans]
                results = aggs.compute_aggfunc_result(
                    aggfunc=aggfunc,
                    agg_frame=right,
                    indices=indices,
                    total=indices["l_counts"],
                )
                return {"aggregates": results, "df_index": df_index}
            if (keep == "all") and (indices.get("matches", None) is None):
                return cond_join_indices.build_indices_ranges_positions_all(
                    booleans=indices["booleans"],
                    indexers=indices["indexers"],
                    positions=indices["positions"],
                    starts=indices["starts"],
                    ends=indices["ends"],
                    index_right=right.index._values,
                    left_index=np.empty(indices["total"], dtype=np.intp),
                    right_index=np.empty(indices["total"], dtype=np.intp),
                )
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
            if (keep == "first") and (indices.get("matches", None) is None):
                return cond_join_indices.build_indices_ranges_positions_first(
                    booleans=indices["booleans"],
                    indexers=indices["indexers"],
                    positions=indices["positions"],
                    starts=indices["starts"],
                    ends=indices["ends"],
                    index_right=right.index._values,
                    left_index=np.empty(indices["l_counts"], dtype=np.intp),
                    right_index=np.empty(indices["l_counts"], dtype=np.intp),
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
            if (keep == "last") and (indices.get("matches", None) is None):
                return cond_join_indices.build_indices_ranges_positions_last(
                    booleans=indices["booleans"],
                    indexers=indices["indexers"],
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
    left_on = []
    right_on = []
    for l_col, r_col, _ in equals:
        l_val = df[l_col].array
        r_val = right[r_col].array
        left_on.append(l_val)
        right_on.append(r_val)
    if outcome.get("non_equi_count"):
        (_, r_col, _), *_ = outcome["conditions"]
        r_val = right[r_col].array
        right_on.append(r_val)
    if outcome.get("is_range_join"):
        _, (_, r_col, _), *_ = outcome["conditions"]
        r_val = right[r_col].array
        right_on.append(r_val)
    if len(left_on) > 1:
        left_on = pd.MultiIndex.from_arrays(left_on)
    else:
        left_on = pd.Index(left_on[0])
    if len(right_on) > 1:
        right_on = pd.MultiIndex.from_arrays(right_on)
    else:
        right_on = pd.Index(right_on[0])
    right_is_sorted = right_on.is_monotonic_increasing
    if not right_is_sorted:
        indexer = right_on.argsort(kind="stable")
        right = right.iloc[indexer]
        right_on = right_on[indexer]
    # extract only the equi columns
    # to get the starts and ends
    if (len_equals := outcome.get("equi_count")) > 1:
        levels = right_on.levels[:len_equals]
        codes = right_on.codes[:len_equals]
        indexer = pd.MultiIndex(
            levels=levels, codes=codes, verify_integrity=False, copy=False
        )
    else:
        indexer = right_on.get_level_values(0)
    try:
        indexers = indexer.get_indexer(left_on)
        booleans = booleans & (indexers != -1)
        if not booleans.any():
            return None
        starts = np.arange(indexer.size, dtype=np.intp)
        ends = starts + 1
    except pd.errors.InvalidIndexError:
        positions, indexer = indexer.factorize()
        indexers = indexer.get_indexer(left_on)
        booleans = booleans & (indexers != -1)
        if not booleans.any():
            return None
        sizes = np.bincount(positions)
        ends = sizes.cumsum()
        starts = np.empty(sizes.size, dtype=np.intp)
        starts[0] = 0
        starts[1:] = ends[:-1]
    starts = starts[indexers]
    ends = ends[indexers]
    sizes = ends - starts
    if not booleans.all():
        sizes = np.where(booleans, sizes, 0)
    indices = {
        "starts": starts,
        "ends": ends,
        "booleans": booleans.astype(np.int8, copy=False),
        "sizes": sizes,
    }
    if aggfunc and not outcome.get("conditions"):
        df_index = df.index._values
        indices["counts_array"] = indices["sizes"]
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
            indices["counts_array"] = indices["counts_array"][booleans]
            df_index = df_index[booleans]
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
        left_index = df.index._values
        right_index = right.index._values
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
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
        return helpers._build_indices_fast_path_range_join_only(
            left_index=df.index._values,
            right_index=right.index._values,
            starts=indices["starts"],
            ends=indices["ends"],
            booleans=indices["booleans"],
            keep=keep,
            total=sizes.sum(),
            matches=np.count_nonzero(booleans),
        )

    # != only
    if not outcome.get("non_equi_count"):
        conditions = helpers._generate_tuples(
            df=df, right=right, conditions=outcome["conditions"]
        )
        indices = helpers._get_positive_matches(
            indices=indices,
            conditions=conditions,
        )
        if indices is None:
            return None
        if aggfunc:
            df_index = df.index._values
            if not indices["booleans"].all():
                booleans = indices["booleans"].astype(np.bool_, copy=False)
                indices["counts_array"] = indices["counts_array"][booleans]
                df_index = df_index[booleans]
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
            left_index=df.index._values,
            right_index=right.index._values,
            starts=indices["starts"],
            ends=indices["ends"],
            booleans=indices["booleans"],
            sizes=indices["sizes"],
            matches=indices["matches"],
            keep=keep,
            total=total,
        )

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

    (left_on, right_on, op), *conditions = outcome["conditions"]
    indices = helpers._update_search_indices(
        left=df[left_on]._values,
        right=right[right_on]._values,
        indices=indices,
        op=op,
    )
    if indices is None:
        return None
    if is_fastpath_range_join:
        (left_on, right_on, op), *conditions = conditions
        indices = helpers._update_search_indices(
            left=df[left_on]._values,
            right=right[right_on]._values,
            indices=indices,
            op=op,
        )
        if indices is None:
            return None
    if aggfunc and not conditions:
        df_index = df.index._values
        indices["counts_array"] = indices["sizes"]
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
            indices["counts_array"] = indices["counts_array"][booleans]
            df_index = df_index[booleans]
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
        left_index = df.index._values
        right_index = right.index._values
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
            left_index=df.index._values,
            right_index=right.index._values,
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
        df_index = df.index._values
        if not indices["booleans"].all():
            booleans = indices["booleans"].astype(np.bool_, copy=False)
            indices["counts_array"] = indices["counts_array"][booleans]
            df_index = df_index[booleans]
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
        left_index=df.index._values,
        right_index=right.index._values,
        starts=indices["starts"],
        ends=indices["ends"],
        booleans=indices["booleans"],
        sizes=indices["sizes"],
        matches=indices["matches"],
        keep=keep,
        total=total,
    )
