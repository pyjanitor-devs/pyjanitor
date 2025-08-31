from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd
from pandas.core.reshape.merge import _MergeOperation

from janitor.cython_functions import cond_join

from . import helpers
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
    outcome = helpers._separate_conditions_based_on_op(
        conditions=conditions, keep_equals_separate=True
    )
    # get rid of nulls, if any
    df = helpers._remove_nulls_multiple_conditions(
        df=df, columns=outcome.pop("l_cols")
    )
    if df is None:
        return None
    right = helpers._remove_nulls_multiple_conditions(
        df=right, columns=outcome.pop("r_cols")
    )
    if right is None:
        return None
    if use_numba and not outcome["conditions"]:
        raise ValueError(
            "At least one non-equi join should be present if `use_numba=True`"
        )
    if use_numba:
        equals = outcome["equals"]
        is_fastpath_range_join = False
        if outcome.get("is_range_join") and (len(equals) == 1):
            _, col, _ = equals[0]
            sorter = {col: 1}
            ge_gt, le_lt, *conditions = outcome["conditions"]
            _, col, _ = ge_gt
            sorter[col] = 1
            _, col, _ = le_lt
            sorter[col] = 1
            sorter = [*sorter]
            right = right.sort_values(
                by=sorter, ignore_index=False, kind="stable"
            )
            grouper = outcome["equals"]
            grouper = grouper[0][1]
            grouped = right.groupby([grouper], sort=False, observed=True)
            le_lt = outcome["conditions"][1][1]
            grouped = grouped[le_lt]
            is_fastpath_range_join = grouped.is_monotonic_increasing.all()
        # is there any >/>=/</<=?
        elif outcome.get("less_than_or_greater_than") and (len(equals) == 1):
            _, col, _ = equals[0]
            sorter = {col: 1}
            (_, col, _), *conditions = outcome["conditions"]
            sorter[col] = 1
            right = right.sort_values(
                by=[*sorter], ignore_index=False, kind="stable"
            )
        elif not outcome.get("less_than_or_greater_than") or (len(equals) > 1):
            sorter = equals[0][1]
            if not right[sorter].is_monotonic_increasing:
                right = right.sort_values(
                    sorter, ignore_index=False, kind="stable"
                )
        return _numba_equi_join(
            df=df,
            right=right,
            is_fastpath_range_join=is_fastpath_range_join,
            conditions=outcome,
            row_count=row_count,
            keep=keep,
        )
    equals = outcome["equals"]
    if use_pandas_merge_for_equi_join:
        use_binary_search = False
    else:
        use_binary_search = _is_binary_search_appropriate(df=df, equals=equals)
    if not use_binary_search:
        left_on = []
        right_on = []
        for l_col, r_col, _ in equals:
            left_on.append(l_col)
            right_on.append(r_col)
        indices = _get_indices_from_pandas_merge(
            df=df, right=right, left_on=left_on, right_on=right_on
        )
        if indices is None:
            return None

        left_indices, right_indices = indices
        if not outcome["conditions"]:
            # patch based on updates in internal code
            # pandas/core/reshape/merge.py#L1692
            # for pandas 2.2
            if left_indices is None:
                left_indices = df.index._values
            else:
                left_indices = df.index._values[left_indices]
            if right_indices is None:
                right_indices = right.index._values
            else:
                right_indices = right.index._values[right_indices]
            return helpers._keep_output(
                keep=keep, left=left_indices, right=right_indices
            )
        # patch based on updates in internal code
        # pandas/core/reshape/merge.py#L1692
        # for pandas 2.2
        if left_indices is None:
            left_indices = np.arange(df.index.size, dtype=np.intp)
        if right_indices is None:
            right_indices = np.arange(right.index.size, dtype=np.intp)
        conditions = helpers._generate_tuples(
            df=df, right=right, conditions=outcome["conditions"]
        )
        matches = helpers._get_positive_matches_no_ranges(
            left_indices=left_indices,
            right_indices=right_indices,
            conditions=conditions,
        )
        if matches is None:
            return None
        if row_count:
            return helpers._get_row_counts_multiple_conditions_no_ranges(
                left_index=df.index._values,
                row_count=row_count,
                indices=left_indices,
                matches=matches,
            )
        return helpers._multiple_conditions_get_indices_no_ranges(
            left_index=df.index._values,
            right_index=right.index._values,
            left_indices=left_indices,
            right_indices=right_indices,
            matches=matches,
            keep=keep,
        )
    _, col, _ = equals[0]
    sorter = {col: 1}
    if outcome.get("is_range_join"):
        ge_gt, le_lt, *conditions = outcome["conditions"]
        _, col, _ = ge_gt
        sorter[col] = 1
        _, col, _ = le_lt
        sorter[col] = 1
        sorter = [*sorter]
        right = right.sort_values(by=sorter, ignore_index=False, kind="stable")
    # is there any >/>=/</<=?
    elif outcome.get("less_than_or_greater_than"):
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
    left_on, right_on, op = equals[0]
    indices = helpers._equal_indices(left=df[left_on], right=right[right_on])
    if indices is None:
        return None
    indices["sizes"] = indices["ends"] - indices["starts"]
    if return_ranges and not outcome.get("conditions"):
        starts = indices["starts"]
        ends = indices["ends"]
        left_index = indices["left_index"]
        right_index = indices["right_index"]
        booleans = indices["booleans"]
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
        booleans = indices["booleans"]
        sizes = indices["sizes"]
        if not booleans.all():
            sizes = np.where(booleans, sizes, 0)
        indices["total"] = sizes.sum()
        indices["matches"] = np.count_nonzero(booleans)
        indices["booleans"] = indices["booleans"].astype(np.int8, copy=False)
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
    indices["booleans"] = indices["booleans"].astype(np.int8, copy=False)
    if not outcome.get("less_than_or_greater_than"):
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
        if row_count:
            return pd.Series(
                index=indices["left_index"],
                data=indices["counts_array"],
                name=row_count,
            )
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
        if row_count and not conditions:
            return pd.Series(
                index=indices["left_index"],
                data=indices["sizes"],
                name=row_count,
            )
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
        if row_count:
            return pd.Series(
                index=indices["left_index"],
                data=indices["counts_array"],
                name=row_count,
            )
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
    if row_count and not conditions:
        return pd.Series(
            index=indices["left_index"],
            data=indices["sizes"],
            name=row_count,
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
    if row_count:
        return pd.Series(
            index=indices["left_index"],
            data=indices["counts_array"],
            name=row_count,
        )
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
    if len(equals) > 1:
        return False
    for left_on, _, _ in equals:
        series = df[left_on]
        if (
            not pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_datetime64_dtype(series)
            and not pd.api.types.is_timedelta64_dtype(series)
        ):
            return False
    return True


def _get_indices_from_pandas_merge(
    df: pd.DataFrame, right: pd.DataFrame, left_on: list, right_on: list
) -> tuple | None:
    """
    Get indices from pandas merge
    """
    left_index, right_index = _MergeOperation(
        df,
        right,
        left_on=left_on,
        right_on=right_on,
        sort=False,
    )._get_join_indexers()
    if left_index is not None:
        if not left_index.size:
            return None
    return left_index, right_index
