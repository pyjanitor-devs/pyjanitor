from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd
from pandas.core.reshape.merge import _MergeOperation

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
        left_index, right_index = indices
        if not outcome["conditions"]:
            return helpers._keep_output(
                keep=keep, left=left_index, right=right_index
            )
        indices = helpers._get_positive_matches_no_ranges(
            df=df,
            right=right,
            left_index=left_index,
            right_index=right_index,
            conditions=outcome["conditions"],
        )
        if indices is None:
            return None
        left_index, right_index, booleans, count_exact_matches = indices
        if count_exact_matches < left_index.size:
            booleans = booleans.astype(np.bool_, copy=False)
            left_index = left_index[booleans]
            right_index = right_index[booleans]
        if row_count:
            return (
                pd.Index(left_index).value_counts(sort=False).rename(row_count)
            )
        return helpers._keep_output(
            keep=keep, left=left_index, right=right_index
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
    left_on, right_on, _ = equals[0]
    indices = helpers._update_search_indices_equi(
        left_array=df[left_on]._values, right_array=right[right_on]._values
    )
    if indices is None:
        return None
    indices["left_index"] = df.index._values
    indices["right_index"] = right.index._values
    if return_ranges and not outcome.get("conditions"):
        return indices
    if not outcome.get("conditions"):
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
    if not outcome.get("less_than_or_greater_than"):
        indices = helpers._build_start_indices(indices=indices)
        indices = helpers._get_positive_matches(
            df=df,
            right=right,
            indices=indices,
            conditions=outcome["conditions"],
            booleans=indices["booleans"],
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
    # range join only
    is_fastpath_range_join = False
    max_size_is_1 = indices["sizes"].max() == 1
    if outcome.get("is_range_join") and not max_size_is_1:
        # we already know that ge_gt is increasing monotonic,
        # (we sorted on both eq and ge_gt)
        # we do need to check le_lt though and see if
        # we can steal some perf. there for a true range join
        # if it is, then we can use a binary search
        # to skip non matched entries
        # no point doing a binary search here
        # ideally it should be duplicated enough
        # to justify the check
        # for a max size of 1, a linear search will be much faster
        # TODO: instead of a max size of 1
        # should we set a miminum threshold for linear search?
        # maybe 50?100?500?
        grouper = outcome["equals"]
        grouper = grouper[0][1]
        grouped = right.groupby([grouper], sort=False, observed=True)
        le_lt = outcome["conditions"][1][1]
        grouped = grouped[le_lt]
        is_fastpath_range_join = grouped.is_monotonic_increasing.all()
    if is_fastpath_range_join:
        ge_gt, le_lt, *conditions = outcome["conditions"]
        left_on, right_on, op = ge_gt
        left_array = df[left_on]._values
        right_array = right[right_on]._values
        indices = helpers._update_search_indices(
            left_array=df[left_on]._values,
            right_array=right[right_on]._values,
            indices=indices,
            op=op,
        )
        if indices is None:
            return None
        left_on, right_on, op = le_lt
        indices = helpers._update_search_indices(
            left_array=df[left_on]._values,
            right_array=right[right_on]._values,
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
            return dict(
                left_index=left_index,
                right_index=right_index,
                starts=indices["starts"],
                ends=indices["ends"],
            )
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
        indices = helpers._build_start_indices(indices=indices)
        indices = helpers._get_positive_matches(
            df=df,
            right=right,
            indices=indices,
            conditions=conditions,
            booleans=indices["booleans"],
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
    # no range join, but at least one </<=/>/>= is present
    if max_size_is_1:
        indices = helpers._build_start_indices(indices=indices)
        indices = helpers._get_positive_matches(
            df=df,
            right=right,
            indices=indices,
            conditions=outcome["conditions"],
            booleans=indices["booleans"],
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
    (left_on, right_on, op), *conditions = outcome["conditions"]
    left_array = df[left_on]._values
    right_array = right[right_on]._values
    indices = helpers._update_search_indices(
        left_array=left_array, right_array=right_array, indices=indices, op=op
    )
    if indices is None:
        return None
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
    indices = helpers._build_start_indices(indices=indices)
    indices = helpers._get_positive_matches(
        df=df,
        right=right,
        indices=indices,
        conditions=conditions,
        booleans=indices["booleans"],
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


def _is_binary_search_appropriate(df: pd.DataFrame, equals: list) -> bool:
    """
    Check if it is appropriate
    to use a binary search approach
    on the equality condition
    """
    if len(equals) > 1:
        return False
    for left_on, *_ in equals:
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
        left_index = df.index._values[left_index]
    # patch based on updates in internal code
    # pandas/core/reshape/merge.py#L1692
    # for pandas 2.2
    elif left_index is None:
        left_index = df.index._values
    if right_index is not None:
        right_index = right.index._values[right_index]
    else:
        right_index = right.index._values
    return left_index, right_index
