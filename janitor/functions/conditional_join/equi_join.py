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

    if use_numba:
        eqs = None
        for left_on, right_on, op in conditions:
            if op == helpers._JoinOperator.STRICTLY_EQUAL.value:
                eqs = (left_on, right_on, op)
                break

        le_lt = None
        ge_gt = None

        for condition in conditions:
            *_, op = condition
            if op in helpers.less_than_join_types:
                if le_lt:
                    continue
                le_lt = condition
            elif op in helpers.greater_than_join_types:
                if ge_gt:
                    continue
                ge_gt = condition
            if le_lt and ge_gt:
                break
        if not le_lt and not ge_gt:
            raise ValueError(
                "At least one less than or greater than "
                "join condition should be present when an equi-join "
                "is present, and use_numba is set to True."
            )
        rest = [
            condition
            for condition in conditions
            if condition not in {eqs, le_lt, ge_gt}
        ]

        right_columns = [eqs[1]]
        df_columns = [eqs[0]]
        # ensure the sort columns are unique
        if ge_gt:
            if ge_gt[1] not in right_columns:
                right_columns.append(ge_gt[1])
            if ge_gt[0] not in df_columns:
                df_columns.append(ge_gt[0])
        if le_lt:
            if le_lt[1] not in right_columns:
                right_columns.append(le_lt[1])
            if le_lt[0] not in df_columns:
                df_columns.append(le_lt[0])

        right_df = right.loc(axis=1)[right_columns]
        left_df = df.loc(axis=1)[df_columns]
        any_nulls = left_df.isna().any(axis=1)
        if any_nulls.all(axis=None):
            return None
        if any_nulls.any():
            left_df = left_df.loc[~any_nulls]
        any_nulls = right_df.isna().any(axis=1)
        if any_nulls.all(axis=None):
            return None
        if any_nulls.any():
            right_df = right.loc[~any_nulls]
        equi_col = right_columns[0]
        # check if the first column is sorted
        # if sorted, check if the second column is sorted
        # per group in the first column
        right_is_sorted = right_df[equi_col].is_monotonic_increasing
        if right_is_sorted:
            grp = right_df.groupby(equi_col, sort=False, observed=True)
            non_equi_col = right_columns[1]
            # groupby.is_monotonic_increasing uses apply under the hood
            # the approach used below circumvents the Series creation
            # (which isn't required here)
            # and just gets a sequence of booleans, before calling `all`
            # to get a single True or False.
            right_is_sorted = all(
                arr.is_monotonic_increasing for _, arr in grp[non_equi_col]
            )
        if not right_is_sorted:
            right_df = right_df.sort_values(right_columns)
        rest = [
            (
                df.loc[left_df.index, left_on],
                right.loc[right_df.index, right_on],
                op,
            )
            for left_on, right_on, op in rest
        ]
        return _numba_equi_join(
            df=left_df,
            right=right_df,
            eqs=eqs,
            ge_gt=ge_gt,
            le_lt=le_lt,
            rest=rest,
            row_count=row_count if row_count else None,
        )
    outcome = helpers._separate_conditions_based_on_op(
        conditions=conditions, keep_equals_separate=True
    )
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

    equals = outcome["equals"]
    if use_pandas_merge_for_equi_join:
        check = False
    else:
        check = _is_binary_search_appropriate(df=df, equals=equals)
    if not check:
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
    # can we steal some performance, if there is a true range join?
    is_fastpath_range_join = False
    _, col, _ = equals[0]
    sorter = {col: 1}
    ge_gt = None
    le_lt = None
    if outcome.get("is_range_join"):
        ge_gt, le_lt, *conditions = outcome["conditions"]
        sorter[col] = 1
        _, col, _ = ge_gt
        sorter[col] = 1
        sorter = [*sorter]
        right = right.sort_values(by=sorter, ignore_index=False, kind="stable")
        # we already know that ge_gt is increasing monotonic,
        # based on the sort above
        # we do need to check le_lt though and see if
        # we can steal some perf. there for a true range join
        grouper = equals[0][1]
        grouped = right.groupby([grouper], sort=False, observed=True)
        grouped = grouped[le_lt[1]]
        is_fastpath_range_join = grouped.is_monotonic_increasing.all()
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
    l_col, r_col, _ = equals[0]
    indices = helpers._equal_indices(left=df[l_col], right=right[r_col])
    if indices is None:
        return None
    left_index, right_index, starts, ends = indices
    indices = dict(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        sizes=ends - starts,
    )
    if return_ranges and not outcome.get("conditions"):
        return indices
    if not outcome.get("conditions"):
        return helpers._build_indices_fast_path_range_join_only(
            left_index=indices["left_index"],
            right_index=indices["right_index"],
            starts=indices["starts"],
            ends=indices["ends"],
            booleans=np.ones(indices["left_index"].size, dtype=np.int8),
            keep=keep,
            total=indices["sizes"].sum(),
            matches=indices["left_index"].size,
        )
    # != only
    if not outcome.get("less_than_or_greater_than"):
        indices = helpers._build_start_indices(indices=indices)
        booleans = np.ones(indices["sizes"].size, dtype=np.int8)
        indices = helpers._get_positive_matches(
            df=df,
            right=right,
            indices=indices,
            conditions=outcome["conditions"],
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
    # range join only
    if is_fastpath_range_join:
        ge_gt, le_lt, *conditions = outcome["conditions"]
        indices = _get_prelim_indices_range_join(
            df=df, right=right, le_lt=le_lt, ge_gt=ge_gt, indices=indices
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
    condition, *conditions = outcome["conditions"]
    indices["sizes"] = indices["ends"] - indices["starts"]
    indices["booleans"] = np.ones(indices["left_index"].size, dtype=np.int8)
    indices = _update_search_indices_equi_join(
        df=df, right=right, indices=indices, condition=condition
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


def _update_search_indices_equi_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    indices: dict,
    condition: tuple,
):
    """
    Update `starts` or `ends` for an equi join
    """
    left_on, right_on, op = condition
    left_array = df.loc[indices["left_index"], left_on]._values
    right_array = right.loc[indices["right_index"], right_on]._values
    left_array, right_array = helpers._convert_to_numpy(
        left=left_array, right=right_array
    )
    new_ends = None
    new_starts = None
    if op == ">":
        new_ends, booleans, sizes, total, matches = (
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
        new_ends, booleans, sizes, total, matches = (
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
        new_starts, booleans, sizes, total, matches = (
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
        new_starts, booleans, sizes, total, matches = (
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
    if new_ends is not None:
        indices["ends"] = new_ends
    if new_starts is not None:
        indices["starts"] = new_starts
    indices["booleans"] = booleans
    indices["total"] = total
    indices["matches"] = matches
    indices["sizes"] = sizes
    return indices


def _get_prelim_indices_range_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    le_lt: tuple,
    ge_gt: tuple,
    indices: dict,
):
    """
    Get preliminary indices for a range join
    """
    indices["sizes"] = np.zeros(indices["left_index"].size, dtype=np.intp)
    indices["booleans"] = np.ones(indices["left_index"].size, dtype=np.int8)
    indices = _update_search_indices_equi_join(
        df=df,
        right=right,
        indices=indices,
        condition=ge_gt,
    )
    if indices is None:
        return None
    indices = _update_search_indices_equi_join(
        df=df, right=right, condition=le_lt, indices=indices
    )
    if indices is None:
        return None
    return indices


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
