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
    # steal some perf here within the binary search
    # search for uniques
    # and later index them with left_positions
    # it is assumed that users will only reach for this
    # if the data is reasonably duplicated; if not
    # pd.merge is superb especially if it's a one-to-one
    # or one-to-many
    left_index = left.index._values
    right_index = right.index._values
    right = right._values
    positions, left = pd.factorize(left, sort=False)
    left = left._values
    starts = right.searchsorted(left, side="left")
    starts = starts[positions]
    ends = right.searchsorted(left, side="right")
    ends = ends[positions]
    booleans = starts < ends
    if not booleans.any():
        return None
    return {
        "starts": starts,
        "ends": ends,
        "booleans": booleans,
        "left_index": left_index,
        "right_index": right_index,
    }


def _less_than_indices(
    left: pd.array,
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
    bools = np.array([], dtype=np.bool_)
    # the idea here is that if there are any equal values
    # shift to the right to the immediate next position
    # that is not equal
    if strict and not booleans.all():
        bools = np.where(booleans, search_indices, search_indices - 1)
        bools = left == right[bools]
    elif strict:
        bools = left == right[search_indices]
    # replace positions where rows are equal
    # with positions from searchsorted('right')
    # positions from searchsorted('right') will never
    # be equal and will be the furthermost in terms of position
    # example : right -> [2, 2, 2, 3], and we need
    # positions where values are not equal for 2;
    # the furthermost will be 3, and searchsorted('right')
    # will return position 3.
    if bools.any():
        replacements = right.searchsorted(left, side="right")
        search_indices = np.where(bools, replacements, search_indices)
    # check again if any of the values
    # have become equal to length of right
    # and get rid of them
    booleans = search_indices < len_right
    if not booleans.any():
        return None
    return search_indices, booleans


def _single_le_ge_join_agg(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    aggfunc: list[tuple],
) -> tuple:
    """
    Compute aggregate on '</<=/>/>='
    """
    left_on, right_on, op = condition
    booleans = _maybe_remove_nulls_from_dataframe(
        df=df, columns=[left_on], return_bools=True
    )
    if booleans is None:
        return None
    right = _maybe_remove_nulls_from_dataframe(df=right, columns=[right_on])
    if right is None:
        return None
    if not right[right_on].is_monotonic_increasing:
        right = right.sort_values(right_on, ignore_index=False, kind="stable")
    len_df = len(df)
    len_right = len(right)
    starts = np.zeros(len_df, dtype=np.intp)
    ends = np.empty(len_df, dtype=np.intp)
    ends[:] = len_right
    sizes = np.zeros(len_df, dtype=np.intp)
    booleans = booleans.astype(np.int8, copy=False)
    indices = {
        "starts": starts,
        "ends": ends,
        "booleans": booleans,
        "sizes": sizes,
    }
    indices = _update_search_indices(
        left=df[left_on]._values,
        right=right[right_on]._values,
        indices=indices,
        op=op,
    )
    if indices is None:
        return None
    indices["counts_array"] = indices["sizes"]
    return compute_aggfunc_result(
        aggfunc=aggfunc, agg_frame=right, indices=indices, df_index=df.index
    )


def compute_aggfunc_result_no_ranges(
    aggfunc: list[tuple],
    agg_frame: pd.DataFrame,
    right_index: np.ndarray,
    booleans: np.ndarray,
    df_index: pd.Index,
) -> dict:
    """
    Compute aggfunc results
    """
    results = []
    column_names = []
    agg_names = []
    for column_name, agg in aggfunc:
        if agg == "size":
            result = pd.Series(
                booleans.astype(np.int8, copy=False), index=df_index
            )
        elif agg == "count":
            series = agg_frame[column_name]
            nulls_mask = series.isna()
            if not nulls_mask.any():
                result = pd.Series(
                    booleans.astype(np.int8, copy=False), index=df_index
                )
            else:
                if not booleans.all():
                    indexer = right_index[booleans]
                    l_index = df_index[booleans]
                else:
                    indexer = right_index[:]
                    l_index = df_index[:]
                result = pd.Series(1, index=series.index)
                result = result.mask(nulls_mask, 0)
                result = result.iloc[indexer]
                result.index = l_index
                result = result.reindex(df_index, fill_value=0)
        elif agg in {"min", "max", "sum"}:
            series = agg_frame[column_name]
            nulls_mask = series.isna()
            if not booleans.all():
                indexer = right_index[booleans]
                l_index = df_index[booleans]
            else:
                indexer = right_index[:]
                l_index = df_index[:]
            if not nulls_mask.any() and (agg == "sum"):
                result = series.iloc[indexer]
                result.index = l_index
                result = result.reindex(df_index, fill_value=0)
            elif not nulls_mask.any():
                result = series.iloc[indexer]
                result.index = l_index
                result = result.reindex(df_index)
            elif agg == "sum":
                result = series.mask(nulls_mask, 0)
                result = result.iloc[indexer]
                result.index = l_index
                result = result.reindex(df_index, fill_value=0)
            else:
                result = series.mask(nulls_mask)
                result = result.iloc[indexer]
                result.index = l_index
                result = result.reindex(df_index)
        results.append(result)
        column_names.append(column_name)
        agg_names.append(agg)

    return {
        "results": results,
        "column_names": column_names,
        "agg_names": agg_names,
        "index": df_index,
    }


def compute_aggfunc_result(
    aggfunc: list[tuple],
    agg_frame: pd.DataFrame,
    indices: dict,
    df_index: pd.Index,
) -> dict:
    """
    Compute aggfunc results
    """
    results = []
    column_names = []
    agg_names = []
    for column_name, agg in aggfunc:
        if agg == "sum":
            series = agg_frame[column_name]
            result = _compute_agg_sum(
                indices=indices, series=series, df_index=df_index
            )
        elif agg == "size":
            result = indices["counts_array"]
            result = pd.Series(result, index=df_index)
        elif agg == "count":
            series = agg_frame[column_name]
            if not series.hasnans:
                result = indices["counts_array"]
            else:
                result = _compute_agg_count(
                    indices=indices,
                    nulls_mask=series.isna().to_numpy(
                        dtype=np.int8, copy=False
                    ),
                )
            result = pd.Series(result, index=df_index)
        else:
            series = agg_frame[column_name]
            # TODO:
            # if i have already computed for
            # start,end pair can i cache/reuse?
            result = _compute_agg_min_max(
                indices=indices,
                series=series,
                compute_max=agg == "max",
                df_index=df_index,
            )
        results.append(result)
        column_names.append(column_name)
        agg_names.append(agg)

    return {
        "results": results,
        "column_names": column_names,
        "agg_names": agg_names,
    }


def compute_aggfunc_result_positions(
    aggfunc: list[tuple],
    agg_frame: pd.DataFrame,
    indices: dict,
    df_index: pd.Index,
) -> dict:
    """
    Compute aggfunc results
    """
    results = []
    column_names = []
    agg_names = []
    for column_name, agg in aggfunc:
        if agg == "sum":
            series = agg_frame[column_name]
            result = _compute_agg_sum_positions(
                indices=indices, series=series, df_index=df_index
            )
        elif agg == "size":
            result = pd.Series(indices["counts_array"], index=df_index)
        elif agg == "count":
            series = agg_frame[column_name]
            if not series.hasnans:
                result = pd.Series(indices["counts_array"], index=df_index)
            else:
                result = _compute_agg_count_positions(
                    indices=indices,
                    nulls_mask=series.isna().to_numpy(
                        dtype=np.int8, copy=False
                    ),
                )
            result = pd.Series(result, index=df_index)
        else:
            series = agg_frame[column_name]
            # TODO:
            # if i have already computed for
            # start,end pair can i cache/reuse?
            result = _compute_agg_min_max_positions(
                indices=indices,
                series=series,
                compute_max=agg == "max",
                df_index=df_index,
            )
        results.append(result)
        column_names.append(column_name)
        agg_names.append(agg)

    return {
        "results": results,
        "column_names": column_names,
        "agg_names": agg_names,
    }


def _compute_agg_count_positions(
    indices: dict, nulls_mask: np.ndarray
) -> pd.array:
    """
    Compute count for a conditional join aggregation
    """
    matches = indices.get("matches")
    if not isinstance(matches, np.ndarray):
        return cond_join.get_counts_from_ranges_positions_nulls(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            nulls_mask=nulls_mask,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    return cond_join.get_counts_from_ranges_matches_positions_nulls(
        booleans=indices["booleans"],
        indexers=indices["indexers"],
        positions=indices["positions"],
        nulls_mask=nulls_mask,
        starts=indices["starts"],
        ends=indices["ends"],
        sizes=indices["sizes"],
        matches=indices["matches"],
    )


def _compute_agg_count(indices: dict, nulls_mask: np.ndarray) -> pd.array:
    """
    Compute sum for a conditional join aggregation
    """
    matches = indices.get("matches")
    if not isinstance(matches, np.ndarray):
        return cond_join.get_counts_from_ranges_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    return cond_join.get_counts_from_ranges_matches_nulls(
        booleans=indices["booleans"],
        nulls_mask=nulls_mask,
        starts=indices["starts"],
        ends=indices["ends"],
        sizes=indices["sizes"],
        matches=indices["matches"],
    )


def _compute_agg_sum_positions(
    indices: dict, series: pd.Series, df_index: pd.Index
) -> pd.array:
    """
    Compute sum for a conditional join aggregation
    """
    nulls_mask = series.isna().to_numpy(dtype=np.int8, copy=False)
    any_nulls = nulls_mask.any()
    matches = indices.get("matches")
    arr = _convert_array_to_numpy(array=series._values)
    is_float_array = pd.api.types.is_float_dtype(series)
    is_int_array = pd.api.types.is_integer_dtype(series)
    if isinstance(matches, np.ndarray) and is_float_array and any_nulls:
        result = cond_join.get_sums_from_ranges_matches_floats_positions_nulls(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            nulls_mask=nulls_mask,
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and is_int_array and any_nulls:
        result = cond_join.get_sums_from_ranges_matches_ints_positions_nulls(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            nulls_mask=nulls_mask,
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and is_float_array:
        result = cond_join.get_sums_from_ranges_matches_positions_floats(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and is_int_array:
        result = cond_join.get_sums_from_ranges_matches_positions_ints(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif is_float_array and any_nulls:
        result = cond_join.get_sums_from_ranges_floats_positions_nulls(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif is_int_array and any_nulls:
        result = cond_join.get_sums_from_ranges_ints_positions_nulls(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif is_float_array:
        result = cond_join.get_sums_from_ranges_positions_floats(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    else:
        result = cond_join.get_sums_from_ranges_positions_ints(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    if pd.api.types.is_extension_array_dtype(series) and is_float_array:
        return pd.Series(result, dtype="Float64", index=df_index)
    if pd.api.types.is_extension_array_dtype(series) and is_int_array:
        return pd.Series(result, dtype="Int64", index=df_index)
    return pd.Series(result, index=df_index)


def _compute_agg_sum(
    indices: dict, series: pd.Series, df_index: pd.Index
) -> pd.array:
    """
    Compute sum for a conditional join aggregation
    """
    nulls_mask = series.isna().to_numpy(dtype=np.int8, copy=False)
    any_nulls = nulls_mask.any()
    matches = indices.get("matches")
    arr = _convert_array_to_numpy(array=series._values)
    is_float_array = pd.api.types.is_float_dtype(series)
    is_int_array = pd.api.types.is_integer_dtype(series)
    if isinstance(matches, np.ndarray) and is_float_array and any_nulls:
        result = cond_join.get_sums_from_ranges_matches_floats_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and is_int_array and any_nulls:
        result = cond_join.get_sums_from_ranges_matches_ints_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and is_float_array:
        result = cond_join.get_sums_from_ranges_matches_floats(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and is_int_array:
        result = cond_join.get_sums_from_ranges_matches_ints(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif is_float_array and any_nulls:
        result = cond_join.get_sums_from_ranges_floats_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif is_int_array and any_nulls:
        result = cond_join.get_sums_from_ranges_ints_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif is_float_array:
        result = cond_join.get_sums_from_ranges_floats(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    else:
        result = cond_join.get_sums_from_ranges_ints(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    if pd.api.types.is_extension_array_dtype(series) and is_float_array:
        return pd.Series(result, dtype="Float64", index=df_index)
    if pd.api.types.is_extension_array_dtype(series) and is_int_array:
        return pd.Series(result, dtype="Int64", index=df_index)
    return pd.Series(result, index=df_index)


def _compute_agg_min_max_positions(
    indices: dict, series: pd.Series, compute_max: bool, df_index: pd.Index
) -> pd.array:
    """
    Compute min/max for a conditional join
    """
    nulls_mask = series.isna().to_numpy(dtype=np.int8, copy=False)
    any_nulls = nulls_mask.any()
    matches = indices.get("matches")
    arr = _convert_array_to_numpy(array=series._values)

    if isinstance(matches, np.ndarray) and compute_max and any_nulls:
        indexer = cond_join.get_max_from_ranges_matches_positions_nulls(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and compute_max:
        indexer = cond_join.get_max_from_ranges_positions_matches(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and any_nulls:
        indexer = cond_join.get_min_from_ranges_matches_positions_nulls(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray):
        indexer = cond_join.get_min_from_ranges_positions_matches(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif compute_max and any_nulls:
        indexer = cond_join.get_max_from_ranges_positions_nulls(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif compute_max:
        indexer = cond_join.get_max_from_positions_ranges(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif any_nulls:
        indexer = cond_join.get_min_from_ranges_positions_nulls(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    else:
        indexer = cond_join.get_min_from_positions_ranges(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    result = series.iloc[indexer]
    bools = indexer < 0
    series_dtype = series.dtype
    bools_any = bools.any()
    # adapted from
    # https://github.com/pandas-dev/pandas/blob/29ce48952aaf857c89bf702a7ec79fdf5e6387b7/pandas/core/dtypes/missing.py#L603
    if bools_any and pd.api.types.is_extension_array_dtype(series_dtype):
        null = series_dtype.na_value
    elif bools_any and (series_dtype.kind in "mM"):
        unit = np.datetime_data(series_dtype)[0]
        null = series_dtype.type("NaT", unit)
    elif bools_any:
        null = np.nan
    if bools_any:
        result = result.mask(bools, null)
    result.index = df_index
    return result


def _compute_agg_min_max(
    indices: dict, series: pd.Series, compute_max: bool, df_index: pd.Index
) -> pd.array:
    """
    Compute min/max for a conditional join
    """
    nulls_mask = series.isna().to_numpy(dtype=np.int8, copy=False)
    any_nulls = nulls_mask.any()
    matches = indices.get("matches")
    arr = _convert_array_to_numpy(array=series._values)

    if isinstance(matches, np.ndarray) and compute_max and any_nulls:
        indexer = cond_join.get_max_from_ranges_matches_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and compute_max:
        indexer = cond_join.get_max_from_ranges_matches(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray) and any_nulls:
        indexer = cond_join.get_min_from_ranges_matches_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif isinstance(matches, np.ndarray):
        indexer = cond_join.get_min_from_ranges_matches(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
        )
    elif compute_max and any_nulls:
        indexer = cond_join.get_max_from_ranges_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif compute_max:
        indexer = cond_join.get_max_from_ranges(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    elif any_nulls:
        indexer = cond_join.get_min_from_ranges_nulls(
            booleans=indices["booleans"],
            nulls_mask=nulls_mask,
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    else:
        indexer = cond_join.get_min_from_ranges(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
        )
    result = series.iloc[indexer]
    bools = indexer < 0
    series_dtype = series.dtype
    bools_any = bools.any()
    # adapted from
    # https://github.com/pandas-dev/pandas/blob/29ce48952aaf857c89bf702a7ec79fdf5e6387b7/pandas/core/dtypes/missing.py#L603
    if bools_any and pd.api.types.is_extension_array_dtype(series_dtype):
        null = series_dtype.na_value
    elif bools_any and (series_dtype.kind in "mM"):
        unit = np.datetime_data(series_dtype)[0]
        null = series_dtype.type("NaT", unit)
    elif bools_any:
        null = np.nan
    if bools_any:
        result = result.mask(bools, null)
    result.index = df_index
    return result


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
        left=left._values,
        right=right._values,
        strict=strict,
    )

    if not outcome:
        return None
    search_indices, booleans = outcome
    left_index = left.index._values
    if not booleans.all():
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]
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
    bools = np.array([], dtype=np.bool_)
    # the idea here is that if there are any equal values
    # shift to the left to the immediate next position
    # that is not equal
    if strict and not booleans.all():
        bools = np.where(booleans, search_indices - 1, search_indices)
        bools = left == right[bools]
    elif strict:
        bools = left == right[search_indices - 1]
        # replace positions where rows are equal with
        # searchsorted('left');
        # this works fine since we will be using the value
        # as the right side of a slice, which is not included
        # in the final computed value
    if bools.any():
        replacements = right.searchsorted(left, side="left")
        search_indices = np.where(bools, replacements, search_indices)
    # any value less than 1 should be discarded
    # since the lowest value for binary search
    # with side='right' should be 1
    booleans = search_indices > 0
    if not booleans.any():
        return None
    return search_indices, booleans


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
    left_index = left.index._values
    outcome = _greater_than_indices(
        left=left.array,
        right=right.array,
        strict=strict,
    )

    if outcome is None:
        return None
    search_indices, booleans = outcome
    if not booleans.all():
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]
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
            left=left._values,
            right=right._values,
            strict=True,
        )
        if outcome is not None:
            lt_left = left.index._values
            search_indices, booleans = outcome
            if not booleans.all():
                lt_left = lt_left[booleans]
                search_indices = search_indices[booleans]
            len_right = right.size
            lt_right = [right_index[ind:len_right] for ind in search_indices]
            lt_left = [lt_left.repeat(len_right - search_indices)]
        outcome = _greater_than_indices(
            left=left._values,
            right=right._values,
            strict=True,
        )
        if outcome is not None:
            gt_left = left.index._values
            search_indices, booleans = outcome
            if not booleans.all():
                gt_left = gt_left[booleans]
                search_indices = search_indices[booleans]
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
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    keep: str,
    row_count: Hashable = None,
    return_ranges: bool = False,
    aggfunc: dict = None,
) -> tuple:
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices,
    or _not_equal_indices).
    """
    left_on, right_on, op = condition

    if (op in less_than_join_types) and (aggfunc is None):
        return _less_than_single_join(
            left=df[left_on],
            right=right[right_on],
            strict=op == _JoinOperator.LESS_THAN.value,
            keep=keep,
            row_count=row_count,
            return_ranges=return_ranges,
        )
    if (op in greater_than_join_types) and (aggfunc is None):
        return _greater_than_single_join(
            left=df[left_on],
            right=right[right_on],
            strict=op == _JoinOperator.GREATER_THAN.value,
            keep=keep,
            row_count=row_count,
            return_ranges=return_ranges,
        )
    if op == _JoinOperator.NOT_EQUAL.value:
        outcome = _not_equal_indices(
            left=df[left_on], right=right[right_on], keep=keep
        )
        if outcome is None:
            return None
        left_index, right_index = outcome
        if row_count:
            return (
                pd.Index(left_index).value_counts(sort=False).rename(row_count)
            )
        return left_index, right_index
    if aggfunc:
        return _single_le_ge_join_agg(
            df=df, right=right, condition=condition, aggfunc=aggfunc
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
        return cond_join.build_indices_from_ranges_keep_all(
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
        return cond_join.build_indices_from_ranges_keep_first(
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

    return cond_join.build_indices_from_ranges_keep_last(
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


def _multiple_conditions_get_indices_no_ranges(
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    matches: np.ndarray,
    keep: str,
):
    """
    get indices for multiple conditions
    """
    if keep == "all":
        return cond_join.build_indices_no_ranges_keep_all(
            left_index=left_index,
            right_index=right_index,
            left_indices=left_indices,
            right_indices=right_indices,
            matches=matches,
        )
    if keep == "first":
        return cond_join.build_indices_no_ranges_keep_first(
            left_index=left_index,
            right_index=right_index,
            left_indices=left_indices,
            right_indices=right_indices,
            matches=matches,
        )
    return cond_join.build_indices_no_ranges_keep_last(
        left_index=left_index,
        right_index=right_index,
        left_indices=left_indices,
        right_indices=right_indices,
        matches=matches,
    )


def _get_row_counts_multiple_conditions_no_ranges(
    left_index: np.ndarray,
    row_count: str,
    indices: np.ndarray,
    matches: np.ndarray,
):
    """Compute row count for multiple conditions"""
    counts_array = np.zeros(left_index[-1] + 1, dtype=np.intp)
    counts_array = cond_join.get_row_count_no_ranges(
        counts_array=counts_array,
        left_indices=indices,
        left_index=left_index,
        matches=matches,
    )
    return pd.Series(data=counts_array, name=row_count)


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
        return any_nulls.to_numpy(dtype=np.int8, copy=False)
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
    matches = np.ones(sizes.sum(), dtype=np.int8)
    counts_array = np.zeros(sizes.size, dtype=np.intp)
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
                )
            )
        elif is_extension_array:
            matches, booleans, counts_array, total, l_counts = (
                cond_join.get_positive_matches_ne_pandas_array(
                    starts=starts,
                    ends=ends,
                    sizes=sizes,
                    op=op,
                    matches=matches,
                    left=left,
                    right=right,
                    counts_array=counts_array,
                    booleans=booleans,
                    left_booleans=left_booleans.astype(np.int8, copy=False),
                    right_booleans=right_booleans.astype(np.int8, copy=False),
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
    matches = np.ones(sizes.sum(), dtype=np.int8)
    counts_array = np.zeros(booleans.size, dtype=np.intp)
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
                )
            )
        elif is_extension_array:
            matches, booleans, counts_array, total, l_counts = (
                cond_join.get_positive_matches_ranges_positions_ne_pandas_array(
                    starts=starts,
                    ends=ends,
                    sizes=sizes,
                    op=op,
                    matches=matches,
                    left=left,
                    right=right,
                    counts_array=counts_array,
                    booleans=booleans,
                    left_booleans=left_booleans.astype(np.int8, copy=False),
                    right_booleans=right_booleans.astype(np.int8, copy=False),
                    positions=positions,
                    indexers=indexers,
                )
            )
        else:
            matches, booleans, counts_array, total, l_counts = (
                cond_join.get_positive_matches_ranges_positions_ne(
                    starts=starts,
                    ends=ends,
                    sizes=sizes,
                    op=op,
                    matches=matches,
                    left=left,
                    right=right,
                    counts_array=counts_array,
                    booleans=booleans,
                    left_booleans=left_booleans.astype(np.int8, copy=False),
                    right_booleans=right_booleans.astype(np.int8, copy=False),
                    positions=positions,
                    indexers=indexers,
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


def _convert_array_to_numpy(array: np.ndarray) -> np.ndarray:
    """
    Ensure array is a numpy array.
    """
    if pd.api.types.is_extension_array_dtype(array):
        array_dtype = array.dtype.numpy_dtype
        array = array.to_numpy(dtype=array_dtype, na_value=-1, copy=False)
    if pd.api.types.is_timedelta64_dtype(array):
        array = array.to_numpy(copy=False)
    if pd.api.types.is_datetime64_dtype(
        array
    ) or pd.api.types.is_timedelta64_dtype(array):
        array = array.view(np.int64)
    return array


def _get_null_mask(series: pd.Series) -> np.ndarray:
    """
    Get boolean of null rows
    """
    any_nulls = series.isna()
    has_nans = any_nulls.any()
    if pd.api.types.is_extension_array_dtype(series):
        return any_nulls.to_numpy(na_value=False, copy=False), has_nans
    return any_nulls, has_nans


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
        elif is_extension_array:
            booleans, total = (
                cond_join.get_positive_matches_no_ranges_ne_pandas_array(
                    op=op,
                    left=left,
                    right=right,
                    right_index=right_index,
                    booleans=booleans,
                    left_booleans=left_booleans.astype(np.int8, copy=False),
                    right_booleans=right_booleans.astype(np.int8, copy=False),
                )
            )

        else:
            booleans, total = cond_join.get_positive_matches_no_ranges_ne(
                op=op,
                left=left,
                right=right,
                right_index=right_index,
                booleans=booleans,
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
):
    """
    Update `starts` or `ends` for non-equi
    """
    left, right = _convert_to_numpy(left=left, right=right)
    if op == ">":
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
