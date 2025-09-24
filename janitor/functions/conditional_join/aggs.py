from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import ensure_float64

from janitor.cython_functions import cond_join_aggs

from . import helpers


def compute_aggfunc_result(
    aggfunc: list[tuple],
    agg_frame: pd.DataFrame,
    indices: dict,
    total: int,
) -> dict:
    """
    Compute aggfunc results
    """
    results = []
    for column_name, agg in aggfunc:
        if agg == "size":
            result = indices["counts_array"]
        elif agg == "count":
            series = agg_frame[column_name]
            if not series.hasnans:
                result = indices["counts_array"]
            else:
                result = _compute_agg_sum(
                    indices=indices,
                    series=series.notna().astype(np.int8, copy=False),
                    total=total,
                )
        elif agg == "sum":
            series = agg_frame[column_name]
            result = _compute_agg_sum(
                indices=indices, series=series, total=total
            )
        else:
            series = agg_frame[column_name]
            # TODO ??:
            # if min/max has already been computed for
            # start,end pair can we cache/reuse?
            result = _compute_agg_min_max(
                indices=indices,
                series=series,
                compute_max=agg == "max",
                total=total,
            )
        results.append(result)

    return results


def _compute_agg_sum(indices: dict, series: pd.Series, total: int) -> pd.array:
    """
    Compute sum for a conditional join aggregation
    """
    matches = indices.get("matches")
    positions = indices.get("positions")
    arr = helpers._convert_array_to_numpy(array=series._values, na_value=0)
    is_float_array = pd.api.types.is_float_dtype(series)
    is_int_array = pd.api.types.is_integer_dtype(series)
    if (not series.index.is_monotonic_increasing) and is_float_array:
        warnings.warn(
            "Summation on a float column "
            "may produce incorrect results, "
            "as the original order has changed."
        )
    if (
        isinstance(matches, np.ndarray)
        and isinstance(positions, np.ndarray)
        and is_float_array
    ):
        result = cond_join_aggs.get_sums_from_ranges_matches_positions_floats(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif (
        isinstance(matches, np.ndarray)
        and isinstance(positions, np.ndarray)
        and is_int_array
    ):
        result = cond_join_aggs.get_sums_from_ranges_matches_positions_ints(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif isinstance(positions, np.ndarray) and is_float_array:
        result = cond_join_aggs.get_sums_from_ranges_positions_floats(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )
    elif isinstance(positions, np.ndarray):
        result = cond_join_aggs.get_sums_from_ranges_positions_ints(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )
    elif isinstance(matches, np.ndarray) and is_float_array:
        result = cond_join_aggs.get_sums_from_ranges_matches_floats(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif isinstance(matches, np.ndarray) and is_int_array:
        result = cond_join_aggs.get_sums_from_ranges_matches_ints(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif is_float_array:
        result = cond_join_aggs.get_sums_from_ranges_floats(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )
    else:
        result = cond_join_aggs.get_sums_from_ranges_ints(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )

    if pd.api.types.is_extension_array_dtype(series) and is_float_array:
        return pd.array(result, dtype="Float64", copy=False)
    if pd.api.types.is_extension_array_dtype(series) and is_int_array:
        return pd.array(result, dtype="Int64", copy=False)
    return result


def _compute_agg_min_max(
    indices: dict,
    series: pd.Series,
    compute_max: bool,
    total: int,
) -> pd.array:
    """
    Compute min/max for a conditional join
    """
    matches = indices.get("matches")
    positions = indices.get("positions")
    arr = series._values
    if pd.api.types.is_extension_array_dtype(series) and series.hasnans:
        arr = ensure_float64(arr)
    arr = helpers._convert_array_to_numpy(array=arr)
    if (
        isinstance(matches, np.ndarray)
        and isinstance(positions, np.ndarray)
        and compute_max
    ):
        indexer = cond_join_aggs.get_max_from_ranges_positions_matches(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif isinstance(matches, np.ndarray) and isinstance(positions, np.ndarray):
        indexer = cond_join_aggs.get_min_from_ranges_positions_matches(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif isinstance(positions, np.ndarray) and compute_max:
        indexer = cond_join_aggs.get_max_from_positions_ranges(
            booleans=indices["booleans"],
            positions=indices["positions"],
            indexers=indices["indexers"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )
    elif isinstance(positions, np.ndarray):
        indexer = cond_join_aggs.get_min_from_positions_ranges(
            booleans=indices["booleans"],
            indexers=indices["indexers"],
            positions=indices["positions"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )
    elif isinstance(matches, np.ndarray) and compute_max:
        indexer = cond_join_aggs.get_max_from_ranges_matches(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif isinstance(matches, np.ndarray):
        indexer = cond_join_aggs.get_min_from_ranges_matches(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            sizes=indices["sizes"],
            matches=matches,
            total=total,
        )
    elif compute_max:
        indexer = cond_join_aggs.get_max_from_ranges(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )
    else:
        indexer = cond_join_aggs.get_min_from_ranges(
            booleans=indices["booleans"],
            arr=arr,
            starts=indices["starts"],
            ends=indices["ends"],
            total=total,
        )
    result = series.array[indexer]
    return result


def compute_aggfunc_result_no_ranges(
    aggfunc: list[tuple],
    agg_frame: pd.DataFrame,
    indexers: np.ndarray,
) -> list:
    """
    Compute aggfunc results
    """
    results = []
    for column_name, agg in aggfunc:
        if agg == "size":
            result = np.ones(indexers.size, dtype=np.int64)
        elif agg == "count":
            result = agg_frame[column_name].array[indexers]
            result = pd.notna(result).astype(np.int64, copy=False)
        else:
            result = agg_frame[column_name].array[indexers]
            if (agg == "sum") and (bools := pd.isna(result)).any():
                result[bools] = 0
        results.append(result)

    return results
