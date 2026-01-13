# helper function for multiple joins

from typing import Hashable

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _get_indices_equi,
    _get_indices_non_equi,
)
from janitor.functions._conditional_join._agg_functions import (
    _max_ends_matches,
    _max_positions,
    _max_starts_ends,
    _max_starts_ends_matches,
    _max_starts_matches,
    _min_ends_matches,
    _min_positions,
    _min_starts_ends,
    _min_starts_ends_matches,
    _min_starts_matches,
    _prod_ends_matches,
    _prod_positions,
    _prod_starts_ends,
    _prod_starts_ends_matches,
    _prod_starts_matches,
    _sum_ends_matches,
    _sum_positions,
    _sum_starts_ends,
    _sum_starts_ends_matches,
    _sum_starts_matches,
)
from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
)


def _agg_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    aggfunc: list,
    eq_check: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Compute aggregation for multiple joins
    """
    if eq_check and not force:
        indices = _get_indices_equi._get_indices(
            df=df,
            right=right,
            conditions=conditions,
            keep="all",
            return_matching_indices=True,
        )
    else:
        indices = _get_indices_non_equi._get_indices(
            df=df,
            right=right,
            conditions=conditions,
            keep="all",
            return_matching_indices=True,
        )
    if not indices["left_index"].size:
        dtypes = right.dtypes
        aggs = {}
        for column_name, agg in aggfunc:
            if agg == "size":
                _dtype = "int64"
            else:
                _dtype = dtypes.loc[column_name]
            out = pd.array([], dtype=_dtype, copy=False)
            new_label = _build_agg_label(column_name=column_name, agg_name=agg)
            aggs[new_label] = out
        return pd.DataFrame(aggs, copy=False)
    # applies to equi join results
    # for many-to-one, or one-to-one
    if not indices.keys() - {"left_index", "right_index"}:
        aggs = {}
        right_index = indices["right_index"]
        for column_name, agg in aggfunc:
            if agg == "size":
                out = np.repeat(1, right_index.size)
            else:
                out = right.loc[right_index, column_name]
                out = out._values
            new_label = _build_agg_label(column_name=column_name, agg_name=agg)
            aggs[new_label] = out
    elif (indices["starts"] is not None) and (indices["ends"] is None):
        aggs = {}
        mapping = {
            "sum": _sum_starts_matches,
            "min": _min_starts_matches,
            "max": _max_starts_matches,
            "prod": _prod_starts_matches,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                out = indices["counts_array"]
            else:
                ser = right.loc[indices["right_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(
                    arr=arr,
                    starts=indices["starts"],
                    counts=indices["counts_array"],
                    matches=indices["matches"],
                    booleans=booleans,
                )
            if agg in {
                "sum",
                "prod",
            } and pd.api.types.is_extension_array_dtype(ser):
                out = pd.array(out, dtype=ser.dtype)
            elif agg in {"min", "max"}:
                bools = out == -1
                out = ser.iloc[out]
                if bools.any():
                    out = out.mask(bools)
                out = out._values
            new_label = _build_agg_label(column_name=column_name, agg_name=agg)
            aggs[new_label] = out
    elif (indices["starts"] is None) and (indices["ends"] is not None):
        aggs = {}
        mapping = {
            "sum": _sum_ends_matches,
            "min": _min_ends_matches,
            "max": _max_ends_matches,
            "prod": _prod_ends_matches,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                out = indices["counts_array"]
            else:
                ser = right.loc[indices["right_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(
                    arr=arr,
                    ends=indices["ends"],
                    counts=indices["counts_array"],
                    matches=indices["matches"],
                    booleans=booleans,
                )
            if agg in {
                "sum",
                "prod",
            } and pd.api.types.is_extension_array_dtype(ser):
                out = pd.array(out, dtype=ser.dtype)
            elif agg in {"min", "max"}:
                bools = out == -1
                out = ser.iloc[out]
                if bools.any():
                    out = out.mask(bools)
                out = out._values
            new_label = _build_agg_label(column_name=column_name, agg_name=agg)
            aggs[new_label] = out
    elif indices.get("positions") is not None:
        aggs = {}
        mapping = {
            "sum": _sum_positions,
            "min": _min_positions,
            "max": _max_positions,
            "prod": _prod_positions,
        }
        starts = indices["starts"]
        ends = indices["ends"]
        positions = indices["positions"]
        counts_array = indices["counts_array"]
        right_index = indices["right_index"]
        for column_name, agg in aggfunc:
            if agg == "size":
                out = counts_array
            else:
                ser = right.loc[right_index, column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(
                    arr=arr,
                    starts=starts,
                    ends=ends,
                    positions=positions,
                    booleans=booleans,
                )
            if agg in {
                "sum",
                "prod",
            } and pd.api.types.is_extension_array_dtype(ser):
                out = pd.array(out, dtype=ser.dtype)
            elif agg in {"min", "max"}:
                bools = out == -1
                out = ser.iloc[out]
                if bools.any():
                    out = out.mask(bools)
                out = out._values
            new_label = _build_agg_label(column_name=column_name, agg_name=agg)
            aggs[new_label] = out
    elif (
        (indices["starts"] is not None)
        and (indices["ends"] is not None)
        and (indices.get("matches", None) is None)
    ):
        aggs = {}
        mapping = {
            "sum": _sum_starts_ends,
            "min": _min_starts_ends,
            "max": _max_starts_ends,
            "prod": _prod_starts_ends,
        }
        starts = indices["starts"]
        ends = indices["ends"]
        right_index = indices["right_index"]
        for column_name, agg in aggfunc:
            if agg == "size":
                out = ends - starts
            else:
                ser = right.loc[right_index, column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(
                    arr=arr,
                    starts=starts,
                    ends=ends,
                    booleans=booleans,
                )
            if agg in {
                "sum",
                "prod",
            } and pd.api.types.is_extension_array_dtype(ser):
                out = pd.array(out, dtype=ser.dtype)
            elif agg in {"min", "max"}:
                bools = out == -1
                out = ser.iloc[out]
                if bools.any():
                    out = out.mask(bools)
                out = out._values
            new_label = _build_agg_label(column_name=column_name, agg_name=agg)
            aggs[new_label] = out
    else:
        aggs = {}
        mapping = {
            "sum": _sum_starts_ends_matches,
            "min": _min_starts_ends_matches,
            "max": _max_starts_ends_matches,
            "prod": _prod_starts_ends_matches,
        }
        starts = indices["starts"]
        ends = indices["ends"]
        counts_array = indices["counts_array"]
        matches = indices["matches"]
        right_index = indices["right_index"]
        for column_name, agg in aggfunc:
            if agg == "size":
                out = counts_array
            else:
                ser = right.loc[right_index, column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(
                    arr=arr,
                    starts=starts,
                    ends=ends,
                    matches=matches,
                    counts=counts_array,
                    booleans=booleans,
                )
            if agg in {
                "sum",
                "prod",
            } and pd.api.types.is_extension_array_dtype(ser):
                out = pd.array(out, dtype=ser.dtype)
            elif agg in {"min", "max"}:
                bools = out == -1
                out = ser.iloc[out]
                if bools.any():
                    out = out.mask(bools)
                out = out._values
            new_label = _build_agg_label(column_name=column_name, agg_name=agg)
            aggs[new_label] = out
    left_index = indices["left_index"]
    left_index = pd.Index(left_index)
    return pd.DataFrame(aggs, copy=False, index=left_index)


def _build_agg_label(column_name: Hashable, agg_name: str):
    if isinstance(column_name, tuple):
        return (*column_name, agg_name)
    return (f"{column_name}", agg_name)
