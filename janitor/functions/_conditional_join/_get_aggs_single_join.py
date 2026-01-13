# helper function for a single join

from typing import Hashable

import pandas as pd

from janitor.functions._conditional_join._agg_functions import (
    _max_ends,
    _max_starts,
    _min_ends,
    _min_starts,
    _prod_ends,
    _prod_starts,
    _sum_ends,
    _sum_starts,
)
from janitor.functions._conditional_join._greater_than_indices import (
    _greater_than_indices,
)
from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
    less_than_join_types,
)
from janitor.functions._conditional_join._less_than_indices import (
    _less_than_indices,
)


def _agg_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    aggfunc: list,
) -> pd.DataFrame:
    """
    Compute aggregation for a single join
    """
    left_on, right_on, op = condition
    if op in less_than_join_types:
        indices = _less_than_indices(
            left=df[left_on],
            right=right[right_on],
            strict=op == "<",
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
        aggs = {}
        starts = indices["starts"]
        right_index = indices["right_index"]
        mapping = {
            "sum": _sum_starts,
            "min": _min_starts,
            "max": _max_starts,
            "prod": _prod_starts,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                out = right_index.size - starts
            else:
                ser = right.loc[right_index, column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(arr=arr, starts=starts, booleans=booleans)
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
    indices = _greater_than_indices(
        left=df[left_on],
        right=right[right_on],
        strict=op == ">",
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
    aggs = {}
    ends = indices["ends"]
    right_index = indices["right_index"]
    mapping = {
        "sum": _sum_ends,
        "min": _min_ends,
        "max": _max_ends,
        "prod": _prod_ends,
    }
    for column_name, agg in aggfunc:
        if agg == "size":
            out = ends
        else:
            ser = right.loc[right_index, column_name]
            arr = ser._values
            booleans = pd.isna(arr)
            arr = _convert_array_to_numpy(array=arr)
            func = mapping[agg]
            out = func(arr=arr, ends=ends, booleans=booleans)
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
