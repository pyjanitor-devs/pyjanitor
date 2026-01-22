# helper function for multiple joins

from typing import Hashable

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _agg_functions, _helpers


def _agg_join_left(df: pd.DataFrame, aggfunc: list, indices: dict) -> pd.DataFrame:
    """
    Compute aggregation for multiple joins
    """
    if not indices["left_index"].size:
        dtypes = df.dtypes
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
    # single join - less than
    if (indices["starts"] is not None) and isinstance(indices["ends"], int):
        aggs = {}
        mapping = {
            "sum": _agg_functions._sum_rev_starts,
            "size": _agg_functions._size_rev_starts,
            # "min": _agg_functions._min_rev_starts,
            # "max": _agg_functions._max_rev_starts,
            # "prod": _agg_functions._prod_rev_starts,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                func = mapping[agg]
                out = func(
                    starts=indices["starts"],
                    index=indices["right_index"],
                )
            else:
                ser = df.loc[indices["left_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _helpers._convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(
                    arr=arr,
                    starts=indices["starts"],
                    index=indices["right_index"],
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
    # single join - greater than
    elif (indices["ends"] is not None) and isinstance(indices["starts"], int):
        aggs = {}
        mapping = {
            "sum": _agg_functions._sum_rev_ends,
            "size": _agg_functions._size_rev_ends,
            # "min": _agg_functions._min_rev_ends,
            # "max": _agg_functions._max_rev_ends,
            # "prod": _agg_functions._prod_rev_ends,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                func = mapping[agg]
                out = func(
                    ends=indices["ends"],
                    index=indices["right_index"],
                )
            else:
                ser = df.loc[indices["left_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _helpers._convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(
                    arr=arr,
                    ends=indices["ends"],
                    index=indices["right_index"],
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
    # # applies to equi join results
    # # for many-to-one, or one-to-one
    # elif not indices.keys() - {"left_index", "right_index"}:
    #     aggs = {}
    #     right_index = indices["right_index"]
    #     for column_name, agg in aggfunc:
    #         if agg == "size":
    #             out = np.repeat(1, right_index.size)
    #         else:
    #             out = right.loc[right_index, column_name]
    #             out = out._values
    #         new_label = _build_agg_label(column_name=column_name, agg_name=agg)
    #         aggs[new_label] = out
    # elif (indices["starts"] is not None) and (indices["ends"] is None):
    #     aggs = {}
    #     mapping = {
    #         "sum": _agg_functions._sum_starts_matches,
    #         "min": _agg_functions._min_starts_matches,
    #         "max": _agg_functions._max_starts_matches,
    #         "prod": _agg_functions._prod_starts_matches,
    #     }
    #     for column_name, agg in aggfunc:
    #         if agg == "size":
    #             out = indices["counts_array"]
    #         else:
    #             ser = right.loc[indices["right_index"], column_name]
    #             arr = ser._values
    #             booleans = pd.isna(arr)
    #             arr = _helpers._convert_array_to_numpy(array=arr)
    #             func = mapping[agg]
    #             out = func(
    #                 arr=arr,
    #                 starts=indices["starts"],
    #                 counts=indices["counts_array"],
    #                 matches=indices["matches"],
    #                 booleans=booleans,
    #             )
    #         if agg in {
    #             "sum",
    #             "prod",
    #         } and pd.api.types.is_extension_array_dtype(ser):
    #             out = pd.array(out, dtype=ser.dtype)
    #         elif agg in {"min", "max"}:
    #             bools = out == -1
    #             out = ser.iloc[out]
    #             if bools.any():
    #                 out = out.mask(bools)
    #             out = out._values
    #         new_label = _build_agg_label(column_name=column_name, agg_name=agg)
    #         aggs[new_label] = out
    # elif (indices["starts"] is None) and (indices["ends"] is not None):
    #     aggs = {}
    #     mapping = {
    #         "sum": _agg_functions._sum_ends_matches,
    #         "min": _agg_functions._min_ends_matches,
    #         "max": _agg_functions._max_ends_matches,
    #         "prod": _agg_functions._prod_ends_matches,
    #     }
    #     for column_name, agg in aggfunc:
    #         if agg == "size":
    #             out = indices["counts_array"]
    #         else:
    #             ser = right.loc[indices["right_index"], column_name]
    #             arr = ser._values
    #             booleans = pd.isna(arr)
    #             arr = _helpers._convert_array_to_numpy(array=arr)
    #             func = mapping[agg]
    #             out = func(
    #                 arr=arr,
    #                 ends=indices["ends"],
    #                 counts=indices["counts_array"],
    #                 matches=indices["matches"],
    #                 booleans=booleans,
    #             )
    #         if agg in {
    #             "sum",
    #             "prod",
    #         } and pd.api.types.is_extension_array_dtype(ser):
    #             out = pd.array(out, dtype=ser.dtype)
    #         elif agg in {"min", "max"}:
    #             bools = out == -1
    #             out = ser.iloc[out]
    #             if bools.any():
    #                 out = out.mask(bools)
    #             out = out._values
    #         new_label = _build_agg_label(column_name=column_name, agg_name=agg)
    #         aggs[new_label] = out
    # elif indices.get("positions") is not None:
    #     aggs = {}
    #     mapping = {
    #         "sum": _agg_functions._sum_positions,
    #         "min": _agg_functions._min_positions,
    #         "max": _agg_functions._max_positions,
    #         "prod": _agg_functions._prod_positions,
    #     }
    #     starts = indices["starts"]
    #     ends = indices["ends"]
    #     positions = indices["positions"]
    #     counts_array = indices["counts_array"]
    #     right_index = indices["right_index"]
    #     for column_name, agg in aggfunc:
    #         if agg == "size":
    #             out = counts_array
    #         else:
    #             ser = right.loc[right_index, column_name]
    #             arr = ser._values
    #             booleans = pd.isna(arr)
    #             arr = _helpers._convert_array_to_numpy(array=arr)
    #             func = mapping[agg]
    #             out = func(
    #                 arr=arr,
    #                 starts=starts,
    #                 ends=ends,
    #                 positions=positions,
    #                 booleans=booleans,
    #             )
    #         if agg in {
    #             "sum",
    #             "prod",
    #         } and pd.api.types.is_extension_array_dtype(ser):
    #             out = pd.array(out, dtype=ser.dtype)
    #         elif agg in {"min", "max"}:
    #             bools = out == -1
    #             out = ser.iloc[out]
    #             if bools.any():
    #                 out = out.mask(bools)
    #             out = out._values
    #         new_label = _build_agg_label(column_name=column_name, agg_name=agg)
    #         aggs[new_label] = out
    # elif (
    #     (indices["starts"] is not None)
    #     and (indices["ends"] is not None)
    #     and (indices.get("matches", None) is None)
    # ):
    #     aggs = {}
    #     mapping = {
    #         "sum": _agg_functions._sum_starts_ends,
    #         "min": _agg_functions._min_starts_ends,
    #         "max": _agg_functions._max_starts_ends,
    #         "prod": _agg_functions._prod_starts_ends,
    #     }
    #     starts = indices["starts"]
    #     ends = indices["ends"]
    #     right_index = indices["right_index"]
    #     for column_name, agg in aggfunc:
    #         if agg == "size":
    #             out = ends - starts
    #         else:
    #             ser = right.loc[right_index, column_name]
    #             arr = ser._values
    #             booleans = pd.isna(arr)
    #             arr = _helpers._convert_array_to_numpy(array=arr)
    #             func = mapping[agg]
    #             out = func(
    #                 arr=arr,
    #                 starts=starts,
    #                 ends=ends,
    #                 booleans=booleans,
    #             )
    #         if agg in {
    #             "sum",
    #             "prod",
    #         } and pd.api.types.is_extension_array_dtype(ser):
    #             out = pd.array(out, dtype=ser.dtype)
    #         elif agg in {"min", "max"}:
    #             bools = out == -1
    #             out = ser.iloc[out]
    #             if bools.any():
    #                 out = out.mask(bools)
    #             out = out._values
    #         new_label = _build_agg_label(column_name=column_name, agg_name=agg)
    #         aggs[new_label] = out
    # else:
    #     aggs = {}
    #     mapping = {
    #         "sum": _agg_functions._sum_starts_ends_matches,
    #         "min": _agg_functions._min_starts_ends_matches,
    #         "max": _agg_functions._max_starts_ends_matches,
    #         "prod": _agg_functions._prod_starts_ends_matches,
    #     }
    #     starts = indices["starts"]
    #     ends = indices["ends"]
    #     counts_array = indices["counts_array"]
    #     matches = indices["matches"]
    #     right_index = indices["right_index"]
    #     for column_name, agg in aggfunc:
    #         if agg == "size":
    #             out = counts_array
    #         else:
    #             ser = right.loc[right_index, column_name]
    #             arr = ser._values
    #             booleans = pd.isna(arr)
    #             arr = _helpers._convert_array_to_numpy(array=arr)
    #             func = mapping[agg]
    #             out = func(
    #                 arr=arr,
    #                 starts=starts,
    #                 ends=ends,
    #                 matches=matches,
    #                 counts=counts_array,
    #                 booleans=booleans,
    #             )
    #         if agg in {
    #             "sum",
    #             "prod",
    #         } and pd.api.types.is_extension_array_dtype(ser):
    #             out = pd.array(out, dtype=ser.dtype)
    #         elif agg in {"min", "max"}:
    #             bools = out == -1
    #             out = ser.iloc[out]
    #             if bools.any():
    #                 out = out.mask(bools)
    #             out = out._values
    #         new_label = _build_agg_label(column_name=column_name, agg_name=agg)
    #         aggs[new_label] = out
    return aggs
    left_index = indices["left_index"]
    left_index = pd.Index(left_index)
    return pd.DataFrame(aggs, copy=False, index=left_index)


def _agg_join_right(right: pd.DataFrame, aggfunc: list, indices: dict) -> pd.DataFrame:
    """
    Compute aggregation for multiple joins
    """
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
    # single join - less than
    if (indices["starts"] is not None) and isinstance(indices["ends"], int):
        aggs = {}
        mapping = {
            "sum": _agg_functions._sum_starts,
            "min": _agg_functions._min_starts,
            "max": _agg_functions._max_starts,
            "prod": _agg_functions._prod_starts,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                out = indices["ends"] - indices["starts"]
            else:
                ser = right.loc[indices["right_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _helpers._convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(arr=arr, starts=indices["starts"], booleans=booleans)
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
    # single join - greater than
    elif (indices["ends"] is not None) and isinstance(indices["starts"], int):
        aggs = {}
        mapping = {
            "sum": _agg_functions._sum_ends,
            "min": _agg_functions._min_ends,
            "max": _agg_functions._max_ends,
            "prod": _agg_functions._prod_ends,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                out = indices["ends"]
            else:
                ser = right.loc[indices["right_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _helpers._convert_array_to_numpy(array=arr)
                func = mapping[agg]
                out = func(arr=arr, ends=indices["ends"], booleans=booleans)
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
    # applies to equi join results
    # for many-to-one, or one-to-one
    elif not indices.keys() - {"left_index", "right_index"}:
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
            "sum": _agg_functions._sum_starts_matches,
            "min": _agg_functions._min_starts_matches,
            "max": _agg_functions._max_starts_matches,
            "prod": _agg_functions._prod_starts_matches,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                out = indices["counts_array"]
            else:
                ser = right.loc[indices["right_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _helpers._convert_array_to_numpy(array=arr)
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
            "sum": _agg_functions._sum_ends_matches,
            "min": _agg_functions._min_ends_matches,
            "max": _agg_functions._max_ends_matches,
            "prod": _agg_functions._prod_ends_matches,
        }
        for column_name, agg in aggfunc:
            if agg == "size":
                out = indices["counts_array"]
            else:
                ser = right.loc[indices["right_index"], column_name]
                arr = ser._values
                booleans = pd.isna(arr)
                arr = _helpers._convert_array_to_numpy(array=arr)
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
            "sum": _agg_functions._sum_positions,
            "min": _agg_functions._min_positions,
            "max": _agg_functions._max_positions,
            "prod": _agg_functions._prod_positions,
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
                arr = _helpers._convert_array_to_numpy(array=arr)
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
            "sum": _agg_functions._sum_starts_ends,
            "min": _agg_functions._min_starts_ends,
            "max": _agg_functions._max_starts_ends,
            "prod": _agg_functions._prod_starts_ends,
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
                arr = _helpers._convert_array_to_numpy(array=arr)
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
            "sum": _agg_functions._sum_starts_ends_matches,
            "min": _agg_functions._min_starts_ends_matches,
            "max": _agg_functions._max_starts_ends_matches,
            "prod": _agg_functions._prod_starts_ends_matches,
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
                arr = _helpers._convert_array_to_numpy(array=arr)
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
