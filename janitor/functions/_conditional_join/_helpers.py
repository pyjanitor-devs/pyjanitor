# helper functions for conditional_join.py

from enum import Enum
from typing import Sequence

import janitor_rs
import numpy as np
import pandas as pd


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


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}

operator_mapping = {">": 0, ">=": 1, "<": 2, "<=": 3, "==": 4, "!=": 5}


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
        return any_nulls
    if any_nulls.any():
        df = df.loc[~any_nulls]
    return df


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


def _separate_conditions_based_on_op(conditions: Sequence):
    """
    Create separate blocks (`equals`, `not_equals`, `le_or_ge`)
    based on `op`
    """
    l_cols = set()
    r_cols = set()
    not_equals = []
    le_or_ge = []
    equals = []
    for condition in conditions:
        left_on, right_on, op = condition
        l_cols.add(left_on)
        r_cols.add(right_on)
        if op == _JoinOperator.NOT_EQUAL.value:
            not_equals.append(condition)
        elif op == _JoinOperator.STRICTLY_EQUAL.value:
            equals.append(condition)
        else:
            le_or_ge.append(condition)
    # check for possibility of a range join
    # keep the first match for le_lt or ge_gt
    le_lt = None
    ge_gt = None
    for condition in conditions:
        left_on, right_on, op = condition
        if le_lt and ge_gt:
            break
        if (op in less_than_join_types) and not le_lt:
            le_lt = (left_on, right_on, op)
        elif (op in greater_than_join_types) and not ge_gt:
            ge_gt = (left_on, right_on, op)
    is_range_join = all((le_lt, ge_gt))
    if is_range_join:
        le_or_ge = [
            condition for condition in le_or_ge if condition not in (ge_gt, le_lt)
        ]
    return {
        "l_cols": l_cols,
        "r_cols": r_cols,
        "is_range_join": is_range_join,
        "equals": equals,
        "not_equals": not_equals,
        "le_lt": le_lt,
        "ge_gt": ge_gt,
        "le_or_ge": le_or_ge,
    }


def _compare_ne_first_run_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_start_ne_1st_int64,
        "int32": janitor_rs.compare_start_ne_1st_int32,
        "int16": janitor_rs.compare_start_ne_1st_int16,
        "int8": janitor_rs.compare_start_ne_1st_int8,
        "uint64": janitor_rs.compare_start_ne_1st_uint64,
        "uint32": janitor_rs.compare_start_ne_1st_uint32,
        "uint16": janitor_rs.compare_start_ne_1st_uint16,
        "uint8": janitor_rs.compare_start_ne_1st_uint8,
        "float64": janitor_rs.compare_start_ne_1st_float64,
        "float32": janitor_rs.compare_start_ne_1st_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_ne_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for starts
    """
    mapping = {
        "int64": janitor_rs.compare_start_ne_int64,
        "int32": janitor_rs.compare_start_ne_int32,
        "int16": janitor_rs.compare_start_ne_int16,
        "int8": janitor_rs.compare_start_ne_int8,
        "uint64": janitor_rs.compare_start_ne_uint64,
        "uint32": janitor_rs.compare_start_ne_uint32,
        "uint16": janitor_rs.compare_start_ne_uint16,
        "uint8": janitor_rs.compare_start_ne_uint8,
        "float64": janitor_rs.compare_start_ne_float64,
        "float32": janitor_rs.compare_start_ne_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        counts_array,
        matches,
        is_extension_array,
        op,
    )


def _compare_ne_first_run_ends_only(
    left: np.ndarray,
    right: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_end_ne_1st_int64,
        "int32": janitor_rs.compare_end_ne_1st_int32,
        "int16": janitor_rs.compare_end_ne_1st_int16,
        "int8": janitor_rs.compare_end_ne_1st_int8,
        "uint64": janitor_rs.compare_end_ne_1st_uint64,
        "uint32": janitor_rs.compare_end_ne_1st_uint32,
        "uint16": janitor_rs.compare_end_ne_1st_uint16,
        "uint8": janitor_rs.compare_end_ne_1st_uint8,
        "float64": janitor_rs.compare_end_ne_1st_float64,
        "float32": janitor_rs.compare_end_ne_1st_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left,
        right,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_ne_ends_only(
    left: np.ndarray,
    right: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for ends
    """
    mapping = {
        "int64": janitor_rs.compare_end_ne_int64,
        "int32": janitor_rs.compare_end_ne_int32,
        "int16": janitor_rs.compare_end_ne_int16,
        "int8": janitor_rs.compare_end_ne_int8,
        "uint64": janitor_rs.compare_end_ne_uint64,
        "uint32": janitor_rs.compare_end_ne_uint32,
        "uint16": janitor_rs.compare_end_ne_uint16,
        "uint8": janitor_rs.compare_end_ne_uint8,
        "float64": janitor_rs.compare_end_ne_float64,
        "float32": janitor_rs.compare_end_ne_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left,
        right,
        ends,
        left_booleans,
        right_booleans,
        counts_array,
        matches,
        is_extension_array,
        op,
    )


def _compare_ne_first_run_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_start_end_ne_1st_int64,
        "int32": janitor_rs.compare_start_end_ne_1st_int32,
        "int16": janitor_rs.compare_start_end_ne_1st_int16,
        "int8": janitor_rs.compare_start_end_ne_1st_int8,
        "uint64": janitor_rs.compare_start_end_ne_1st_uint64,
        "uint32": janitor_rs.compare_start_end_ne_1st_uint32,
        "uint16": janitor_rs.compare_start_end_ne_1st_uint16,
        "uint8": janitor_rs.compare_start_end_ne_1st_uint8,
        "float64": janitor_rs.compare_start_end_ne_1st_float64,
        "float32": janitor_rs.compare_start_end_ne_1st_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_ne_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for starts and ends
    """
    mapping = {
        "int64": janitor_rs.compare_start_end_ne_int64,
        "int32": janitor_rs.compare_start_end_ne_int32,
        "int16": janitor_rs.compare_start_end_ne_int16,
        "int8": janitor_rs.compare_start_end_ne_int8,
        "uint64": janitor_rs.compare_start_end_ne_uint64,
        "uint32": janitor_rs.compare_start_end_ne_uint32,
        "uint16": janitor_rs.compare_start_end_ne_uint16,
        "uint8": janitor_rs.compare_start_end_ne_uint8,
        "float64": janitor_rs.compare_start_end_ne_float64,
        "float32": janitor_rs.compare_start_end_ne_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left,
        right,
        starts,
        ends,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    )


def _compare_first_run_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_first_start_int64,
        "int32": janitor_rs.compare_first_start_int32,
        "int16": janitor_rs.compare_first_start_int16,
        "int8": janitor_rs.compare_first_start_int8,
        "uint64": janitor_rs.compare_first_start_uint64,
        "uint32": janitor_rs.compare_first_start_uint32,
        "uint16": janitor_rs.compare_first_start_uint16,
        "uint8": janitor_rs.compare_first_start_uint8,
        "float64": janitor_rs.compare_first_start_float64,
        "float32": janitor_rs.compare_first_start_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(left, right, starts, op)


def _compare_starts_only(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for starts
    """
    mapping = {
        "int64": janitor_rs.compare_start_int64,
        "int32": janitor_rs.compare_start_int32,
        "int16": janitor_rs.compare_start_int16,
        "int8": janitor_rs.compare_start_int8,
        "uint64": janitor_rs.compare_start_uint64,
        "uint32": janitor_rs.compare_start_uint32,
        "uint16": janitor_rs.compare_start_uint16,
        "uint8": janitor_rs.compare_start_uint8,
        "float64": janitor_rs.compare_start_float64,
        "float32": janitor_rs.compare_start_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(left, right, starts, counts_array, matches, op)


def _compare_first_run_ends_only(
    left: np.ndarray, right: np.ndarray, ends: np.ndarray, op: int
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_first_end_int64,
        "int32": janitor_rs.compare_first_end_int32,
        "int16": janitor_rs.compare_first_end_int16,
        "int8": janitor_rs.compare_first_end_int8,
        "uint64": janitor_rs.compare_first_end_uint64,
        "uint32": janitor_rs.compare_first_end_uint32,
        "uint16": janitor_rs.compare_first_end_uint16,
        "uint8": janitor_rs.compare_first_end_uint8,
        "float64": janitor_rs.compare_first_end_float64,
        "float32": janitor_rs.compare_first_end_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(left, right, ends, op)


def _compare_ends_only(
    left: np.ndarray,
    right: np.ndarray,
    ends: np.ndarray,
    counts_array: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for ends
    """
    mapping = {
        "int64": janitor_rs.compare_end_int64,
        "int32": janitor_rs.compare_end_int32,
        "int16": janitor_rs.compare_end_int16,
        "int8": janitor_rs.compare_end_int8,
        "uint64": janitor_rs.compare_end_uint64,
        "uint32": janitor_rs.compare_end_uint32,
        "uint16": janitor_rs.compare_end_uint16,
        "uint8": janitor_rs.compare_end_uint8,
        "float64": janitor_rs.compare_end_float64,
        "float32": janitor_rs.compare_end_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(left, right, ends, counts_array, matches, op)


def _compare_first_run_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_first_start_end_int64,
        "int32": janitor_rs.compare_first_start_end_int32,
        "int16": janitor_rs.compare_first_start_end_int16,
        "int8": janitor_rs.compare_first_start_end_int8,
        "uint64": janitor_rs.compare_first_start_end_uint64,
        "uint32": janitor_rs.compare_first_start_end_uint32,
        "uint16": janitor_rs.compare_first_start_end_uint16,
        "uint8": janitor_rs.compare_first_start_end_uint8,
        "float64": janitor_rs.compare_first_start_end_float64,
        "float32": janitor_rs.compare_first_start_end_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(left, right, starts, ends, op)


def _compare_starts_ends(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    matches: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for starts and ends
    """
    mapping = {
        "int64": janitor_rs.compare_start_end_int64,
        "int32": janitor_rs.compare_start_end_int32,
        "int16": janitor_rs.compare_start_end_int16,
        "int8": janitor_rs.compare_start_end_int8,
        "uint64": janitor_rs.compare_start_end_uint64,
        "uint32": janitor_rs.compare_start_end_uint32,
        "uint16": janitor_rs.compare_start_end_uint16,
        "uint8": janitor_rs.compare_start_end_uint8,
        "float64": janitor_rs.compare_start_end_float64,
        "float32": janitor_rs.compare_start_end_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(left, right, starts, ends, matches, op)


def _compare_positions(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_posns_int64,
        "int32": janitor_rs.compare_posns_int32,
        "int16": janitor_rs.compare_posns_int16,
        "int8": janitor_rs.compare_posns_int8,
        "uint64": janitor_rs.compare_posns_uint64,
        "uint32": janitor_rs.compare_posns_uint32,
        "uint16": janitor_rs.compare_posns_uint16,
        "uint8": janitor_rs.compare_posns_uint8,
        "float64": janitor_rs.compare_posns_float64,
        "float32": janitor_rs.compare_posns_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left=left,
        right=right,
        positions=positions,
        starts=starts,
        ends=ends,
        op=op,
    )


def _compare_positions_ne(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    left_booleans: np.ndarray | None,
    right_booleans: np.ndarray | None,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    Compute booleans for first run
    """
    mapping = {
        "int64": janitor_rs.compare_posns_ne_int64,
        "int32": janitor_rs.compare_posns_ne_int32,
        "int16": janitor_rs.compare_posns_ne_int16,
        "int8": janitor_rs.compare_posns_ne_int8,
        "uint64": janitor_rs.compare_posns_ne_uint64,
        "uint32": janitor_rs.compare_posns_ne_uint32,
        "uint16": janitor_rs.compare_posns_ne_uint16,
        "uint8": janitor_rs.compare_posns_ne_uint8,
        "float64": janitor_rs.compare_posns_ne_float64,
        "float32": janitor_rs.compare_posns_ne_float32,
    }
    dtype_name = left.dtype.name
    func = mapping[dtype_name]
    return func(
        left=left,
        right=right,
        starts=starts,
        ends=ends,
        positions=positions,
        left_booleans=left_booleans,
        right_booleans=right_booleans,
        is_extension_array=is_extension_array,
        op=op,
    )


def _convert_array_to_numpy(
    array: np.ndarray,
    na_value: int = 0,
) -> np.ndarray:
    """
    Ensure array is a numpy array.
    """
    if pd.api.types.is_extension_array_dtype(array):
        array_dtype = array.dtype.numpy_dtype
        array = array.to_numpy(dtype=array_dtype, na_value=na_value, copy=False)
    if pd.api.types.is_timedelta64_dtype(array):
        array = array.to_numpy(copy=False)
    if pd.api.types.is_datetime64_dtype(array) or pd.api.types.is_timedelta64_dtype(
        array
    ):
        array = array.view(np.int64)
    return array


def _get_positive_matches_positions(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    op: str,
    left_booleans: np.ndarray | None,
    right_booleans: np.ndarray | None,
    is_extension_array: bool,
):
    """
    Compute positive matches for left vs right
    """
    if (left_booleans is None) and (right_booleans is None):
        positions, counts_array, total = _compare_positions(
            left=left,
            right=right,
            positions=positions,
            starts=starts,
            ends=ends,
            op=operator_mapping[op],
        )
    else:
        positions, counts_array, total = _compare_positions_ne(
            left=left,
            right=right,
            positions=positions,
            starts=starts,
            ends=ends,
            left_booleans=left_booleans,
            right_booleans=right_booleans,
            is_extension_array=is_extension_array,
            op=operator_mapping[op],
        )

    return positions, counts_array, total


def _get_positive_matches_conditions_posns(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    left_index: np.ndarray,
    right_index: np.ndarray,
    positions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
):
    """
    Get positive matches for conditions
    """

    (left_on, right_on, op), *rest = conditions
    left_array = df.loc[left_index, left_on]
    right_array = right.loc[right_index, right_on]
    left_booleans, right_booleans, is_extension_array = _get_boolean_args_for_ne(
        op=op, left=left_array, right=right_array
    )
    left_array = _convert_array_to_numpy(array=left_array._values)
    right_array = _convert_array_to_numpy(array=right_array._values)
    positions, counts_array, total = _get_positive_matches_positions(
        left=left_array,
        right=right_array,
        positions=positions,
        starts=starts,
        ends=ends,
        op=op,
        left_booleans=left_booleans,
        right_booleans=right_booleans,
        is_extension_array=is_extension_array,
    )
    if total == 0:
        return None
    for left_on, right_on, op in rest:
        left_array = df.loc[left_index, left_on]
        right_array = right.loc[right_index, right_on]
        left_booleans, right_booleans, is_extension_array = _get_boolean_args_for_ne(
            op=op, left=left_array, right=right_array
        )
        left_array = _convert_array_to_numpy(array=left_array._values)
        right_array = _convert_array_to_numpy(array=right_array._values)
        positions, counts_array, total = _get_positive_matches_positions(
            left=left_array,
            right=right_array,
            positions=positions,
            starts=starts,
            ends=ends,
            op=op,
            left_booleans=left_booleans,
            right_booleans=right_booleans,
            is_extension_array=is_extension_array,
        )
        if total == 0:
            return None
    return {
        "positions": positions,
        "counts_array": counts_array,
        "total": total,
    }


def _get_positive_matches(
    left: np.ndarray,
    right: np.ndarray,
    starts: np.ndarray | None,
    ends: np.ndarray | None,
    counts_array: np.ndarray | None,
    matches: np.ndarray | None,
    op: str,
    left_booleans: np.ndarray | None,
    right_booleans: np.ndarray | None,
    is_extension_array: bool,
):
    """
    Compute positive matches for left vs right
    """
    if (starts is not None) and (ends is None) and (counts_array is None):
        if (left_booleans is None) and (right_booleans is None):
            matches, counts_array, total = _compare_first_run_starts_only(
                left=left,
                right=right,
                starts=starts,
                op=operator_mapping[op],
            )
        else:
            matches, counts_array, total = _compare_ne_first_run_starts_only(
                left=left,
                right=right,
                starts=starts,
                left_booleans=left_booleans,
                right_booleans=right_booleans,
                is_extension_array=is_extension_array,
                op=operator_mapping[op],
            )
    elif (starts is None) and (ends is not None) and (counts_array is None):
        if (left_booleans is None) and (right_booleans is None):
            matches, counts_array, total = _compare_first_run_ends_only(
                left=left,
                right=right,
                ends=ends,
                op=operator_mapping[op],
            )
        else:
            matches, counts_array, total = _compare_ne_first_run_ends_only(
                left=left,
                right=right,
                ends=ends,
                left_booleans=left_booleans,
                right_booleans=right_booleans,
                is_extension_array=is_extension_array,
                op=operator_mapping[op],
            )
    elif (starts is not None) and (ends is None) and (counts_array is not None):
        if (left_booleans is None) and (right_booleans is None):
            matches, counts_array, total = _compare_starts_only(
                left=left,
                right=right,
                starts=starts,
                counts_array=counts_array,
                matches=matches,
                op=operator_mapping[op],
            )
        else:
            matches, counts_array, total = _compare_ne_starts_only(
                left=left,
                right=right,
                starts=starts,
                left_booleans=left_booleans,
                right_booleans=right_booleans,
                is_extension_array=is_extension_array,
                counts_array=counts_array,
                matches=matches,
                op=operator_mapping[op],
            )
    elif (starts is None) and (ends is not None) and (counts_array is not None):
        if (left_booleans is None) and (right_booleans is None):
            matches, counts_array, total = _compare_ends_only(
                left=left,
                right=right,
                ends=ends,
                counts_array=counts_array,
                matches=matches,
                op=operator_mapping[op],
            )
        else:
            matches, counts_array, total = _compare_ne_ends_only(
                left=left,
                right=right,
                ends=ends,
                left_booleans=left_booleans,
                right_booleans=right_booleans,
                is_extension_array=is_extension_array,
                counts_array=counts_array,
                matches=matches,
                op=operator_mapping[op],
            )
    elif (starts is not None) and (ends is not None) and (counts_array is None):
        if (left_booleans is None) and (right_booleans is None):
            matches, counts_array, total = _compare_first_run_starts_ends(
                left=left,
                right=right,
                starts=starts,
                ends=ends,
                op=operator_mapping[op],
            )
        else:
            matches, counts_array, total = _compare_ne_first_run_starts_ends(
                left=left,
                right=right,
                starts=starts,
                ends=ends,
                left_booleans=left_booleans,
                right_booleans=right_booleans,
                is_extension_array=is_extension_array,
                op=operator_mapping[op],
            )
    elif (starts is not None) and (ends is not None) and (counts_array is not None):
        if (left_booleans is None) and (right_booleans is None):
            matches, counts_array, total = _compare_starts_ends(
                left=left,
                right=right,
                starts=starts,
                ends=ends,
                matches=matches,
                op=operator_mapping[op],
            )
        else:
            matches, counts_array, total = _compare_ne_starts_ends(
                left=left,
                right=right,
                starts=starts,
                ends=ends,
                left_booleans=left_booleans,
                right_booleans=right_booleans,
                is_extension_array=is_extension_array,
                matches=matches,
                op=operator_mapping[op],
            )

    return matches, counts_array, total


def _get_positive_matches_conditions(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    left_index: np.ndarray,
    starts: np.ndarray | None,
    ends: np.ndarray | None,
):
    """
    Get positive matches for conditions
    """
    counts_array = None
    matches = None
    (left_on, right_on, op), *rest = conditions
    left_array = df.loc[left_index, left_on]
    right_array = right[right_on]
    left_booleans, right_booleans, is_extension_array = _get_boolean_args_for_ne(
        op=op, left=left_array, right=right_array
    )
    left_array = _convert_array_to_numpy(array=left_array._values)
    right_array = _convert_array_to_numpy(array=right_array._values)
    matches, counts_array, total = _get_positive_matches(
        left=left_array,
        right=right_array,
        starts=starts,
        ends=ends,
        counts_array=counts_array,
        matches=matches,
        op=op,
        left_booleans=left_booleans,
        right_booleans=right_booleans,
        is_extension_array=is_extension_array,
    )
    if total == 0:
        return None
    for left_on, right_on, op in rest:
        left_array = df.loc[left_index, left_on]
        right_array = right[right_on]
        left_booleans, right_booleans, is_extension_array = _get_boolean_args_for_ne(
            op=op, left=left_array, right=right_array
        )
        left_array = _convert_array_to_numpy(array=left_array._values)
        right_array = _convert_array_to_numpy(array=right_array._values)
        matches, counts_array, total = _get_positive_matches(
            left=left_array,
            right=right_array,
            starts=starts,
            ends=ends,
            counts_array=counts_array,
            matches=matches,
            op=op,
            left_booleans=left_booleans,
            right_booleans=right_booleans,
            is_extension_array=is_extension_array,
        )
        if total == 0:
            return None
    return {"matches": matches, "counts_array": counts_array, "total": total}


def _get_boolean_args_for_ne(
    op: str, left: np.ndarray | None, right: np.ndarray | None
) -> tuple:
    """
    Get boolean arguments for !=
    """
    if op != "!=":
        return None, None, False
    left_booleans = left.isna()
    right_booleans = right.isna()
    if not any((left_booleans.any(), right_booleans.any())):
        return None, None, False
    is_extension_array = pd.api.types.is_extension_array_dtype(left)
    left_booleans = left_booleans.to_numpy(na_value=False, copy=False, dtype=np.bool_)
    right_booleans = right_booleans.to_numpy(na_value=False, copy=False, dtype=np.bool_)
    return left_booleans, right_booleans, is_extension_array


def _build_indices_positions(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    counts_array: np.ndarray,
    total: int,
    keep: str,
):
    """
    Build indices for multiple joins
    """
    if keep == "all":
        left_index = janitor_rs.repeat_index(
            index=left_index, counts=counts_array, length=total
        )
        right_index = janitor_rs.build_positional_index(
            index=right_index, positions=positions, length=total
        )
    elif keep == "first":
        total = np.count_nonzero(counts_array)
        left_index = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right_index = janitor_rs.build_positional_index_first(
            index=right_index,
            starts=starts,
            ends=ends,
            counts=counts_array,
            positions=positions,
            length=total,
        )
    else:
        total = np.count_nonzero(counts_array)
        left_index = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right_index = janitor_rs.build_positional_index_last(
            index=right_index,
            starts=starts,
            ends=ends,
            counts=counts_array,
            positions=positions,
            length=total,
        )
    return {"left_index": left_index, "right_index": right_index}


def build_indices_matches(
    left_index: np.ndarray,
    right_index: np.ndarray,
    counts_array: np.ndarray,
    starts: np.ndarray | None,
    ends: np.ndarray | None,
    matches: np.ndarray,
    total: int,
    keep: str,
) -> dict:
    """
    Build indices for multiple joins, where `matches` exist
    """
    # return starts, ends, counts_array, matches, total, left_index
    if (keep == "all") and (starts is not None) and (ends is None):
        left = janitor_rs.repeat_index(
            index=left_index,
            counts=counts_array,
            length=total,
        )
        right = janitor_rs.index_starts_only(
            index=right_index, starts=starts, matches=matches, length=total
        )
    elif (keep == "all") and (starts is None) and (ends is not None):
        left = janitor_rs.repeat_index(
            index=left_index,
            counts=counts_array,
            length=total,
        )
        right = janitor_rs.index_ends_only(
            index=right_index, ends=ends, matches=matches, length=total
        )
    elif (keep == "all") and (starts is not None) and (ends is not None):
        left = janitor_rs.repeat_index(
            index=left_index,
            counts=counts_array,
            length=total,
        )
        right = janitor_rs.index_starts_and_ends(
            index=right_index,
            starts=starts,
            ends=ends,
            matches=matches,
            length=total,
        )

    elif (keep == "first") and (starts is not None) and (ends is None):
        total = np.count_nonzero(counts_array)
        left_index = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right = janitor_rs.index_starts_only_keep_first(
            index=right_index, starts=starts, matches=matches, length=total
        )
    elif (keep == "first") and (starts is None) and (ends is not None):
        total = np.count_nonzero(counts_array)
        left_index = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right = janitor_rs.index_ends_only_keep_first(
            index=right_index, ends=ends, matches=matches, length=total
        )
    elif (keep == "first") and (starts is not None) and (ends is not None):
        total = np.count_nonzero(counts_array)
        left = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right = janitor_rs.index_starts_and_ends_keep_first(
            index=right_index,
            starts=starts,
            ends=ends,
            counts=counts_array,
            matches=matches,
            length=total,
        )

    elif (keep == "last") and (starts is not None) and (ends is None):
        total = np.count_nonzero(counts_array)
        left = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right = janitor_rs.index_starts_only_keep_last(
            index=right_index, starts=starts, matches=matches, length=total
        )
    elif (keep == "last") and (starts is None) and (ends is not None):
        total = np.count_nonzero(counts_array)
        left = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right = janitor_rs.index_ends_only_keep_last(
            index=right_index, ends=ends, matches=matches, length=total
        )
    elif (keep == "last") and (starts is not None) and (ends is not None):
        total = np.count_nonzero(counts_array)
        left = janitor_rs.trim_index(
            index=left_index, counts=counts_array, length=total
        )
        right = janitor_rs.index_starts_and_ends_keep_last(
            index=right_index,
            starts=starts,
            ends=ends,
            matches=matches,
            length=total,
            counts=counts_array,
        )
    return {"left_index": left, "right_index": right}
