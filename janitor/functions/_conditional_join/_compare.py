import numpy as np
import janitor_rs





def _compare_ne_no_ranges(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    left_booleans: np.ndarray,
    right_booleans: np.ndarray,
    is_extension_array: bool,
    op: int,
) -> tuple:
    """
    Compute comparsons for no ranges (no starts/ends) for != operator
    """
    mapping = {
        "int64": janitor_rs.compare_no_range_ne_int64,
        "int32": janitor_rs.compare_no_range_ne_int32,
        "int16": janitor_rs.compare_no_range_ne_int16,
        "int8": janitor_rs.compare_no_range_ne_int8,
        "uint64": janitor_rs.compare_no_range_ne_uint64,
        "uint32": janitor_rs.compare_no_range_ne_uint32,
        "uint16": janitor_rs.compare_no_range_ne_uint16,
        "uint8": janitor_rs.compare_no_range_ne_uint8,
        "float64": janitor_rs.compare_no_range_ne_float64,
        "float32": janitor_rs.compare_no_range_ne_float32,
    }
    dtype_name = left.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    )


def _compare_no_ranges(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    op: int,
) -> tuple:
    """
    Compute comparsons for no ranges (no starts/ends)
    """
    mapping = {
        "int64": janitor_rs.compare_no_range_int64,
        "int32": janitor_rs.compare_no_range_int32,
        "int16": janitor_rs.compare_no_range_int16,
        "int8": janitor_rs.compare_no_range_int8,
        "uint64": janitor_rs.compare_no_range_uint64,
        "uint32": janitor_rs.compare_no_range_uint32,
        "uint16": janitor_rs.compare_no_range_uint16,
        "uint8": janitor_rs.compare_no_range_uint8,
        "float64": janitor_rs.compare_no_range_float64,
        "float32": janitor_rs.compare_no_range_float32,
    }
    dtype_name = left.dtype.name
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
    return func(
        left,
        right,
        positions,
        op,
    )


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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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
    try:
        func = mapping[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}")
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