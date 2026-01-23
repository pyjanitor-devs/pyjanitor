import janitor_rs
import numpy as np


def _sum_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_start_int64,
        "int32": janitor_rs.compute_sum_start_int32,
        "int16": janitor_rs.compute_sum_start_int16,
        "int8": janitor_rs.compute_sum_start_int8,
        "uint64": janitor_rs.compute_sum_start_uint64,
        "uint32": janitor_rs.compute_sum_start_uint32,
        "uint16": janitor_rs.compute_sum_start_uint16,
        "uint8": janitor_rs.compute_sum_start_uint8,
        "float64": janitor_rs.compute_sum_start_f64,
        "float32": janitor_rs.compute_sum_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, booleans=booleans)


def _sum_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_end_int64,
        "int32": janitor_rs.compute_sum_end_int32,
        "int16": janitor_rs.compute_sum_end_int16,
        "int8": janitor_rs.compute_sum_end_int8,
        "uint64": janitor_rs.compute_sum_end_uint64,
        "uint32": janitor_rs.compute_sum_end_uint32,
        "uint16": janitor_rs.compute_sum_end_uint16,
        "uint8": janitor_rs.compute_sum_end_uint8,
        "float64": janitor_rs.compute_sum_end_f64,
        "float32": janitor_rs.compute_sum_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, booleans=booleans)


def _size_rev_starts(
    starts: np.ndarray,
    index: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start(starts=starts, index=index, length=length)


def _size_rev_ends(
    ends: np.ndarray,
    index: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_end(ends=ends, index=index, length=length)


def _size_rev_starts_ends(
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start_end(
        starts=starts, ends=ends, index=index, length=length
    )


def _size_rev_ends_matches(
    ends: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_end_matches(
        ends=ends, index=index, matches=matches, length=length
    )


def _size_rev_starts_matches(
    starts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start_matches(
        starts=starts, index=index, matches=matches, length=length
    )


def _size_rev_starts_ends_matches(
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_start_end_matches(
        starts=starts, ends=ends, index=index, matches=matches, length=length
    )


def _size_rev_positions(
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute size_rev
    """
    return janitor_rs.compute_size_rev_positions(
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        length=length,
    )


def _min_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_int64,
        "int32": janitor_rs.compute_min_start_int32,
        "int16": janitor_rs.compute_min_start_int16,
        "int8": janitor_rs.compute_min_start_int8,
        "uint64": janitor_rs.compute_min_start_uint64,
        "uint32": janitor_rs.compute_min_start_uint32,
        "uint16": janitor_rs.compute_min_start_uint16,
        "uint8": janitor_rs.compute_min_start_uint8,
        "float64": janitor_rs.compute_min_start_f64,
        "float32": janitor_rs.compute_min_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, booleans=booleans)


def _min_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_end_int64,
        "int32": janitor_rs.compute_min_end_int32,
        "int16": janitor_rs.compute_min_end_int16,
        "int8": janitor_rs.compute_min_end_int8,
        "uint64": janitor_rs.compute_min_end_uint64,
        "uint32": janitor_rs.compute_min_end_uint32,
        "uint16": janitor_rs.compute_min_end_uint16,
        "uint8": janitor_rs.compute_min_end_uint8,
        "float64": janitor_rs.compute_min_end_f64,
        "float32": janitor_rs.compute_min_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, booleans=booleans)


def _max_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_int64,
        "int32": janitor_rs.compute_max_start_int32,
        "int16": janitor_rs.compute_max_start_int16,
        "int8": janitor_rs.compute_max_start_int8,
        "uint64": janitor_rs.compute_max_start_uint64,
        "uint32": janitor_rs.compute_max_start_uint32,
        "uint16": janitor_rs.compute_max_start_uint16,
        "uint8": janitor_rs.compute_max_start_uint8,
        "float64": janitor_rs.compute_max_start_f64,
        "float32": janitor_rs.compute_max_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, booleans=booleans)


def _max_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_end_int64,
        "int32": janitor_rs.compute_max_end_int32,
        "int16": janitor_rs.compute_max_end_int16,
        "int8": janitor_rs.compute_max_end_int8,
        "uint64": janitor_rs.compute_max_end_uint64,
        "uint32": janitor_rs.compute_max_end_uint32,
        "uint16": janitor_rs.compute_max_end_uint16,
        "uint8": janitor_rs.compute_max_end_uint8,
        "float64": janitor_rs.compute_max_end_f64,
        "float32": janitor_rs.compute_max_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, booleans=booleans)


def _prod_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_int64,
        "int32": janitor_rs.compute_prod_start_int32,
        "int16": janitor_rs.compute_prod_start_int16,
        "int8": janitor_rs.compute_prod_start_int8,
        "uint64": janitor_rs.compute_prod_start_uint64,
        "uint32": janitor_rs.compute_prod_start_uint32,
        "uint16": janitor_rs.compute_prod_start_uint16,
        "uint8": janitor_rs.compute_prod_start_uint8,
        "float64": janitor_rs.compute_prod_start_f64,
        "float32": janitor_rs.compute_prod_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, booleans=booleans)


def _prod_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_end_int64,
        "int32": janitor_rs.compute_prod_end_int32,
        "int16": janitor_rs.compute_prod_end_int16,
        "int8": janitor_rs.compute_prod_end_int8,
        "uint64": janitor_rs.compute_prod_end_uint64,
        "uint32": janitor_rs.compute_prod_end_uint32,
        "uint16": janitor_rs.compute_prod_end_uint16,
        "uint8": janitor_rs.compute_prod_end_uint8,
        "float64": janitor_rs.compute_prod_end_f64,
        "float32": janitor_rs.compute_prod_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, booleans=booleans)


def _sum_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_start_match_int64,
        "int32": janitor_rs.compute_sum_start_match_int32,
        "int16": janitor_rs.compute_sum_start_match_int16,
        "int8": janitor_rs.compute_sum_start_match_int8,
        "uint64": janitor_rs.compute_sum_start_match_uint64,
        "uint32": janitor_rs.compute_sum_start_match_uint32,
        "uint16": janitor_rs.compute_sum_start_match_uint16,
        "uint8": janitor_rs.compute_sum_start_match_uint8,
        "float64": janitor_rs.compute_sum_start_match_f64,
        "float32": janitor_rs.compute_sum_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _sum_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_end_match_int64,
        "int32": janitor_rs.compute_sum_end_match_int32,
        "int16": janitor_rs.compute_sum_end_match_int16,
        "int8": janitor_rs.compute_sum_end_match_int8,
        "uint64": janitor_rs.compute_sum_end_match_uint64,
        "uint32": janitor_rs.compute_sum_end_match_uint32,
        "uint16": janitor_rs.compute_sum_end_match_uint16,
        "uint8": janitor_rs.compute_sum_end_match_uint8,
        "float64": janitor_rs.compute_sum_end_match_f64,
        "float32": janitor_rs.compute_sum_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _max_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_match_int64,
        "int32": janitor_rs.compute_max_start_match_int32,
        "int16": janitor_rs.compute_max_start_match_int16,
        "int8": janitor_rs.compute_max_start_match_int8,
        "uint64": janitor_rs.compute_max_start_match_uint64,
        "uint32": janitor_rs.compute_max_start_match_uint32,
        "uint16": janitor_rs.compute_max_start_match_uint16,
        "uint8": janitor_rs.compute_max_start_match_uint8,
        "float64": janitor_rs.compute_max_start_match_f64,
        "float32": janitor_rs.compute_max_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _max_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_end_match_int64,
        "int32": janitor_rs.compute_max_end_match_int32,
        "int16": janitor_rs.compute_max_end_match_int16,
        "int8": janitor_rs.compute_max_end_match_int8,
        "uint64": janitor_rs.compute_max_end_match_uint64,
        "uint32": janitor_rs.compute_max_end_match_uint32,
        "uint16": janitor_rs.compute_max_end_match_uint16,
        "uint8": janitor_rs.compute_max_end_match_uint8,
        "float64": janitor_rs.compute_max_end_match_f64,
        "float32": janitor_rs.compute_max_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _min_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_match_int64,
        "int32": janitor_rs.compute_min_start_match_int32,
        "int16": janitor_rs.compute_min_start_match_int16,
        "int8": janitor_rs.compute_min_start_match_int8,
        "uint64": janitor_rs.compute_min_start_match_uint64,
        "uint32": janitor_rs.compute_min_start_match_uint32,
        "uint16": janitor_rs.compute_min_start_match_uint16,
        "uint8": janitor_rs.compute_min_start_match_uint8,
        "float64": janitor_rs.compute_min_start_match_f64,
        "float32": janitor_rs.compute_min_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _min_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_end_match_int64,
        "int32": janitor_rs.compute_min_end_match_int32,
        "int16": janitor_rs.compute_min_end_match_int16,
        "int8": janitor_rs.compute_min_end_match_int8,
        "uint64": janitor_rs.compute_min_end_match_uint64,
        "uint32": janitor_rs.compute_min_end_match_uint32,
        "uint16": janitor_rs.compute_min_end_match_uint16,
        "uint8": janitor_rs.compute_min_end_match_uint8,
        "float64": janitor_rs.compute_min_end_match_f64,
        "float32": janitor_rs.compute_min_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _sum_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_positions_int64,
        "int32": janitor_rs.compute_sum_positions_int32,
        "int16": janitor_rs.compute_sum_positions_int16,
        "int8": janitor_rs.compute_sum_positions_int8,
        "uint64": janitor_rs.compute_sum_positions_uint64,
        "uint32": janitor_rs.compute_sum_positions_uint32,
        "uint16": janitor_rs.compute_sum_positions_uint16,
        "uint8": janitor_rs.compute_sum_positions_uint8,
        "float64": janitor_rs.compute_sum_positions_f64,
        "float32": janitor_rs.compute_sum_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _prod_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_match_int64,
        "int32": janitor_rs.compute_prod_start_match_int32,
        "int16": janitor_rs.compute_prod_start_match_int16,
        "int8": janitor_rs.compute_prod_start_match_int8,
        "uint64": janitor_rs.compute_prod_start_match_uint64,
        "uint32": janitor_rs.compute_prod_start_match_uint32,
        "uint16": janitor_rs.compute_prod_start_match_uint16,
        "uint8": janitor_rs.compute_prod_start_match_uint8,
        "float64": janitor_rs.compute_prod_start_match_f64,
        "float32": janitor_rs.compute_prod_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _prod_ends_matches(
    arr: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_end_match_int64,
        "int32": janitor_rs.compute_prod_end_match_int32,
        "int16": janitor_rs.compute_prod_end_match_int16,
        "int8": janitor_rs.compute_prod_end_match_int8,
        "uint64": janitor_rs.compute_prod_end_match_uint64,
        "uint32": janitor_rs.compute_prod_end_match_uint32,
        "uint16": janitor_rs.compute_prod_end_match_uint16,
        "uint8": janitor_rs.compute_prod_end_match_uint8,
        "float64": janitor_rs.compute_prod_end_match_f64,
        "float32": janitor_rs.compute_prod_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, counts=counts, matches=matches, booleans=booleans)


def _prod_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_positions_int64,
        "int32": janitor_rs.compute_prod_positions_int32,
        "int16": janitor_rs.compute_prod_positions_int16,
        "int8": janitor_rs.compute_prod_positions_int8,
        "uint64": janitor_rs.compute_prod_positions_uint64,
        "uint32": janitor_rs.compute_prod_positions_uint32,
        "uint16": janitor_rs.compute_prod_positions_uint16,
        "uint8": janitor_rs.compute_prod_positions_uint8,
        "float64": janitor_rs.compute_prod_positions_f64,
        "float32": janitor_rs.compute_prod_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _min_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_positions_int64,
        "int32": janitor_rs.compute_min_positions_int32,
        "int16": janitor_rs.compute_min_positions_int16,
        "int8": janitor_rs.compute_min_positions_int8,
        "uint64": janitor_rs.compute_min_positions_uint64,
        "uint32": janitor_rs.compute_min_positions_uint32,
        "uint16": janitor_rs.compute_min_positions_uint16,
        "uint8": janitor_rs.compute_min_positions_uint8,
        "float64": janitor_rs.compute_min_positions_f64,
        "float32": janitor_rs.compute_min_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _max_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_positions_int64,
        "int32": janitor_rs.compute_max_positions_int32,
        "int16": janitor_rs.compute_max_positions_int16,
        "int8": janitor_rs.compute_max_positions_int8,
        "uint64": janitor_rs.compute_max_positions_uint64,
        "uint32": janitor_rs.compute_max_positions_uint32,
        "uint16": janitor_rs.compute_max_positions_uint16,
        "uint8": janitor_rs.compute_max_positions_uint8,
        "float64": janitor_rs.compute_max_positions_f64,
        "float32": janitor_rs.compute_max_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        positions=positions,
        booleans=booleans,
    )


def _max_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_end_int64,
        "int32": janitor_rs.compute_max_start_end_int32,
        "int16": janitor_rs.compute_max_start_end_int16,
        "int8": janitor_rs.compute_max_start_end_int8,
        "uint64": janitor_rs.compute_max_start_end_uint64,
        "uint32": janitor_rs.compute_max_start_end_uint32,
        "uint16": janitor_rs.compute_max_start_end_uint16,
        "uint8": janitor_rs.compute_max_start_end_uint8,
        "float64": janitor_rs.compute_max_start_end_f64,
        "float32": janitor_rs.compute_max_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _min_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_end_int64,
        "int32": janitor_rs.compute_min_start_end_int32,
        "int16": janitor_rs.compute_min_start_end_int16,
        "int8": janitor_rs.compute_min_start_end_int8,
        "uint64": janitor_rs.compute_min_start_end_uint64,
        "uint32": janitor_rs.compute_min_start_end_uint32,
        "uint16": janitor_rs.compute_min_start_end_uint16,
        "uint8": janitor_rs.compute_min_start_end_uint8,
        "float64": janitor_rs.compute_min_start_end_f64,
        "float32": janitor_rs.compute_min_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _sum_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_start_end_int64,
        "int32": janitor_rs.compute_sum_start_end_int32,
        "int16": janitor_rs.compute_sum_start_end_int16,
        "int8": janitor_rs.compute_sum_start_end_int8,
        "uint64": janitor_rs.compute_sum_start_end_uint64,
        "uint32": janitor_rs.compute_sum_start_end_uint32,
        "uint16": janitor_rs.compute_sum_start_end_uint16,
        "uint8": janitor_rs.compute_sum_start_end_uint8,
        "float64": janitor_rs.compute_sum_start_end_f64,
        "float32": janitor_rs.compute_sum_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _prod_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_end_int64,
        "int32": janitor_rs.compute_prod_start_end_int32,
        "int16": janitor_rs.compute_prod_start_end_int16,
        "int8": janitor_rs.compute_prod_start_end_int8,
        "uint64": janitor_rs.compute_prod_start_end_uint64,
        "uint32": janitor_rs.compute_prod_start_end_uint32,
        "uint16": janitor_rs.compute_prod_start_end_uint16,
        "uint8": janitor_rs.compute_prod_start_end_uint8,
        "float64": janitor_rs.compute_prod_start_end_f64,
        "float32": janitor_rs.compute_prod_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _prod_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_start_end_match_int64,
        "int32": janitor_rs.compute_prod_start_end_match_int32,
        "int16": janitor_rs.compute_prod_start_end_match_int16,
        "int8": janitor_rs.compute_prod_start_end_match_int8,
        "uint64": janitor_rs.compute_prod_start_end_match_uint64,
        "uint32": janitor_rs.compute_prod_start_end_match_uint32,
        "uint16": janitor_rs.compute_prod_start_end_match_uint16,
        "uint8": janitor_rs.compute_prod_start_end_match_uint8,
        "float64": janitor_rs.compute_prod_start_end_match_f64,
        "float32": janitor_rs.compute_prod_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _sum_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_start_end_int64,
        "int32": janitor_rs.compute_sum_start_end_int32,
        "int16": janitor_rs.compute_sum_start_end_int16,
        "int8": janitor_rs.compute_sum_start_end_int8,
        "uint64": janitor_rs.compute_sum_start_end_uint64,
        "uint32": janitor_rs.compute_sum_start_end_uint32,
        "uint16": janitor_rs.compute_sum_start_end_uint16,
        "uint8": janitor_rs.compute_sum_start_end_uint8,
        "float64": janitor_rs.compute_sum_start_end_f64,
        "float32": janitor_rs.compute_sum_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, ends=ends, booleans=booleans)


def _sum_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_start_end_match_int64,
        "int32": janitor_rs.compute_sum_start_end_match_int32,
        "int16": janitor_rs.compute_sum_start_end_match_int16,
        "int8": janitor_rs.compute_sum_start_end_match_int8,
        "uint64": janitor_rs.compute_sum_start_end_match_uint64,
        "uint32": janitor_rs.compute_sum_start_end_match_uint32,
        "uint16": janitor_rs.compute_sum_start_end_match_uint16,
        "uint8": janitor_rs.compute_sum_start_end_match_uint8,
        "float64": janitor_rs.compute_sum_start_end_match_f64,
        "float32": janitor_rs.compute_sum_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _min_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_start_end_match_int64,
        "int32": janitor_rs.compute_min_start_end_match_int32,
        "int16": janitor_rs.compute_min_start_end_match_int16,
        "int8": janitor_rs.compute_min_start_end_match_int8,
        "uint64": janitor_rs.compute_min_start_end_match_uint64,
        "uint32": janitor_rs.compute_min_start_end_match_uint32,
        "uint16": janitor_rs.compute_min_start_end_match_uint16,
        "uint8": janitor_rs.compute_min_start_end_match_uint8,
        "float64": janitor_rs.compute_min_start_end_match_f64,
        "float32": janitor_rs.compute_min_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _max_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_start_end_match_int64,
        "int32": janitor_rs.compute_max_start_end_match_int32,
        "int16": janitor_rs.compute_max_start_end_match_int16,
        "int8": janitor_rs.compute_max_start_end_match_int8,
        "uint64": janitor_rs.compute_max_start_end_match_uint64,
        "uint32": janitor_rs.compute_max_start_end_match_uint32,
        "uint16": janitor_rs.compute_max_start_end_match_uint16,
        "uint8": janitor_rs.compute_max_start_end_match_uint8,
        "float64": janitor_rs.compute_max_start_end_match_f64,
        "float32": janitor_rs.compute_max_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


def _prod_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_int64,
        "int32": janitor_rs.compute_prod_rev_start_int32,
        "int16": janitor_rs.compute_prod_rev_start_int16,
        "int8": janitor_rs.compute_prod_rev_start_int8,
        "uint64": janitor_rs.compute_prod_rev_start_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_uint8,
        "float64": janitor_rs.compute_prod_rev_start_f64,
        "float32": janitor_rs.compute_prod_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _prod_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_end_int64,
        "int32": janitor_rs.compute_prod_rev_end_int32,
        "int16": janitor_rs.compute_prod_rev_end_int16,
        "int8": janitor_rs.compute_prod_rev_end_int8,
        "uint64": janitor_rs.compute_prod_rev_end_uint64,
        "uint32": janitor_rs.compute_prod_rev_end_uint32,
        "uint16": janitor_rs.compute_prod_rev_end_uint16,
        "uint8": janitor_rs.compute_prod_rev_end_uint8,
        "float64": janitor_rs.compute_prod_rev_end_f64,
        "float32": janitor_rs.compute_prod_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _prod_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_match_int64,
        "int32": janitor_rs.compute_prod_rev_start_match_int32,
        "int16": janitor_rs.compute_prod_rev_start_match_int16,
        "int8": janitor_rs.compute_prod_rev_start_match_int8,
        "uint64": janitor_rs.compute_prod_rev_start_match_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_match_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_match_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_match_uint8,
        "float64": janitor_rs.compute_prod_rev_start_match_f64,
        "float32": janitor_rs.compute_prod_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _prod_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_end_match_int64,
        "int32": janitor_rs.compute_prod_rev_end_match_int32,
        "int16": janitor_rs.compute_prod_rev_end_match_int16,
        "int8": janitor_rs.compute_prod_rev_end_match_int8,
        "uint64": janitor_rs.compute_prod_rev_end_match_uint64,
        "uint32": janitor_rs.compute_prod_rev_end_match_uint32,
        "uint16": janitor_rs.compute_prod_rev_end_match_uint16,
        "uint8": janitor_rs.compute_prod_rev_end_match_uint8,
        "float64": janitor_rs.compute_prod_rev_end_match_f64,
        "float32": janitor_rs.compute_prod_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _prod_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_positions_int64,
        "int32": janitor_rs.compute_prod_rev_positions_int32,
        "int16": janitor_rs.compute_prod_rev_positions_int16,
        "int8": janitor_rs.compute_prod_rev_positions_int8,
        "uint64": janitor_rs.compute_prod_rev_positions_uint64,
        "uint32": janitor_rs.compute_prod_rev_positions_uint32,
        "uint16": janitor_rs.compute_prod_rev_positions_uint16,
        "uint8": janitor_rs.compute_prod_rev_positions_uint8,
        "float64": janitor_rs.compute_prod_rev_positions_f64,
        "float32": janitor_rs.compute_prod_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _prod_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_end_int64,
        "int32": janitor_rs.compute_prod_rev_start_end_int32,
        "int16": janitor_rs.compute_prod_rev_start_end_int16,
        "int8": janitor_rs.compute_prod_rev_start_end_int8,
        "uint64": janitor_rs.compute_prod_rev_start_end_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_end_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_end_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_end_uint8,
        "float64": janitor_rs.compute_prod_rev_start_end_f64,
        "float32": janitor_rs.compute_prod_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr, starts=starts, ends=ends, index=index, booleans=booleans, length=length
    )


def _prod_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_start_end_match_int64,
        "int32": janitor_rs.compute_prod_rev_start_end_match_int32,
        "int16": janitor_rs.compute_prod_rev_start_end_match_int16,
        "int8": janitor_rs.compute_prod_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_prod_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_prod_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_prod_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_prod_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_prod_rev_start_end_match_f64,
        "float32": janitor_rs.compute_prod_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _min_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_int64,
        "int32": janitor_rs.compute_min_rev_start_int32,
        "int16": janitor_rs.compute_min_rev_start_int16,
        "int8": janitor_rs.compute_min_rev_start_int8,
        "uint64": janitor_rs.compute_min_rev_start_uint64,
        "uint32": janitor_rs.compute_min_rev_start_uint32,
        "uint16": janitor_rs.compute_min_rev_start_uint16,
        "uint8": janitor_rs.compute_min_rev_start_uint8,
        "float64": janitor_rs.compute_min_rev_start_f64,
        "float32": janitor_rs.compute_min_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _min_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_end_int64,
        "int32": janitor_rs.compute_min_rev_end_int32,
        "int16": janitor_rs.compute_min_rev_end_int16,
        "int8": janitor_rs.compute_min_rev_end_int8,
        "uint64": janitor_rs.compute_min_rev_end_uint64,
        "uint32": janitor_rs.compute_min_rev_end_uint32,
        "uint16": janitor_rs.compute_min_rev_end_uint16,
        "uint8": janitor_rs.compute_min_rev_end_uint8,
        "float64": janitor_rs.compute_min_rev_end_f64,
        "float32": janitor_rs.compute_min_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _min_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_match_int64,
        "int32": janitor_rs.compute_min_rev_start_match_int32,
        "int16": janitor_rs.compute_min_rev_start_match_int16,
        "int8": janitor_rs.compute_min_rev_start_match_int8,
        "uint64": janitor_rs.compute_min_rev_start_match_uint64,
        "uint32": janitor_rs.compute_min_rev_start_match_uint32,
        "uint16": janitor_rs.compute_min_rev_start_match_uint16,
        "uint8": janitor_rs.compute_min_rev_start_match_uint8,
        "float64": janitor_rs.compute_min_rev_start_match_f64,
        "float32": janitor_rs.compute_min_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _min_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_end_match_int64,
        "int32": janitor_rs.compute_min_rev_end_match_int32,
        "int16": janitor_rs.compute_min_rev_end_match_int16,
        "int8": janitor_rs.compute_min_rev_end_match_int8,
        "uint64": janitor_rs.compute_min_rev_end_match_uint64,
        "uint32": janitor_rs.compute_min_rev_end_match_uint32,
        "uint16": janitor_rs.compute_min_rev_end_match_uint16,
        "uint8": janitor_rs.compute_min_rev_end_match_uint8,
        "float64": janitor_rs.compute_min_rev_end_match_f64,
        "float32": janitor_rs.compute_min_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _min_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_positions_int64,
        "int32": janitor_rs.compute_min_rev_positions_int32,
        "int16": janitor_rs.compute_min_rev_positions_int16,
        "int8": janitor_rs.compute_min_rev_positions_int8,
        "uint64": janitor_rs.compute_min_rev_positions_uint64,
        "uint32": janitor_rs.compute_min_rev_positions_uint32,
        "uint16": janitor_rs.compute_min_rev_positions_uint16,
        "uint8": janitor_rs.compute_min_rev_positions_uint8,
        "float64": janitor_rs.compute_min_rev_positions_f64,
        "float32": janitor_rs.compute_min_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _min_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_end_int64,
        "int32": janitor_rs.compute_min_rev_start_end_int32,
        "int16": janitor_rs.compute_min_rev_start_end_int16,
        "int8": janitor_rs.compute_min_rev_start_end_int8,
        "uint64": janitor_rs.compute_min_rev_start_end_uint64,
        "uint32": janitor_rs.compute_min_rev_start_end_uint32,
        "uint16": janitor_rs.compute_min_rev_start_end_uint16,
        "uint8": janitor_rs.compute_min_rev_start_end_uint8,
        "float64": janitor_rs.compute_min_rev_start_end_f64,
        "float32": janitor_rs.compute_min_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr, starts=starts, ends=ends, index=index, booleans=booleans, length=length
    )


def _min_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_start_end_match_int64,
        "int32": janitor_rs.compute_min_rev_start_end_match_int32,
        "int16": janitor_rs.compute_min_rev_start_end_match_int16,
        "int8": janitor_rs.compute_min_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_min_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_min_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_min_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_min_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_min_rev_start_end_match_f64,
        "float32": janitor_rs.compute_min_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _max_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_int64,
        "int32": janitor_rs.compute_max_rev_start_int32,
        "int16": janitor_rs.compute_max_rev_start_int16,
        "int8": janitor_rs.compute_max_rev_start_int8,
        "uint64": janitor_rs.compute_max_rev_start_uint64,
        "uint32": janitor_rs.compute_max_rev_start_uint32,
        "uint16": janitor_rs.compute_max_rev_start_uint16,
        "uint8": janitor_rs.compute_max_rev_start_uint8,
        "float64": janitor_rs.compute_max_rev_start_f64,
        "float32": janitor_rs.compute_max_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _max_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_end_int64,
        "int32": janitor_rs.compute_max_rev_end_int32,
        "int16": janitor_rs.compute_max_rev_end_int16,
        "int8": janitor_rs.compute_max_rev_end_int8,
        "uint64": janitor_rs.compute_max_rev_end_uint64,
        "uint32": janitor_rs.compute_max_rev_end_uint32,
        "uint16": janitor_rs.compute_max_rev_end_uint16,
        "uint8": janitor_rs.compute_max_rev_end_uint8,
        "float64": janitor_rs.compute_max_rev_end_f64,
        "float32": janitor_rs.compute_max_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _max_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_match_int64,
        "int32": janitor_rs.compute_max_rev_start_match_int32,
        "int16": janitor_rs.compute_max_rev_start_match_int16,
        "int8": janitor_rs.compute_max_rev_start_match_int8,
        "uint64": janitor_rs.compute_max_rev_start_match_uint64,
        "uint32": janitor_rs.compute_max_rev_start_match_uint32,
        "uint16": janitor_rs.compute_max_rev_start_match_uint16,
        "uint8": janitor_rs.compute_max_rev_start_match_uint8,
        "float64": janitor_rs.compute_max_rev_start_match_f64,
        "float32": janitor_rs.compute_max_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _max_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_end_match_int64,
        "int32": janitor_rs.compute_max_rev_end_match_int32,
        "int16": janitor_rs.compute_max_rev_end_match_int16,
        "int8": janitor_rs.compute_max_rev_end_match_int8,
        "uint64": janitor_rs.compute_max_rev_end_match_uint64,
        "uint32": janitor_rs.compute_max_rev_end_match_uint32,
        "uint16": janitor_rs.compute_max_rev_end_match_uint16,
        "uint8": janitor_rs.compute_max_rev_end_match_uint8,
        "float64": janitor_rs.compute_max_rev_end_match_f64,
        "float32": janitor_rs.compute_max_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _max_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_positions_int64,
        "int32": janitor_rs.compute_max_rev_positions_int32,
        "int16": janitor_rs.compute_max_rev_positions_int16,
        "int8": janitor_rs.compute_max_rev_positions_int8,
        "uint64": janitor_rs.compute_max_rev_positions_uint64,
        "uint32": janitor_rs.compute_max_rev_positions_uint32,
        "uint16": janitor_rs.compute_max_rev_positions_uint16,
        "uint8": janitor_rs.compute_max_rev_positions_uint8,
        "float64": janitor_rs.compute_max_rev_positions_f64,
        "float32": janitor_rs.compute_max_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _max_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_end_int64,
        "int32": janitor_rs.compute_max_rev_start_end_int32,
        "int16": janitor_rs.compute_max_rev_start_end_int16,
        "int8": janitor_rs.compute_max_rev_start_end_int8,
        "uint64": janitor_rs.compute_max_rev_start_end_uint64,
        "uint32": janitor_rs.compute_max_rev_start_end_uint32,
        "uint16": janitor_rs.compute_max_rev_start_end_uint16,
        "uint8": janitor_rs.compute_max_rev_start_end_uint8,
        "float64": janitor_rs.compute_max_rev_start_end_f64,
        "float32": janitor_rs.compute_max_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr, starts=starts, ends=ends, index=index, booleans=booleans, length=length
    )


def _max_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_start_end_match_int64,
        "int32": janitor_rs.compute_max_rev_start_end_match_int32,
        "int16": janitor_rs.compute_max_rev_start_end_match_int16,
        "int8": janitor_rs.compute_max_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_max_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_max_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_max_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_max_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_max_rev_start_end_match_f64,
        "float32": janitor_rs.compute_max_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _prod_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute prod
    """
    mapping = {
        "int64": janitor_rs.compute_prod_rev_no_range_int64,
        "int32": janitor_rs.compute_prod_rev_no_range_int32,
        "int16": janitor_rs.compute_prod_rev_no_range_int16,
        "int8": janitor_rs.compute_prod_rev_no_range_int8,
        "uint64": janitor_rs.compute_prod_rev_no_range_uint64,
        "uint32": janitor_rs.compute_prod_rev_no_range_uint32,
        "uint16": janitor_rs.compute_prod_rev_no_range_uint16,
        "uint8": janitor_rs.compute_prod_rev_no_range_uint8,
        "float64": janitor_rs.compute_prod_rev_no_range_f64,
        "float32": janitor_rs.compute_prod_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _max_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute max
    """
    mapping = {
        "int64": janitor_rs.compute_max_rev_no_range_int64,
        "int32": janitor_rs.compute_max_rev_no_range_int32,
        "int16": janitor_rs.compute_max_rev_no_range_int16,
        "int8": janitor_rs.compute_max_rev_no_range_int8,
        "uint64": janitor_rs.compute_max_rev_no_range_uint64,
        "uint32": janitor_rs.compute_max_rev_no_range_uint32,
        "uint16": janitor_rs.compute_max_rev_no_range_uint16,
        "uint8": janitor_rs.compute_max_rev_no_range_uint8,
        "float64": janitor_rs.compute_max_rev_no_range_f64,
        "float32": janitor_rs.compute_max_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _min_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute min
    """
    mapping = {
        "int64": janitor_rs.compute_min_rev_no_range_int64,
        "int32": janitor_rs.compute_min_rev_no_range_int32,
        "int16": janitor_rs.compute_min_rev_no_range_int16,
        "int8": janitor_rs.compute_min_rev_no_range_int8,
        "uint64": janitor_rs.compute_min_rev_no_range_uint64,
        "uint32": janitor_rs.compute_min_rev_no_range_uint32,
        "uint16": janitor_rs.compute_min_rev_no_range_uint16,
        "uint8": janitor_rs.compute_min_rev_no_range_uint8,
        "float64": janitor_rs.compute_min_rev_no_range_f64,
        "float32": janitor_rs.compute_min_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _sum_rev_no_ranges(
    arr: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_no_range_int64,
        "int32": janitor_rs.compute_sum_rev_no_range_int32,
        "int16": janitor_rs.compute_sum_rev_no_range_int16,
        "int8": janitor_rs.compute_sum_rev_no_range_int8,
        "uint64": janitor_rs.compute_sum_rev_no_range_uint64,
        "uint32": janitor_rs.compute_sum_rev_no_range_uint32,
        "uint16": janitor_rs.compute_sum_rev_no_range_uint16,
        "uint8": janitor_rs.compute_sum_rev_no_range_uint8,
        "float64": janitor_rs.compute_sum_rev_no_range_f64,
        "float32": janitor_rs.compute_sum_rev_no_range_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
        length=length,
    )


def _sum_rev_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_int64,
        "int32": janitor_rs.compute_sum_rev_start_int32,
        "int16": janitor_rs.compute_sum_rev_start_int16,
        "int8": janitor_rs.compute_sum_rev_start_int8,
        "uint64": janitor_rs.compute_sum_rev_start_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_uint8,
        "float64": janitor_rs.compute_sum_rev_start_f64,
        "float32": janitor_rs.compute_sum_rev_start_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, starts=starts, index=index, booleans=booleans, length=length)


def _sum_rev_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_end_int64,
        "int32": janitor_rs.compute_sum_rev_end_int32,
        "int16": janitor_rs.compute_sum_rev_end_int16,
        "int8": janitor_rs.compute_sum_rev_end_int8,
        "uint64": janitor_rs.compute_sum_rev_end_uint64,
        "uint32": janitor_rs.compute_sum_rev_end_uint32,
        "uint16": janitor_rs.compute_sum_rev_end_uint16,
        "uint8": janitor_rs.compute_sum_rev_end_uint8,
        "float64": janitor_rs.compute_sum_rev_end_f64,
        "float32": janitor_rs.compute_sum_rev_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(arr=arr, ends=ends, index=index, booleans=booleans, length=length)


def _sum_rev_starts_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    index: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_match_int64,
        "int32": janitor_rs.compute_sum_rev_start_match_int32,
        "int16": janitor_rs.compute_sum_rev_start_match_int16,
        "int8": janitor_rs.compute_sum_rev_start_match_int8,
        "uint64": janitor_rs.compute_sum_rev_start_match_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_match_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_match_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_match_uint8,
        "float64": janitor_rs.compute_sum_rev_start_match_f64,
        "float32": janitor_rs.compute_sum_rev_start_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _sum_rev_ends_matches(
    arr: np.ndarray,
    index: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_end_match_int64,
        "int32": janitor_rs.compute_sum_rev_end_match_int32,
        "int16": janitor_rs.compute_sum_rev_end_match_int16,
        "int8": janitor_rs.compute_sum_rev_end_match_int8,
        "uint64": janitor_rs.compute_sum_rev_end_match_uint64,
        "uint32": janitor_rs.compute_sum_rev_end_match_uint32,
        "uint16": janitor_rs.compute_sum_rev_end_match_uint16,
        "uint8": janitor_rs.compute_sum_rev_end_match_uint8,
        "float64": janitor_rs.compute_sum_rev_end_match_f64,
        "float32": janitor_rs.compute_sum_rev_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )


def _sum_rev_positions(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    positions: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_positions_int64,
        "int32": janitor_rs.compute_sum_rev_positions_int32,
        "int16": janitor_rs.compute_sum_rev_positions_int16,
        "int8": janitor_rs.compute_sum_rev_positions_int8,
        "uint64": janitor_rs.compute_sum_rev_positions_uint64,
        "uint32": janitor_rs.compute_sum_rev_positions_uint32,
        "uint16": janitor_rs.compute_sum_rev_positions_uint16,
        "uint8": janitor_rs.compute_sum_rev_positions_uint8,
        "float64": janitor_rs.compute_sum_rev_positions_f64,
        "float32": janitor_rs.compute_sum_rev_positions_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
        length=length,
    )


def _sum_rev_starts_ends(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_end_int64,
        "int32": janitor_rs.compute_sum_rev_start_end_int32,
        "int16": janitor_rs.compute_sum_rev_start_end_int16,
        "int8": janitor_rs.compute_sum_rev_start_end_int8,
        "uint64": janitor_rs.compute_sum_rev_start_end_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_end_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_end_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_end_uint8,
        "float64": janitor_rs.compute_sum_rev_start_end_f64,
        "float32": janitor_rs.compute_sum_rev_start_end_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr, starts=starts, ends=ends, index=index, booleans=booleans, length=length
    )


def _sum_rev_starts_ends_matches(
    arr: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    index: np.ndarray,
    counts: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    length: int,
) -> tuple:
    """
    Compute sum
    """
    mapping = {
        "int64": janitor_rs.compute_sum_rev_start_end_match_int64,
        "int32": janitor_rs.compute_sum_rev_start_end_match_int32,
        "int16": janitor_rs.compute_sum_rev_start_end_match_int16,
        "int8": janitor_rs.compute_sum_rev_start_end_match_int8,
        "uint64": janitor_rs.compute_sum_rev_start_end_match_uint64,
        "uint32": janitor_rs.compute_sum_rev_start_end_match_uint32,
        "uint16": janitor_rs.compute_sum_rev_start_end_match_uint16,
        "uint8": janitor_rs.compute_sum_rev_start_end_match_uint8,
        "float64": janitor_rs.compute_sum_rev_start_end_match_f64,
        "float32": janitor_rs.compute_sum_rev_start_end_match_f32,
    }
    dtype_name = arr.dtype.name
    func = mapping[dtype_name]
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
        length=length,
    )
