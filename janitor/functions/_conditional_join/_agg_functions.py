import janitor_rs
import numpy as np

from ._dtype_dispatch import _rs_func


def _sum_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    func = _rs_func("compute_sum_start", arr.dtype.name)
    return func(arr=arr, starts=starts, booleans=booleans)


def _sum_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute sum
    """
    func = _rs_func("compute_sum_end", arr.dtype.name)
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
    func = _rs_func("compute_min_start", arr.dtype.name)
    return func(arr=arr, starts=starts, booleans=booleans)


def _min_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute min
    """
    func = _rs_func("compute_min_end", arr.dtype.name)
    return func(arr=arr, ends=ends, booleans=booleans)


def _max_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    func = _rs_func("compute_max_start", arr.dtype.name)
    return func(arr=arr, starts=starts, booleans=booleans)


def _max_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute max
    """
    func = _rs_func("compute_max_end", arr.dtype.name)
    return func(arr=arr, ends=ends, booleans=booleans)


def _prod_starts(
    arr: np.ndarray,
    starts: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    func = _rs_func("compute_prod_start", arr.dtype.name)
    return func(arr=arr, starts=starts, booleans=booleans)


def _prod_ends(
    arr: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
) -> tuple:
    """
    Compute prod
    """
    func = _rs_func("compute_prod_end", arr.dtype.name)
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
    func = _rs_func("compute_sum_start_match", arr.dtype.name)
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
    func = _rs_func("compute_sum_end_match", arr.dtype.name)
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
    func = _rs_func("compute_max_start_match", arr.dtype.name)
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
    func = _rs_func("compute_max_end_match", arr.dtype.name)
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
    func = _rs_func("compute_min_start_match", arr.dtype.name)
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
    func = _rs_func("compute_min_end_match", arr.dtype.name)
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
    func = _rs_func("compute_sum_positions", arr.dtype.name)
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
    func = _rs_func("compute_prod_start_match", arr.dtype.name)
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
    func = _rs_func("compute_prod_end_match", arr.dtype.name)
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
    func = _rs_func("compute_prod_positions", arr.dtype.name)
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
    func = _rs_func("compute_min_positions", arr.dtype.name)
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
    func = _rs_func("compute_max_positions", arr.dtype.name)
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
    func = _rs_func("compute_max_start_end", arr.dtype.name)
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
    func = _rs_func("compute_min_start_end", arr.dtype.name)
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
    func = _rs_func("compute_sum_start_end", arr.dtype.name)
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
    func = _rs_func("compute_prod_start_end", arr.dtype.name)
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
    func = _rs_func("compute_prod_start_end_match", arr.dtype.name)
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )


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
    func = _rs_func("compute_sum_start_end_match", arr.dtype.name)
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
    func = _rs_func("compute_min_start_end_match", arr.dtype.name)
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
    func = _rs_func("compute_max_start_end_match", arr.dtype.name)
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
    func = _rs_func("compute_prod_rev_start", arr.dtype.name)
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
    func = _rs_func("compute_prod_rev_end", arr.dtype.name)
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
    func = _rs_func("compute_prod_rev_start_match", arr.dtype.name)
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
    func = _rs_func("compute_prod_rev_end_match", arr.dtype.name)
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
    func = _rs_func("compute_prod_rev_positions", arr.dtype.name)
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
    func = _rs_func("compute_prod_rev_start_end", arr.dtype.name)
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
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
    func = _rs_func("compute_prod_rev_start_end_match", arr.dtype.name)
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
    func = _rs_func("compute_min_rev_start", arr.dtype.name)
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
    func = _rs_func("compute_min_rev_end", arr.dtype.name)
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
    func = _rs_func("compute_min_rev_start_match", arr.dtype.name)
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
    func = _rs_func("compute_min_rev_end_match", arr.dtype.name)
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
    func = _rs_func("compute_min_rev_positions", arr.dtype.name)
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
    func = _rs_func("compute_min_rev_start_end", arr.dtype.name)
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
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
    func = _rs_func("compute_min_rev_start_end_match", arr.dtype.name)
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
    func = _rs_func("compute_max_rev_start", arr.dtype.name)
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
    func = _rs_func("compute_max_rev_end", arr.dtype.name)
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
    func = _rs_func("compute_max_rev_start_match", arr.dtype.name)
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
    func = _rs_func("compute_max_rev_end_match", arr.dtype.name)
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
    func = _rs_func("compute_max_rev_positions", arr.dtype.name)
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
    func = _rs_func("compute_max_rev_start_end", arr.dtype.name)
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
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
    func = _rs_func("compute_max_rev_start_end_match", arr.dtype.name)
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
    func = _rs_func("compute_prod_rev_no_range", arr.dtype.name)
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
    func = _rs_func("compute_max_rev_no_range", arr.dtype.name)
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
    func = _rs_func("compute_min_rev_no_range", arr.dtype.name)
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
    func = _rs_func("compute_sum_rev_no_range", arr.dtype.name)
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
    func = _rs_func("compute_sum_rev_start", arr.dtype.name)
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
    func = _rs_func("compute_sum_rev_end", arr.dtype.name)
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
    func = _rs_func("compute_sum_rev_start_match", arr.dtype.name)
    return func(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
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
    func = _rs_func("compute_sum_rev_end_match", arr.dtype.name)
    return func(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
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
    func = _rs_func("compute_sum_rev_positions", arr.dtype.name)
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
    func = _rs_func("compute_sum_rev_start_end", arr.dtype.name)
    return func(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        booleans=booleans,
        length=length,
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
    func = _rs_func("compute_sum_rev_start_end_match", arr.dtype.name)
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
