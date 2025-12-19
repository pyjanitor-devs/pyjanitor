"""Various Functions powered by Numba"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import literal_unroll, njit, prange, types
from numba.extending import overload
from pandas.api.types import (
    is_datetime64_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

# https://numba.discourse.group/t/uint64-vs-int64-indexing-performance-difference/1500
# indexing with unsigned integers offers more performance


@njit(nogil=True)
def _numba_less_than_base(arr: np.ndarray, value: Any):
    """
    Get earliest position in `arr`
    where arr[i] <= `value`
    """
    # adapted from numba/np/array_math.py
    min_idx = 0
    max_idx = len(arr)
    while min_idx < max_idx:
        # to avoid overflow
        mid_idx = min_idx + ((max_idx - min_idx) >> 1)
        _mid_idx = np.uintp(mid_idx)
        if arr[_mid_idx] < value:
            min_idx = mid_idx + 1
        else:
            max_idx = mid_idx
    return min_idx


@njit(nogil=True)
def _numba_greater_than_base(arr: np.ndarray, value: Any):
    """
    Get latest position in `arr`
    where arr[i] >= `value`
    """
    # adapted from numba/np/array_math.py
    min_idx = 0
    max_idx = len(arr)
    while min_idx < max_idx:
        # to avoid overflow
        mid_idx = min_idx + ((max_idx - min_idx) >> 1)
        _mid_idx = np.uintp(mid_idx)
        if arr[_mid_idx] > value:
            max_idx = mid_idx
        else:
            min_idx = mid_idx + 1
    return min_idx


@njit(nogil=True)
def _numba_less_than(
    arr: np.ndarray,
    value: Any,
    strict: bool = False,
):
    """
    Get earliest position in `arr`
    where arr[i] </<= `value`
    """
    min_idx = _numba_less_than_base(arr=arr, value=value)
    if min_idx == arr.size:
        return -1
    if strict & (value == arr[min_idx]):
        min_idx = _numba_greater_than_base(arr=arr, value=value)
    if min_idx == arr.size:
        return -1
    return min_idx


@njit(nogil=True)
def _numba_greater_than(arr: np.ndarray, value: Any, strict: bool = False):
    """
    Get latest position in `arr`
    where arr[i] >/>= `value`
    """
    min_idx = _numba_greater_than_base(arr=arr, value=value)
    if not min_idx:
        return -1
    if strict & (value == arr[min_idx - 1]):
        min_idx = _numba_less_than_base(arr=arr, value=value)
    if not min_idx:
        return -1
    return min_idx


@njit(nogil=True)
def _le_lt_join_not_range_join(
    source_array: np.ndarray,
    search_array: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    strict: bool,
    booleans: np.ndarray,
):
    """
    Compute results for le/lt join, not in a range join
    """
    counts = 0
    new_slice_starts = np.empty(slice_starts.size, dtype=np.intp)
    for num in prange(booleans.size):
        l1 = source_array[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = search_array[slice_start:slice_end]
        position = _numba_less_than(arr=r1, value=l1, strict=strict)
        if position == -1:
            counts += 1
            new_slice_starts[num] = slice_end
            booleans[num] = False
        else:
            new_slice_starts[num] = slice_start + position
    return booleans, new_slice_starts, counts


@njit(nogil=True)
def _le_lt_join_range_join(
    source_array: np.ndarray,
    search_array: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    strict: bool,
    booleans: np.ndarray,
):
    """
    Compute results for le/lt join, in a range join
    """
    counts = 0
    new_slice_starts = np.empty(slice_starts.size, dtype=np.intp)
    for num in prange(booleans.size):
        l1 = source_array[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        if not booleans[num]:
            continue
        r1 = search_array[slice_start:slice_end]
        position = _numba_less_than(arr=r1, value=l1, strict=strict)
        if position == -1:
            counts += 1
            new_slice_starts[num] = slice_end
            booleans[num] = False
        else:
            new_slice_starts[num] = slice_start + position
    return booleans, new_slice_starts, counts


@njit(nogil=True)
def _ge_or_gt_equi_join(
    source_array: np.ndarray,
    search_array: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    strict: bool,
    booleans: np.ndarray,
):
    """Compute results for a greater than or equal join"""
    counts = 0
    new_slice_ends = np.empty(slice_ends.size, dtype=np.intp)
    for num in prange(booleans.size):
        l1 = source_array[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = search_array[slice_start:slice_end]
        position = _numba_greater_than(arr=r1, value=l1, strict=strict)
        if position == -1:
            counts += 1
            new_slice_ends[num] = slice_start
            booleans[num] = False
        else:
            new_slice_ends[num] = slice_start + position
    return booleans, new_slice_ends, counts


@njit(nogil=True)
def _generate_final_indices_monotonic_equi_join(
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    booleans: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
):
    """
    Create final indices if monotonic
    """
    begin = 0
    total = (slice_ends - slice_starts).sum()
    r_index = np.empty(total, dtype=np.intp)
    l_index = np.empty(total, dtype=np.intp)
    for _num in range(slice_ends.size):
        num = np.uintp(_num)
        if not booleans[num]:
            continue
        start = slice_starts[num]
        end = slice_ends[num]
        for n in range(start, end):
            n_ = np.uintp(n)
            r_index[begin] = right_index[n_]
            l_index[begin] = left_index[num]
            begin += 1
    return l_index, r_index


@njit(nogil=True)
def _eq_join_tuple_compare(
    booleans: np.ndarray,
    indices: np.ndarray,
    left_index: np.ndarray,
    start_indices: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    tupled: tuple,
):
    """Comparison when tupled is present in an equi join"""

    for _tuple in literal_unroll(tupled):
        left_arr = _tuple[0]
        right_arr = _tuple[1]
        op = _tuple[2]
        for n in range(left_index.size):
            _n = np.uintp(n)
            if not booleans[_n]:
                continue
            start = starts[_n]
            end = ends[_n]
            pos = 0
            counter = 0
            left_val = left_arr[_n]
            ind = start_indices[_n]
            for nn in range(start, end):
                _nn = np.uintp(nn)
                right_val = right_arr[_nn]
                boolean = _compare(left_val, right_val, op)
                _ind = np.uintp(ind + pos)
                # pos should always increment
                # no matter what happens
                # with the conditionals below
                pos += 1
                boolean = indices[_ind] & boolean
                indices[_ind] = boolean
                counter += np.intp(boolean)
            boolean_ = counter > 0
            booleans[_n] &= boolean_

    return booleans, indices


@njit(nogil=True)
def _get_row_count_tupled_equi_join(
    left_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    start_indices: np.ndarray,
    indices: np.ndarray,
    booleans: np.ndarray,
):
    """Get row count for an equi join with a tupled"""
    counts = np.zeros(left_index.size, dtype=np.intp)
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not booleans[_n]:
            continue
        start = starts[_n]
        end = ends[_n]
        pos = 0
        counter = 0
        ind = start_indices[_n]
        for _ in range(start, end):
            _ind = np.uintp(ind + pos)
            # pos should always increment
            # no matter what happens
            # with the condition below
            pos += 1
            if not indices[_ind]:
                continue
            counter += 1
        counts[_n] = counter
    return left_index, counts


@njit(nogil=True)
def _generate_indices_equi_join_tupled(
    indices: np.ndarray,
    left_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    start_indices: np.ndarray,
    right_index: np.ndarray,
    booleans: np.ndarray,
):
    """Build indices for equi join with tupled"""
    total = indices.sum()
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    indexer = 0
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not booleans[_n]:
            continue
        start = starts[_n]
        end = ends[_n]
        pos = 0
        counter = 0
        ind = start_indices[_n]
        l_index = left_index[_n]
        for nn in range(start, end):
            _ind = np.uintp(ind + pos)
            # pos should always increment
            # no matter what happens
            # with the condition below
            pos += 1
            if not indices[_ind]:
                continue
            _nn = np.uintp(nn)
            _indexer = np.uintp(indexer)
            left_indices[_indexer] = l_index
            right_indices[_indexer] = right_index[_nn]
            indexer += 1
            if indexer == total:
                counter = 1
                break
        if counter == 1:
            break
    return left_indices, right_indices


@njit(parallel=True, cache=True)
def _numba_equi_single_le_ge_join(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    arr1: np.ndarray,
    arr2: np.ndarray,
    strict: bool,
    less_than: bool,
    row_count: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a less than/greater than join
    """
    length = left_index.size
    booleans = np.ones(length, dtype=np.bool_)
    if less_than:
        booleans, slice_starts, counts = _le_lt_join_not_range_join(
            source_array=arr1,
            search_array=arr2,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            strict=strict,
            booleans=booleans,
        )
    else:
        booleans, slice_ends, counts = _ge_or_gt_equi_join(
            source_array=arr1,
            search_array=arr2,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            strict=strict,
            booleans=booleans,
        )
    if counts == length:
        return None, None
    if row_count:
        return left_index, slice_ends - slice_starts
    return _generate_final_indices_monotonic_equi_join(
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        booleans=booleans,
        left_index=left_index,
        right_index=right_index,
    )


@njit(parallel=True, cache=True)
def _numba_equi_single_le_ge_tupled_join(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    arr1: np.ndarray,
    arr2: np.ndarray,
    strict: bool,
    less_than: bool,
    row_count: bool,
    tupled: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a less than/greater than join,
    and a tupled
    """
    length = left_index.size
    booleans = np.ones(length, dtype=np.bool_)
    if less_than:
        booleans, slice_starts, counts = _le_lt_join_not_range_join(
            source_array=arr1,
            search_array=arr2,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            strict=strict,
            booleans=booleans,
        )
    else:
        booleans, slice_ends, counts = _ge_or_gt_equi_join(
            source_array=arr1,
            search_array=arr2,
            slice_starts=slice_starts,
            slice_ends=slice_ends,
            strict=strict,
            booleans=booleans,
        )

    if counts == length:
        return None, None

    l_booleans = booleans[:]
    _indices = np.cumsum(slice_ends - slice_starts)
    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    start_indices[1:] = _indices[:-1]
    _indices = _indices[-1]
    indices = np.ones(_indices, dtype=np.bool_)
    starts_slice = slice_starts[:]
    ends_slice = slice_ends[:]

    l_booleans, indices = _eq_join_tuple_compare(
        booleans=l_booleans,
        indices=indices,
        left_index=left_index,
        start_indices=start_indices,
        starts=starts_slice,
        ends=ends_slice,
        tupled=tupled,
    )

    if not np.any(l_booleans):
        return None, None
    if row_count:
        return _get_row_count_tupled_equi_join(
            left_index=left_index,
            starts=starts_slice,
            ends=ends_slice,
            start_indices=start_indices,
            indices=indices,
            booleans=l_booleans,
        )
    return _generate_indices_equi_join_tupled(
        indices=indices,
        left_index=left_index,
        starts=starts_slice,
        ends=ends_slice,
        start_indices=start_indices,
        right_index=right_index,
        booleans=l_booleans,
    )


@njit(parallel=True, cache=True)
def _numba_equi_join_range_join_monotonic(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    ge_arr1: np.ndarray,
    ge_arr2: np.ndarray,
    ge_strict: bool,
    le_arr1: np.ndarray,
    le_arr2: np.ndarray,
    le_strict: bool,
    row_count: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a range join
    """
    length = left_index.size
    booleans = np.ones(length, dtype=np.bool_)
    booleans, slice_ends, counts = _ge_or_gt_equi_join(
        source_array=ge_arr1,
        search_array=ge_arr2,
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        strict=ge_strict,
        booleans=booleans,
    )
    if counts == length:
        return None, None

    booleans, slice_starts, counts = _le_lt_join_not_range_join(
        source_array=le_arr1,
        search_array=le_arr2,
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        strict=le_strict,
        booleans=booleans,
    )
    if counts == length:
        return None, None

    if row_count:
        return left_index, slice_ends - slice_starts
    # no need to run a comparison
    # since all groups are monotonic increasing
    # simply create left and right indices
    return _generate_final_indices_monotonic_equi_join(
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        booleans=booleans,
        left_index=left_index,
        right_index=right_index,
    )


@njit(parallel=True, cache=True)
def _numba_equi_join_range_join_multiple_monotonic(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    ge_arr1: np.ndarray,
    ge_arr2: np.ndarray,
    ge_strict: bool,
    le_arr1: np.ndarray,
    le_arr2: np.ndarray,
    le_strict: bool,
    row_count: bool,
    tupled: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a range join
    """
    length = left_index.size
    booleans = np.ones(length, dtype=np.bool_)
    booleans, slice_ends, counts = _ge_or_gt_equi_join(
        source_array=ge_arr1,
        search_array=ge_arr2,
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        strict=ge_strict,
        booleans=booleans,
    )
    if counts == length:
        return None, None

    booleans, slice_starts, counts = _le_lt_join_not_range_join(
        source_array=le_arr1,
        search_array=le_arr2,
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        strict=le_strict,
        booleans=booleans,
    )
    if counts == length:
        return None, None

    _indices = np.cumsum(slice_ends - slice_starts)
    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    start_indices[1:] = _indices[:-1]
    _indices = _indices[-1]
    indices = np.ones(_indices, dtype=np.bool_)
    starts_slice = slice_starts[:]
    ends_slice = slice_ends[:]
    l_booleans = booleans[:]

    l_booleans, indices = _eq_join_tuple_compare(
        booleans=l_booleans,
        indices=indices,
        left_index=left_index,
        start_indices=start_indices,
        starts=starts_slice,
        ends=ends_slice,
        tupled=tupled,
    )

    if not np.any(l_booleans):
        return None, None

    if row_count:
        return _get_row_count_tupled_equi_join(
            left_index=left_index,
            starts=starts_slice,
            ends=ends_slice,
            start_indices=start_indices,
            indices=indices,
            booleans=l_booleans,
        )
    return _generate_indices_equi_join_tupled(
        indices=indices,
        left_index=left_index,
        starts=starts_slice,
        ends=ends_slice,
        start_indices=start_indices,
        right_index=right_index,
        booleans=l_booleans,
    )


@njit(parallel=True, cache=True)
def _numba_equi_join_range_join_non_monotonic(
    left_index: np.ndarray,
    right_index: np.ndarray,
    slice_starts: np.ndarray,
    slice_ends: np.ndarray,
    ge_arr1: np.ndarray,
    ge_arr2: np.ndarray,
    ge_strict: bool,
    row_count: bool,
    tupled: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices for an equi join
    and a range join
    """
    length = left_index.size
    booleans = np.ones(length, dtype=np.bool_)
    booleans, slice_ends, counts = _ge_or_gt_equi_join(
        source_array=ge_arr1,
        search_array=ge_arr2,
        slice_starts=slice_starts,
        slice_ends=slice_ends,
        strict=ge_strict,
        booleans=booleans,
    )
    if counts == length:
        return None, None

    _indices = np.cumsum(slice_ends - slice_starts)
    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    start_indices[1:] = _indices[:-1]
    _indices = _indices[-1]
    indices = np.ones(_indices, dtype=np.bool_)
    starts_slice = slice_starts[:]
    ends_slice = slice_ends[:]
    l_booleans = booleans[:]

    l_booleans, indices = _eq_join_tuple_compare(
        booleans=l_booleans,
        indices=indices,
        left_index=left_index,
        start_indices=start_indices,
        starts=starts_slice,
        ends=ends_slice,
        tupled=tupled,
    )

    if not np.any(l_booleans):
        return None, None

    if row_count:
        return _get_row_count_tupled_equi_join(
            left_index=left_index,
            starts=starts_slice,
            ends=ends_slice,
            start_indices=start_indices,
            indices=indices,
            booleans=l_booleans,
        )
    return _generate_indices_equi_join_tupled(
        indices=indices,
        left_index=left_index,
        starts=starts_slice,
        ends=ends_slice,
        start_indices=start_indices,
        right_index=right_index,
        booleans=l_booleans,
    )


@njit(nogil=True)
def _not_montonic_multi_unroll(
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    indices: np.ndarray,
    l_booleans: np.ndarray,
    left_index: np.ndarray,
    counts: np.ndarray,
    tupled: tuple,
):
    """
    Get matching locations for non equi join
    """

    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    start_indices[1:] = counts.cumsum()[:-1]
    for _tuple in literal_unroll(tupled):
        left_arr = _tuple[0]
        right_arr = _tuple[1]
        op = _tuple[2]
        for n in range(left_index.size):
            _n = np.uintp(n)
            if (not l_booleans[_n]) | (not counts[_n]):
                l_booleans[_n] = False
                continue
            counter = 0
            nn = left_indices[_n]
            _nn = np.uintp(nn)
            size = counts[_n]
            start_index = start_indices[_n]
            left_val = left_arr[_nn]
            for ind in range(start_index, start_index + size):
                _ind = np.uintp(ind)
                nnn = right_indices[_ind]
                _nnn = np.uintp(nnn)
                right_val = right_arr[_nnn]
                boolean = _compare(left_val, right_val, op)
                boolean = indices[_ind] & boolean
                indices[_ind] = boolean
                counter += np.intp(boolean)
            boolean_ = counter > 0
            l_booleans[_n] &= boolean_
    return l_booleans, start_indices, indices


@njit(cache=True, parallel=False, nogil=True)
def _numba_non_equi_join_not_monotonic_keep_all(
    tupled: tuple,
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    load_factor: int,
    starts: np.ndarray,
    row_count: bool,
) -> tuple:
    """
    Get indices if there are more than two join conditions
    """
    left_indices, right_indices, counts = (
        _numba_non_equi_join_not_monotonic_keep_all_indices(
            left_regions=left_regions,
            right_regions=right_regions,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    )
    if left_indices is None:
        return None, None
    indices = np.ones(right_indices.size, dtype=np.bool_)
    l_booleans = np.ones(left_index.size, dtype=np.bool_)
    l_booleans, start_indices, indices = _not_montonic_multi_unroll(
        left_indices=left_indices,
        right_indices=right_indices,
        left_index=left_index,
        counts=counts,
        tupled=tupled,
        indices=indices,
        l_booleans=l_booleans,
    )
    if not np.any(l_booleans):
        return None, None
    if row_count:
        total = l_booleans.sum()
        r_counts = np.zeros(total, dtype=np.intp)
        left_indexes = np.empty(total, dtype=np.intp)
        indexer = 0
        for n in range(l_booleans.size):
            _n = np.uintp(n)
            if not l_booleans[_n]:
                continue
            size = counts[_n]
            start_index = start_indices[_n]
            counter = 0
            for ind in range(start_index, start_index + size):
                _ind = np.uintp(ind)
                if not indices[_ind]:
                    continue
                counter += 1
            nn = left_indices[_n]
            _nn = np.uintp(nn)
            left_indexes[indexer] = left_index[_nn]
            r_counts[indexer] = counter
            indexer += 1
        return left_indexes, r_counts
    total = indices.sum()
    left_indexes = np.empty(total, dtype=np.intp)
    right_indexes = np.empty(total, dtype=np.intp)
    indexer = 0
    counter = 0
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not l_booleans[_n]:
            continue
        nn = left_indices[_n]
        _nn = np.uintp(nn)
        size = counts[_n]
        start_index = start_indices[_n]
        left_val = left_index[_nn]
        for ind in range(start_index, start_index + size):
            _ind = np.uintp(ind)
            boolean = indices[_ind]
            if not boolean:
                continue
            nnn = right_indices[_ind]
            _nnn = np.uintp(nnn)
            right_val = right_index[_nnn]
            _indexer = np.uintp(indexer)
            left_indexes[_indexer] = left_val
            right_indexes[_indexer] = right_val
            indexer += 1
            if indexer == total:
                counter = 1
                break
        if counter == 1:
            break
    return left_indexes, right_indexes


@njit(cache=True, parallel=False, nogil=True)
def _numba_non_equi_join_not_monotonic_keep_first(
    tupled: bool,
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    load_factor: int,
    starts: np.ndarray,
) -> tuple:
    """
    Get indices if there are more than two join conditions
    """
    left_indices, right_indices, counts = (
        _numba_non_equi_join_not_monotonic_keep_all_indices(
            left_regions=left_regions,
            right_regions=right_regions,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    )
    if left_indices is None:
        return None, None
    indices = np.ones(right_indices.size, dtype=np.bool_)
    l_booleans = np.ones(left_index.size, dtype=np.bool_)
    l_booleans, start_indices, indices = _not_montonic_multi_unroll(
        left_indices=left_indices,
        right_indices=right_indices,
        left_index=left_index,
        counts=counts,
        tupled=tupled,
        indices=indices,
        l_booleans=l_booleans,
    )
    total = l_booleans.sum()
    if not total:
        return None, None
    left_indexes = np.empty(total, dtype=np.intp)
    right_indexes = np.empty(total, dtype=np.intp)
    indexer = 0
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not l_booleans[_n]:
            continue
        nn = left_indices[_n]
        _nn = np.uintp(nn)
        size = counts[_n]
        start_index = start_indices[_n]
        left_val = left_index[_nn]
        base = -1
        for ind in range(start_index, start_index + size):
            _ind = np.uintp(ind)
            boolean = indices[_ind]
            if not boolean:
                continue
            nnn = right_indices[_ind]
            _nnn = np.uintp(nnn)
            right_val = right_index[_nnn]
            if (base == -1) | (right_val < base):
                base = right_val
        _indexer = np.uintp(indexer)
        left_indexes[_indexer] = left_val
        right_indexes[_indexer] = base
        indexer += 1
    return left_indexes, right_indexes


@njit(cache=True, parallel=False, nogil=True)
def _numba_non_equi_join_not_monotonic_keep_last(
    tupled: tuple,
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    load_factor: int,
    starts: np.ndarray,
) -> tuple:
    """
    Get indices if there are more than two join conditions
    """
    left_indices, right_indices, counts = (
        _numba_non_equi_join_not_monotonic_keep_all_indices(
            left_regions=left_regions,
            right_regions=right_regions,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    )
    if left_indices is None:
        return None, None
    indices = np.ones(right_indices.size, dtype=np.bool_)
    l_booleans = np.ones(left_index.size, dtype=np.bool_)
    l_booleans, start_indices, indices = _not_montonic_multi_unroll(
        left_indices=left_indices,
        right_indices=right_indices,
        left_index=left_index,
        counts=counts,
        tupled=tupled,
        indices=indices,
        l_booleans=l_booleans,
    )

    total = l_booleans.sum()
    if not total:
        return None, None
    left_indexes = np.empty(total, dtype=np.intp)
    right_indexes = np.empty(total, dtype=np.intp)
    indexer = 0
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not l_booleans[_n]:
            continue
        nn = left_indices[_n]
        _nn = np.uintp(nn)
        size = counts[_n]
        start_index = start_indices[_n]
        left_val = left_index[_nn]
        base = np.inf
        for ind in range(start_index, start_index + size):
            _ind = np.uintp(ind)
            boolean = indices[_ind]
            if not boolean:
                continue
            nnn = right_indices[_ind]
            _nnn = np.uintp(nnn)
            right_val = right_index[_nnn]
            if (base == np.inf) | (right_val > base):
                base = right_val
        _indexer = np.uintp(indexer)
        left_indexes[_indexer] = left_val
        right_indexes[_indexer] = base
        indexer += 1
    return left_indexes, right_indexes


@njit(inline="always")
def compare_values(left_val, right_val, op):
    if op == 0:
        return left_val > right_val
    if op == 1:
        return left_val >= right_val
    if op == 2:
        return left_val < right_val
    if op == 3:
        return left_val <= right_val
    if op == 4:
        return left_val != right_val
    return left_val == right_val


def _compare(x, y, op):
    if (
        (is_numeric_dtype(x) and is_numeric_dtype(y))
        or (is_datetime64_dtype(x) and is_datetime64_dtype(y))
        or (is_timedelta64_dtype(x) and is_timedelta64_dtype(y))
    ):
        return compare_values(x, y, op)


accepted_types = (
    types.NPDatetime,
    types.Integer,
    types.Float,
    types.NPTimedelta,
)


@overload(_compare)
def _numba_compare(x, y, op):
    if (
        isinstance(x, accepted_types)
        and isinstance(y, accepted_types)
        and isinstance(op, types.Integer)
    ):

        def impl(x, y, op):
            return compare_values(x, y, op)

        return impl
    else:
        raise TypeError("Unsupported Type")


@njit(cache=True, parallel=True)
def _range_join_sorted_dual_keep_all(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    start_indices: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
):
    """
    Get indices for a dual non equi join
    """
    for ind in prange(left_index.size):
        _ind = np.uintp(ind)
        start = starts[_ind]
        end = ends[_ind]
        indexer = start_indices[_ind]
        lindex = left_index[_ind]
        for num in range(start, end):
            _num = np.uintp(num)
            rindex = right_index[_num]
            _indexer = np.uintp(indexer)
            left_indices[_indexer] = lindex
            right_indices[_indexer] = rindex
            indexer += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_range_join_sorted_keep_first_or_last_dual(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    position: bool,
):
    """
    Get indices for a non equi join
    """
    for ind in prange(left_index.size):
        _ind = np.uintp(ind)
        start = starts[_ind]
        end = ends[_ind]
        lindex = left_index[_ind]
        base_index = right_index[np.uintp(start)]
        for num in range(start, end):
            _num = np.uintp(num)
            rindex = right_index[_num]
            keep_first = rindex < base_index
            keep_first = position & keep_first
            keep_last = rindex > base_index
            keep_last = (not position) & keep_last
            boolean = keep_first | keep_last
            if boolean:
                base_index = rindex
        left_indices[_ind] = lindex
        right_indices[_ind] = base_index
    return left_indices, right_indices


@njit(nogil=True)
def _monotonic_increasing_unroll(
    left_index: np.ndarray,
    right_index: np.ndarray,
    tupled: tuple,
    starts: np.ndarray,
    start_indices: np.ndarray,
    indices: np.ndarray,
):
    """
    Compute matching locations for multiple conditions
    """
    l_booleans = np.ones(left_index.size, dtype=np.bool_)
    end = right_index.size
    for _tuple in literal_unroll(tupled):
        left_arr = _tuple[0]
        right_arr = _tuple[1]
        op = _tuple[2]
        for n in range(left_index.size):
            _n = np.uintp(n)
            if not l_booleans[_n]:
                continue
            start = starts[_n]
            pos = 0
            counter = 0
            left_val = left_arr[_n]
            ind = start_indices[_n]
            for nn in range(start, end):
                _nn = np.uintp(nn)
                right_val = right_arr[_nn]
                boolean = _compare(left_val, right_val, op)
                _ind = np.uintp(ind + pos)
                # pos should always increment
                # no matter what happens
                # with the conditionals below
                pos += 1
                boolean = indices[_ind] & boolean
                indices[_ind] = boolean
                counter += np.intp(boolean)
            boolean_ = counter > 0
            l_booleans[_n] &= boolean_
    return l_booleans, indices


@njit(cache=True, parallel=False, nogil=True)
def _numba_non_equi_join_monotonic_increasing_keep_all(
    tupled: tuple,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    indices: np.ndarray,
    start_indices: np.ndarray,
    row_count: bool,
) -> tuple:
    """
    Get indices if there are more than two join conditions,
    and a range join, sorted on both right columns, exists.
    """
    l_booleans, indices = _monotonic_increasing_unroll(
        left_index=left_index,
        right_index=right_index,
        tupled=tupled,
        starts=starts,
        start_indices=start_indices,
        indices=indices,
    )
    if not np.any(l_booleans):
        return None, None
    if row_count:
        counts = np.zeros(left_index.size, dtype=np.intp)
        end = right_index.size
        for n in range(left_index.size):
            _n = np.uintp(n)
            if not l_booleans[_n]:
                continue
            start = starts[_n]
            pos = 0
            counter = 0
            ind = start_indices[_n]
            l_index = left_index[_n]
            for nn in range(start, end):
                _ind = np.uintp(ind + pos)
                # pos should always increment
                # no matter what happens
                # with the condition below
                pos += 1
                if not indices[_ind]:
                    continue
                counter += 1
            counts[_n] = counter
        return left_index, counts
    total = indices.sum()
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    indexer = 0
    end = right_index.size
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not l_booleans[_n]:
            continue
        start = starts[_n]
        pos = 0
        counter = 0
        ind = start_indices[_n]
        l_index = left_index[_n]
        for nn in range(start, end):
            _ind = np.uintp(ind + pos)
            # pos should always increment
            # no matter what happens
            # with the condition below
            pos += 1
            if not indices[_ind]:
                continue
            _nn = np.uintp(nn)
            _indexer = np.uintp(indexer)
            left_indices[_indexer] = l_index
            right_indices[_indexer] = right_index[_nn]
            indexer += 1
            if indexer == total:
                counter = 1
                break
        if counter == 1:
            break
    return left_indices, right_indices


@njit(cache=True, parallel=False, nogil=True)
def _numba_non_equi_join_monotonic_increasing_keep_first_or_last(
    tupled: tuple,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    indices: np.ndarray,
    start_indices: np.ndarray,
    position: bool,
) -> tuple:
    """
    Get earliest/latest indices if there are more than two join conditions,
    and a range join, sorted on both right columns, exists.
    """
    l_booleans, indices = _monotonic_increasing_unroll(
        left_index=left_index,
        right_index=right_index,
        tupled=tupled,
        starts=starts,
        start_indices=start_indices,
        indices=indices,
    )

    total = l_booleans.sum()
    if not total:
        return None, None
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    indexer = 0
    end = right_index.size
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not l_booleans[_n]:
            continue
        start = starts[_n]
        pos = 0
        ind = start_indices[_n]
        l_index = left_index[_n]
        base = -1
        for nn in range(start, end):
            _ind = np.uintp(ind + pos)
            # pos should always increment
            # no matter what happens
            # with the condition below
            pos += 1
            if not indices[_ind]:
                continue
            _nn = np.uintp(nn)
            value = right_index[_nn]
            keep_first = value < base
            keep_first = position & keep_first
            keep_last = value > base
            keep_last = (not position) & keep_last
            boolean = keep_first | keep_last
            boolean = (base == -1) | boolean
            if boolean:
                base = value
        _indexer = np.uintp(indexer)
        left_indices[_indexer] = l_index
        right_indices[_indexer] = base
        indexer += 1
    return left_indices, right_indices


@njit(nogil=True)
def _range_join_sorted_unroll(
    tupled: tuple,
    left_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    start_indices: np.ndarray,
    indices: np.ndarray,
):
    """
    Get matching locations for multiple conditions for a sorted range join
    """
    l_booleans = np.ones(left_index.size, dtype=np.bool_)
    for _tuple in literal_unroll(tupled):
        left_arr = _tuple[0]
        right_arr = _tuple[1]
        op = _tuple[2]
        for n in range(left_index.size):
            _n = np.uintp(n)
            if not l_booleans[_n]:
                continue
            start = starts[_n]
            end = ends[_n]
            pos = 0
            counter = 0
            ind = start_indices[_n]
            for nn in range(start, end):
                _nn = np.uintp(nn)
                left_val = left_arr[_n]
                right_val = right_arr[_nn]
                boolean = _compare(left_val, right_val, op)
                _ind = np.uintp(ind + pos)
                # pos should always increment
                # no matter what happens
                # with the conditionals below
                pos += 1
                boolean = indices[_ind] & boolean
                indices[_ind] = boolean
                counter += np.intp(boolean)
            boolean_ = counter > 0
            l_booleans[_n] &= boolean_
    return l_booleans, indices


@njit(cache=True, parallel=False, nogil=True)
def _range_join_sorted_multiple_keep_all(
    tupled: tuple,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    indices: np.ndarray,
    start_indices: np.ndarray,
    row_count: bool,
) -> tuple:
    """
    Get indices if there are more than two join conditions,
    and a range join, sorted on both right columns, exists.
    """
    l_booleans, indices = _range_join_sorted_unroll(
        tupled=tupled,
        left_index=left_index,
        starts=starts,
        ends=ends,
        start_indices=start_indices,
        indices=indices,
    )
    if not np.any(l_booleans):
        return None, None
    if row_count:
        counts = np.zeros(left_index.size, dtype=np.intp)
        for n in range(left_index.size):
            _n = np.uintp(n)
            if not l_booleans[_n]:
                continue
            start = starts[_n]
            end = ends[_n]
            pos = 0
            counter = 0
            ind = start_indices[_n]
            for _ in range(start, end):
                _ind = np.uintp(ind + pos)
                # pos should always increment
                # no matter what happens
                # with the condition below
                pos += 1
                if not indices[_ind]:
                    continue
                counter += 1
            counts[_n] = counter
        return left_index, counts
    total = indices.sum()
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    indexer = 0
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not l_booleans[_n]:
            continue
        start = starts[_n]
        end = ends[_n]
        pos = 0
        counter = 0
        ind = start_indices[_n]
        l_index = left_index[_n]
        for nn in range(start, end):
            _ind = np.uintp(ind + pos)
            # pos should always increment
            # no matter what happens
            # with the condition below
            pos += 1
            if not indices[_ind]:
                continue
            _nn = np.uintp(nn)
            _indexer = np.uintp(indexer)
            left_indices[_indexer] = l_index
            right_indices[_indexer] = right_index[_nn]
            indexer += 1
            if indexer == total:
                counter = 1
                break
        if counter == 1:
            break
    return left_indices, right_indices


@njit(cache=True, parallel=False, nogil=True)
def _range_join_sorted_multiple_keep_first_or_last(
    tupled: tuple,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    indices: np.ndarray,
    start_indices: np.ndarray,
    position: bool,
) -> tuple:
    """
    Get earliest/latest indices if there are more than two join conditions,
    and a range join, sorted on both right columns, exists.
    """
    l_booleans, indices = _range_join_sorted_unroll(
        tupled=tupled,
        left_index=left_index,
        starts=starts,
        ends=ends,
        start_indices=start_indices,
        indices=indices,
    )
    total = l_booleans.sum()
    if not total:
        return None, None
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    indexer = 0
    for n in range(left_index.size):
        _n = np.uintp(n)
        if not l_booleans[_n]:
            continue
        start = starts[_n]
        end = ends[_n]
        pos = 0
        ind = start_indices[_n]
        l_index = left_index[_n]
        base = -1
        for nn in range(start, end):
            _ind = np.uintp(ind + pos)
            # pos should always increment
            # no matter what happens
            # with the condition below
            pos += 1
            if not indices[_ind]:
                continue
            _nn = np.uintp(nn)
            value = right_index[_nn]
            keep_first = value < base
            keep_first = position & keep_first
            keep_last = value > base
            keep_last = (not position) & keep_last
            boolean = keep_first | keep_last
            boolean = (base == -1) | boolean
            if boolean:
                base = value
        _indexer = np.uintp(indexer)
        left_indices[_indexer] = l_index
        right_indices[_indexer] = base
        indexer += 1
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_increasing_keep_first_or_last_dual(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    position: bool,
):
    """
    Get indices for a non equi join
    """
    end = right_index.size
    for ind in prange(left_index.size):
        _ind = np.uintp(ind)
        start = starts[_ind]
        lindex = left_index[_ind]
        base_index = right_index[np.uintp(start)]
        for num in range(start, end):
            _num = np.uintp(num)
            rindex = right_index[_num]
            keep_first = rindex < base_index
            keep_first = position & keep_first
            keep_last = rindex > base_index
            keep_last = (not position) & keep_last
            boolean = keep_first | keep_last
            if boolean:
                base_index = rindex
        left_indices[_ind] = lindex
        right_indices[_ind] = base_index
    return left_indices, right_indices


@njit(cache=True, parallel=True)
def _numba_non_equi_join_monotonic_increasing_keep_all_dual(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    start_indices: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
):
    """
    Get indices for a non equi join
    """
    end = right_index.size
    for ind in prange(left_index.size):
        _ind = np.uintp(ind)
        start = starts[_ind]
        indexer = start_indices[_ind]
        lindex = left_index[_ind]
        for num in range(start, end):
            _num = np.uintp(num)
            rindex = right_index[_num]
            _indexer = np.uintp(indexer)
            left_indices[_indexer] = lindex
            right_indices[_indexer] = rindex
            indexer += 1
    return left_indices, right_indices


@njit(nogil=True)
def _numba_non_equi_join_not_monotonic_keep_all_indices(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
):
    """
    Get indices for non-equi join,
    where the right regions are not monotonic
    """
    # first pass - get actual length
    length = left_regions.size
    end = right_regions.size
    end -= 1
    # add the last region
    # no need to have this checked within an if-else statement
    # in the for loop below
    region = right_regions[np.uintp(end)]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    # keep track of the maxxes array
    # how many cells have actual values?
    maxxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uintp(indexer)
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_regions[_num]
            arr = maxxes[:maxxes_counter]
            if region > arr[-1]:
                # it is larger than the max in the maxxes array
                # shove it into the last column
                posn = maxxes_counter - 1
                posn_ = np.uintp(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                # we dont need to compute positions in the first run?
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn = _numba_less_than_base(arr=arr, value=region)
                sorted_array, positions_array, lengths, maxxes = _numba_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    maxxes=maxxes,
                    lengths=lengths,
                    region=region,
                    posn=posn,
                    num=num,
                )
            r_count += 1
            posn_ = np.uintp(posn)
            # have we exceeded the size of this column?
            # do we need to trim and move data to other columns?
            check = (lengths[posn_] == (load_factor * 2)) & (
                r_count < right_regions.size
            )
            if check:
                (
                    sorted_array,
                    positions_array,
                    lengths,
                    maxxes,
                    maxxes_counter,
                ) = _expand_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    lengths=lengths,
                    maxxes=maxxes,
                    posn=posn,
                    maxxes_counter=maxxes_counter,
                    load_factor=load_factor,
                )
        # now we do a binary search
        # for left region in right region
        # 1. find the position in maxxes
        # - this indicates which column in sorted_arrays contains our region
        # 2. search in the specific region for the positions
        # where left_region <= right_region
        l_region = left_regions[_indexer]
        arr = maxxes[:maxxes_counter]
        if l_region > arr[-1]:
            end = start
            continue
        posn = _numba_less_than_base(arr=arr, value=l_region)
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than_base(arr=arr, value=l_region)
        difference = len_arr - _posn
        total += difference
        # step into the remaining columns
        for ind in range(posn + 1, maxxes_counter):
            ind_ = np.uintp(ind)
            len_arr = lengths[ind_]
            difference += len_arr
            total += len_arr
        l_booleans[_indexer] = True
        end = start
    if total == 0:
        return None, None, None
    # second pass - fill arrays with indices
    length = left_regions.size
    end = right_regions.size
    end -= 1
    region = right_regions[np.uintp(end)]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    left_counts = np.zeros(length, dtype=np.intp)
    left_indices = np.empty(length, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    l_indexer = 0
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uintp(indexer)
        if not l_booleans[_indexer]:
            l_indexer += 1
            continue
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_regions[_num]
            arr = maxxes[:maxxes_counter]
            if region > arr[-1]:
                posn = maxxes_counter - 1
                posn_ = np.uintp(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn = _numba_less_than_base(arr=arr, value=region)
                sorted_array, positions_array, lengths, maxxes = _numba_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    maxxes=maxxes,
                    lengths=lengths,
                    region=region,
                    posn=posn,
                    num=num,
                )
            r_count += 1
            posn_ = np.uintp(posn)
            # have we reached the max size of this column?
            # do we need to trim and move data to other columns?
            check = (lengths[posn_] == (load_factor * 2)) & (
                r_count < right_regions.size
            )
            if check:
                (
                    sorted_array,
                    positions_array,
                    lengths,
                    maxxes,
                    maxxes_counter,
                ) = _expand_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    lengths=lengths,
                    maxxes=maxxes,
                    posn=posn,
                    maxxes_counter=maxxes_counter,
                    load_factor=load_factor,
                )
        # now we do a binary search
        # for left region in right region
        l_region = left_regions[_indexer]
        arr = maxxes[:maxxes_counter]
        left_indices[np.uintp(l_indexer)] = indexer
        if l_region > arr[-1]:
            end = start
            l_indexer += 1
            continue
        counter = 0
        posn = _numba_less_than_base(arr=arr, value=l_region)
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than_base(arr=arr, value=l_region)
        for ind in range(_posn, len_arr):
            ind_ = np.uintp(ind)
            begin_ = np.uintp(begin)
            r_pos = positions_array[ind_, posn_]
            right_indices[begin_] = r_pos
            begin += 1
            counter += 1
        for ind in range(posn + 1, maxxes_counter):
            ind_ = np.uintp(ind)
            len_arr = lengths[ind_]
            for num in range(len_arr):
                _num = np.uintp(num)
                begin_ = np.uintp(begin)
                r_pos = positions_array[_num, ind_]
                right_indices[begin_] = r_pos
                begin += 1
                counter += 1
        left_counts[l_indexer] = counter
        left_indices[l_indexer] = indexer
        l_indexer += 1
        end = start
    return left_indices, right_indices, left_counts


@njit(cache=True)
def _numba_non_equi_join_not_monotonic_dual_keep_all(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
    row_count: bool,
):
    """
    Get indices for non-equi join,
    where the right regions are not monotonic
    """
    # first pass - get actual length
    length = left_index.size
    end = right_index.size
    end -= 1
    # add the last region
    # no need to have this checked within an if-else statement
    # in the for loop below
    region = right_regions[np.uintp(end)]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    # keep track of the maxxes array
    # how many cells have actual values?
    maxxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    if row_count:
        counts = np.zeros(length, dtype=np.intp)
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uintp(indexer)
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_regions[_num]
            arr = maxxes[:maxxes_counter]
            if region > arr[-1]:
                # it is larger than the max in the maxxes array
                # shove it into the last column
                posn = maxxes_counter - 1
                posn_ = np.uintp(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                # we dont need to compute positions in the first run?
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn = _numba_less_than_base(arr=arr, value=region)
                sorted_array, positions_array, lengths, maxxes = _numba_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    maxxes=maxxes,
                    lengths=lengths,
                    region=region,
                    posn=posn,
                    num=num,
                )
            r_count += 1
            posn_ = np.uintp(posn)
            # have we exceeded the size of this column?
            # do we need to trim and move data to other columns?
            check = (lengths[posn_] == (load_factor * 2)) & (r_count < right_index.size)
            if check:
                (
                    sorted_array,
                    positions_array,
                    lengths,
                    maxxes,
                    maxxes_counter,
                ) = _expand_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    lengths=lengths,
                    maxxes=maxxes,
                    posn=posn,
                    maxxes_counter=maxxes_counter,
                    load_factor=load_factor,
                )
        # now we do a binary search
        # for left region in right region
        # 1. find the position in maxxes
        # - this indicates which column in sorted_arrays contains our region
        # 2. search in the specific region for the positions
        # where left_region <= right_region
        l_region = left_regions[_indexer]
        arr = maxxes[:maxxes_counter]
        if l_region > arr[-1]:
            end = start
            continue
        posn = _numba_less_than_base(arr=arr, value=l_region)
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than_base(arr=arr, value=l_region)
        l_count = len_arr - _posn
        total += len_arr - _posn
        # step into the remaining columns
        for ind in range(posn + 1, maxxes_counter):
            ind_ = np.uintp(ind)
            len_arr = lengths[ind_]
            l_count += len_arr
            total += len_arr
        if row_count:
            counts[_indexer] = l_count
        l_booleans[_indexer] = True
        end = start
    if total == 0:
        return None, None
    if row_count:
        return left_index, counts
    # second pass - fill arrays with indices
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uintp(end)]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uintp(indexer)
        if not l_booleans[_indexer]:
            continue
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_regions[_num]
            arr = maxxes[:maxxes_counter]
            if region > arr[-1]:
                posn = maxxes_counter - 1
                posn_ = np.uintp(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn = _numba_less_than_base(arr=arr, value=region)
                sorted_array, positions_array, lengths, maxxes = _numba_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    maxxes=maxxes,
                    lengths=lengths,
                    region=region,
                    posn=posn,
                    num=num,
                )
            r_count += 1
            posn_ = np.uintp(posn)
            # have we reached the max size of this column?
            # do we need to trim and move data to other columns?
            check = (lengths[posn_] == (load_factor * 2)) & (r_count < right_index.size)
            if check:
                (
                    sorted_array,
                    positions_array,
                    lengths,
                    maxxes,
                    maxxes_counter,
                ) = _expand_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    lengths=lengths,
                    maxxes=maxxes,
                    posn=posn,
                    maxxes_counter=maxxes_counter,
                    load_factor=load_factor,
                )
        # now we do a binary search
        # for left region in right region
        l_region = left_regions[_indexer]
        arr = maxxes[:maxxes_counter]
        if l_region > arr[-1]:
            end = start
            continue
        posn = _numba_less_than_base(arr=arr, value=l_region)
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than_base(arr=arr, value=l_region)
        l_index = left_index[_indexer]
        for ind in range(_posn, len_arr):
            ind_ = np.uintp(ind)
            begin_ = np.uintp(begin)
            r_pos = positions_array[ind_, posn_]
            r_pos = np.uintp(r_pos)
            r_index = right_index[r_pos]
            left_indices[begin_] = l_index
            right_indices[begin_] = r_index
            begin += 1
        for ind in range(posn + 1, maxxes_counter):
            ind_ = np.uintp(ind)
            len_arr = lengths[ind_]
            for num in range(len_arr):
                _num = np.uintp(num)
                begin_ = np.uintp(begin)
                left_indices[begin_] = l_index
                r_pos = positions_array[_num, ind_]
                r_pos = np.uintp(r_pos)
                r_index = right_index[r_pos]
                right_indices[begin_] = r_index
                begin += 1
        end = start
    return left_indices, right_indices


@njit(cache=True)
def _numba_non_equi_join_not_monotonic_dual_keep_first(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
):
    """
    Get indices for non-equi join - first match
    """
    # first pass - get the actual length
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uintp(end)]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uintp(indexer)
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_regions[_num]
            arr = maxxes[:maxxes_counter]
            if region > arr[-1]:
                posn = maxxes_counter - 1
                posn_ = np.uintp(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn = _numba_less_than_base(arr=arr, value=region)
                sorted_array, positions_array, lengths, maxxes = _numba_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    maxxes=maxxes,
                    lengths=lengths,
                    region=region,
                    posn=posn,
                    num=num,
                )
            r_count += 1
            posn_ = np.uintp(posn)
            # have we exceeded the size of this column?
            # do we need to trim and move data to other columns?
            check = (lengths[posn_] == (load_factor * 2)) & (r_count < right_index.size)
            if check:
                (
                    sorted_array,
                    positions_array,
                    lengths,
                    maxxes,
                    maxxes_counter,
                ) = _expand_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    lengths=lengths,
                    maxxes=maxxes,
                    posn=posn,
                    maxxes_counter=maxxes_counter,
                    load_factor=load_factor,
                )
        l_region = left_regions[_indexer]
        arr = maxxes[:maxxes_counter]
        if l_region > arr[-1]:
            end = start
            continue
        posn = _numba_less_than_base(arr=arr, value=l_region)
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than_base(arr=arr, value=l_region)
        base_index = -1
        for ind in range(_posn, len_arr):
            ind_ = np.uintp(ind)
            r_pos = positions_array[ind_, posn_]
            r_pos = np.uintp(r_pos)
            r_index = right_index[r_pos]
            if (base_index == -1) | (r_index < base_index):
                base_index = r_index
        # step into the remaining columns
        for ind in range(posn + 1, maxxes_counter):
            ind_ = np.uintp(ind)
            len_arr = lengths[ind_]
            # step into the rows for each column
            for num in range(len_arr):
                _num = np.uintp(num)
                r_pos = positions_array[_num, ind_]
                r_pos = np.uintp(r_pos)
                r_index = right_index[r_pos]
                if (base_index == -1) | (r_index < base_index):
                    base_index = r_index
        total += 1
        l_booleans[_indexer] = True
        r_indices[_indexer] = base_index
        end = start
    if total == 0:
        return None, None
    # second pass - fill arrays with indices
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uintp(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uintp(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit(cache=True)
def _numba_non_equi_join_not_monotonic_dual_keep_last(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
):
    """
    Get indices for non-equi join - last match
    """
    # first pass - get the actual length
    length = left_index.size
    end = right_index.size
    end -= 1
    region = right_regions[np.uintp(end)]
    sorted_array[0, 0] = region
    positions_array[0, 0] = end
    maxxes_counter = 1
    maxxes[0] = region
    lengths[0] = 1
    r_count = 0
    total = 0
    l_booleans = np.zeros(length, dtype=np.bool_)
    r_indices = np.empty(length, dtype=np.intp)
    for indexer in range(length - 1, -1, -1):
        _indexer = np.uintp(indexer)
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_regions[_num]
            arr = maxxes[:maxxes_counter]
            if region > arr[-1]:
                posn = maxxes_counter - 1
                posn_ = np.uintp(posn)
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                posn = _numba_less_than_base(arr=arr, value=region)
                sorted_array, positions_array, lengths, maxxes = _numba_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    maxxes=maxxes,
                    lengths=lengths,
                    region=region,
                    posn=posn,
                    num=num,
                )
            r_count += 1
            posn_ = np.uintp(posn)
            # have we exceeded the size of this column?
            # do we need to trim and move data to other columns?
            check = (lengths[posn_] == (load_factor * 2)) & (r_count < right_index.size)
            if check:
                (
                    sorted_array,
                    positions_array,
                    lengths,
                    maxxes,
                    maxxes_counter,
                ) = _expand_sorted_array(
                    sorted_array=sorted_array,
                    positions_array=positions_array,
                    lengths=lengths,
                    maxxes=maxxes,
                    posn=posn,
                    maxxes_counter=maxxes_counter,
                    load_factor=load_factor,
                )
        l_region = left_regions[_indexer]
        arr = maxxes[:maxxes_counter]
        posn = _numba_less_than_base(arr=arr, value=l_region)
        if l_region > arr[-1]:
            end = start
            continue
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        arr = sorted_array[:len_arr, posn_]
        _posn = _numba_less_than_base(arr=arr, value=l_region)
        base_index = np.inf
        for ind in range(_posn, len_arr):
            ind_ = np.uintp(ind)
            r_pos = positions_array[ind_, posn_]
            r_pos = np.uintp(r_pos)
            r_index = right_index[r_pos]
            if (base_index == np.inf) | (r_index > base_index):
                base_index = r_index
        # step into the remaining columns
        for ind in range(posn + 1, maxxes_counter):
            ind_ = np.uintp(ind)
            len_arr = lengths[ind_]
            # step into the rows for each column
            for num in range(len_arr):
                _num = np.uintp(num)
                r_pos = positions_array[_num, ind_]
                r_pos = np.uintp(r_pos)
                r_index = right_index[r_pos]
                if (base_index == np.inf) | (r_index > base_index):
                    base_index = r_index
        total += 1
        l_booleans[_indexer] = True
        r_indices[_indexer] = base_index
        end = start
    if total == 0:
        return None, None
    # second pass - fill arrays with indices
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    n = 0
    for ind in range(length):
        _ind = np.uintp(ind)
        if not l_booleans[_ind]:
            continue
        _n = np.uintp(n)
        left_indices[_n] = left_index[_ind]
        right_indices[_n] = r_indices[_ind]
        n += 1
    return left_indices, right_indices


@njit
def _numba_sorted_array(
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    region: int,
    posn: int,
    num: int,
) -> tuple:
    """
    Adaptation of grantjenk's sortedcontainers.

    Args:
        sorted_array: array of regions to keep in sorted order.
        positions_array: positions of regions in the sorted_array.
        maxxes: array of max values per column in the sorted_array.
        lengths: array of lengths per column in the sorted_array.
        region: integer to insert into sorted_array.
        posn: binary search position of region in maxxes array.
            Determines which column in the sorted_array
            the region will go to.
        num: position of region in right_regions array.
            Inserted into positions_array to keep
            in sync with the region the sorted_array.
    """
    # the sorted array is an adaptation
    # of grantjenks' sortedcontainers
    posn_ = np.uintp(posn)
    len_arr = lengths[posn_]
    # grab the specific column that the region falls into
    arr = sorted_array[:len_arr, posn_]
    # get the insertion position for the region
    insort_posn = _numba_less_than_base(arr=arr, value=region)
    # make space for the region
    # shift downwards before inserting
    # shift in this order to avoid issues with assignment override
    # which could create wrong values
    for ind in range(len_arr - 1, insort_posn - 1, -1):
        ind_ = np.uintp(ind)
        _ind = np.uintp(ind + 1)
        sorted_array[_ind, posn_] = sorted_array[ind_, posn_]
        positions_array[_ind, posn_] = positions_array[ind_, posn_]
    # now we can safely insert the region
    insort = np.uintp(insort_posn)
    sorted_array[insort, posn_] = region
    positions_array[insort, posn_] = num
    # update the length and the maxxes arrays
    lengths[posn_] += 1
    maxxes[posn_] = sorted_array[np.uintp(len_arr), posn_]
    return sorted_array, positions_array, lengths, maxxes


@njit
def _expand_sorted_array(
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    lengths: np.ndarray,
    maxxes: np.ndarray,
    posn: int,
    maxxes_counter: int,
    load_factor: int,
):
    """
    Expand sorted_array if it exceeds load_factor * 2
    Adapted from grantjenks' sortedcontainers.

    Args:
        sorted_array: array of regions to keep in sorted order.
        positions_array: positions of regions in the sorted_array.
        maxxes: array of max values per column in the sorted_array.
        lengths: array of lengths per column in the sorted_array.
        region: integer to insert into sorted_array.
        posn: binary search position of region in maxxes array.
            Determines which column in the sorted_array
            the region will go to.
        maxxes_counter: keeps a count of the number
            of entries in the maxxes array that have
            actual values.
        num: position of region in right_regions array.
            Inserted into positions_array to keep
            in sync with the region the sorted_array.
    """
    # shift from left+1 to right
    for pos in range(maxxes_counter - 1, posn, -1):
        forward = np.uintp(pos + 1)
        current = np.uintp(pos)
        sorted_array[:, forward] = sorted_array[:, current]
        positions_array[:, forward] = positions_array[:, current]
        maxxes[forward] = maxxes[current]
        lengths[forward] = lengths[current]
    # share half the load from left to left+1
    forward = np.uintp(posn + 1)
    current = np.uintp(posn)
    maxxes[forward] = sorted_array[-1, current]
    lengths[forward] = load_factor
    sorted_array[:load_factor, forward] = sorted_array[load_factor:, current]
    positions_array[:load_factor, forward] = positions_array[load_factor:, current]
    # update the length and maxxes arrays
    lengths[current] = load_factor
    maxxes[current] = sorted_array[np.uintp(load_factor - 1), current]
    maxxes_counter += 1
    return sorted_array, positions_array, lengths, maxxes, maxxes_counter
