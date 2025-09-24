"""Various conditional_join functions powered by Numba"""

from __future__ import annotations

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
def _get_indices_ranges_keep_all(
    booleans: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    matches: np.ndarray,
    sizes: np.ndarray,
    counts_array: np.ndarray,
    tuples: tuple,
) -> np.ndarray | tuple[None, None]:
    """
    Get indices for starts and ends,
    after iterating through `tuples`
    """
    booleans, counts_array, matches, total, _ = _get_positive_matches_ranges(
        tuples=tuples,
        starts=starts,
        ends=ends,
        matches=matches,
        booleans=booleans,
        sizes=sizes,
        counts_array=counts_array,
    )
    if booleans is None:
        return None, None
    total = total[np.uintp(0)]
    left_indices = np.empty(total, dtype=np.intp)
    begin = 0
    for n in range(booleans.size):
        n_ = np.uintp(n)
        if not booleans[n_]:
            continue
        size = counts_array[n_]
        val = left_index[n_]
        for _ in range(size):
            begin_ = np.uintp(begin)
            left_indices[begin_] = val
            begin += 1
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    startt = 0
    for n in range(booleans.size):
        _n = np.uintp(n)
        start = starts[_n]
        end = ends[_n]
        if not booleans[_n]:
            size = sizes[_n]
            startt += size
            continue
        for nn in range(start, end):
            if not matches[np.uintp(startt)]:
                startt += 1
                continue
            nn_ = np.uintp(nn)
            val = right_index[nn_]
            begin_ = np.uintp(begin)
            right_indices[begin_] = val
            begin += 1
            startt += 1
    return left_indices, right_indices


@njit(nogil=True)
def _get_indices_ranges_keep_first(
    booleans: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    matches: np.ndarray,
    sizes: np.ndarray,
    counts_array: np.ndarray,
    tuples: tuple,
) -> np.ndarray | tuple[None, None]:
    """
    Get indices for starts and ends,
    after iterating through `tuples`
    """
    booleans, counts_array, matches, _, total = _get_positive_matches_ranges(
        tuples=tuples,
        starts=starts,
        ends=ends,
        matches=matches,
        booleans=booleans,
        sizes=sizes,
        counts_array=counts_array,
    )
    if booleans is None:
        return None, None
    total = total[np.uintp(0)]
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    startt = 0
    for n in range(booleans.size):
        _n = np.uintp(n)
        start = starts[_n]
        end = ends[_n]
        if not booleans[_n]:
            size = sizes[_n]
            startt += size
            continue
        base = -1
        for nn in range(start, end):
            if not matches[np.uintp(startt)]:
                startt += 1
                continue
            nn_ = np.uintp(nn)
            r_val = right_index[nn_]
            boolean = (base < 0) or (base > r_val)
            if boolean:
                base = r_val
            startt += 1
        begin_ = np.uintp(begin)
        l_val = left_index[_n]
        left_indices[begin_] = l_val
        right_indices[begin_] = base
        begin += 1
    return left_indices, right_indices


@njit(nogil=True)
def _get_indices_ranges_keep_last(
    booleans: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    matches: np.ndarray,
    sizes: np.ndarray,
    counts_array: np.ndarray,
    tuples: tuple,
) -> np.ndarray | tuple[None, None]:
    """
    Get indices for starts and ends,
    after iterating through `tuples`
    """
    booleans, counts_array, matches, _, total = _get_positive_matches_ranges(
        tuples=tuples,
        starts=starts,
        ends=ends,
        matches=matches,
        booleans=booleans,
        sizes=sizes,
        counts_array=counts_array,
    )
    if booleans is None:
        return None, None
    total = total[np.uintp(0)]
    left_indices = np.empty(total, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    startt = 0
    for n in range(booleans.size):
        _n = np.uintp(n)
        start = starts[_n]
        end = ends[_n]
        if not booleans[_n]:
            size = sizes[_n]
            startt += size
            continue
        base = -1
        for nn in range(start, end):
            if not matches[np.uintp(startt)]:
                startt += 1
                continue
            nn_ = np.uintp(nn)
            r_val = right_index[nn_]
            boolean = base < r_val
            if boolean:
                base = r_val
            startt += 1
        begin_ = np.uintp(begin)
        l_val = left_index[_n]
        left_indices[begin_] = l_val
        right_indices[begin_] = base
        begin += 1
    return left_indices, right_indices


@njit(nogil=True)
def _get_indices_for_regions_keep_all(
    booleans: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    left_region: np.ndarray,
    right_region: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    load_factor: int,
    tuples: tuple,
) -> np.ndarray | tuple[None, None]:
    """
    Get indices where left regions are both <= right regions;
    applies where join conditions are more than two,
    and >/>=/</<= count > 1
    """
    booleans, right_positions, sizes, total, _ = (
        _numba_get_matches_from_regions(
            left_region=left_region,
            right_region=right_region,
            booleans=booleans,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    )
    if booleans is None:
        return None, None
    total = total[np.uintp(0)]
    if tuples is None:
        left_indices = np.empty(total, dtype=np.intp)
        begin = 0
        for n in range(booleans.size):
            n_ = np.uintp(n)
            if not booleans[n_]:
                continue
            size = sizes[n_]
            val = left_index[n_]
            for _ in range(size):
                begin_ = np.uintp(begin)
                left_indices[begin_] = val
                begin += 1
        right_indices = np.empty(total, dtype=np.intp)
        for n in range(total):
            n_ = np.uintp(n)
            pos = right_positions[n_]
            pos_ = np.uintp(pos)
            val = right_index[pos_]
            right_indices[n_] = val
        return left_indices, right_indices
    matches = np.ones(total, dtype=np.bool_)
    counts_array = np.zeros(booleans.size, dtype=np.intp)
    booleans, matches, counts_array, total, _ = _get_positive_matches_regions(
        tuples=tuples,
        right_positions=right_positions,
        sizes=sizes,
        counts_array=counts_array,
        booleans=booleans,
        matches=matches,
    )
    if total is None:
        return None, None
    left_indices = np.empty(total[np.uintp(0)], dtype=np.intp)
    begin = 0
    for n in range(booleans.size):
        n_ = np.uintp(n)
        if not booleans[n_]:
            continue
        size = counts_array[n_]
        val = left_index[n_]
        for _ in range(size):
            begin_ = np.uintp(begin)
            left_indices[begin_] = val
            begin += 1
    right_indices = np.empty(total[np.uintp(0)], dtype=np.intp)
    begin = 0
    for n in range(matches.size):
        n_ = np.uintp(n)
        if not matches[n_]:
            continue
        pos = right_positions[n_]
        pos_ = np.uintp(pos)
        val = right_index[pos_]
        begin_ = np.uintp(begin)
        right_indices[begin_] = val
        begin += 1
    return left_indices, right_indices


@njit(nogil=True)
def _get_indices_for_regions_keep_first(
    booleans: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    left_region: np.ndarray,
    right_region: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    load_factor: int,
    tuples: tuple,
) -> np.ndarray | tuple[None, None]:
    """
    Get indices where left regions are both <= right regions;
    applies where join conditions are more than two,
    and >/>=/</<= count > 1
    """
    booleans, right_positions, sizes, total, l_count = (
        _numba_get_matches_from_regions(
            left_region=left_region,
            right_region=right_region,
            booleans=booleans,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    )
    if booleans is None:
        return None, None
    if tuples is None:
        total = l_count[np.uintp(0)]
        left_indices = np.empty(total, dtype=np.intp)
        right_indices = np.empty(total, dtype=np.intp)
        begin = 0  # indexer for final indices
        start = 0  # indexer into right_positions
        for n in range(booleans.size):
            n_ = np.uintp(n)
            size = sizes[n_]
            if not booleans[n_]:
                # stay in sync; always increment
                # when moving to the next iteration
                start += size
                continue
            base = -1
            for _ in range(size):
                start_ = np.uintp(start)
                pos = right_positions[start_]
                pos_ = np.uintp(pos)
                r_val = right_index[pos_]
                boolean = (base < 0) or (base > r_val)
                if boolean:
                    base = r_val
                start += 1
            begin_ = np.uintp(begin)
            right_indices[begin_] = base
            l_val = left_index[n_]
            left_indices[begin_] = l_val
            begin += 1
        return left_indices, right_indices
    matches = np.ones(total[np.uintp(0)], dtype=np.bool_)
    counts_array = np.zeros(booleans.size, dtype=np.intp)
    booleans, matches, counts_array, _, total = _get_positive_matches_regions(
        tuples=tuples,
        right_positions=right_positions,
        sizes=sizes,
        counts_array=counts_array,
        booleans=booleans,
        matches=matches,
    )
    if total is None:
        return None, None
    left_indices = np.empty(total[np.uintp(0)], dtype=np.intp)
    right_indices = np.empty(total[np.uintp(0)], dtype=np.intp)
    begin = 0  # indexer for final indices
    start = 0  # indexer into right_positions
    for n in range(booleans.size):
        n_ = np.uintp(n)
        size = sizes[n_]
        if not booleans[n_]:
            # stay in sync; always increment
            # when moving to the next iteration
            start += size
            continue
        base = -1
        for _ in range(size):
            start_ = np.uintp(start)
            if not matches[start_]:
                # stay in sync; always increment
                # when moving to the next iteration
                start += 1
                continue
            start_ = np.uintp(start)
            pos = right_positions[start_]
            pos_ = np.uintp(pos)
            r_val = right_index[pos_]
            boolean = (base < 0) or (base > r_val)
            if boolean:
                base = r_val
            start += 1
        begin_ = np.uintp(begin)
        right_indices[begin_] = base
        l_val = left_index[n_]
        left_indices[begin_] = l_val
        begin += 1
    return left_indices, right_indices


@njit(nogil=True)
def _get_indices_for_regions_keep_last(
    booleans: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    left_region: np.ndarray,
    right_region: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    load_factor: int,
    tuples: tuple,
) -> np.ndarray | tuple[None, None]:
    """
    Get indices where left regions are both <= right regions;
    applies where join conditions are more than two,
    and >/>=/</<= count > 1
    """
    booleans, right_positions, sizes, total, l_count = (
        _numba_get_matches_from_regions(
            left_region=left_region,
            right_region=right_region,
            booleans=booleans,
            maxxes=maxxes,
            lengths=lengths,
            sorted_array=sorted_array,
            positions_array=positions_array,
            starts=starts,
            load_factor=load_factor,
        )
    )
    if booleans is None:
        return None, None
    if tuples is None:
        total = l_count[np.uintp(0)]
        left_indices = np.empty(total, dtype=np.intp)
        right_indices = np.empty(total, dtype=np.intp)
        begin = 0  # indexer for final indices
        start = 0  # indexer into right_positions
        for n in range(booleans.size):
            n_ = np.uintp(n)
            size = sizes[n_]
            if not booleans[n_]:
                # stay in sync; always increment
                # when moving to the next iteration
                start += size
                continue
            base = -1
            for _ in range(size):
                start_ = np.uintp(start)
                pos = right_positions[start_]
                pos_ = np.uintp(pos)
                r_val = right_index[pos_]
                boolean = base < r_val
                if boolean:
                    base = r_val
                start += 1
            begin_ = np.uintp(begin)
            right_indices[begin_] = base
            l_val = left_index[n_]
            left_indices[begin_] = l_val
            begin += 1
        return left_indices, right_indices
    matches = np.ones(total[np.uintp(0)], dtype=np.bool_)
    counts_array = np.zeros(booleans.size, dtype=np.intp)
    booleans, matches, counts_array, _, total = _get_positive_matches_regions(
        tuples=tuples,
        right_positions=right_positions,
        sizes=sizes,
        counts_array=counts_array,
        booleans=booleans,
        matches=matches,
    )
    if total is None:
        return None, None
    left_indices = np.empty(total[np.uintp(0)], dtype=np.intp)
    right_indices = np.empty(total[np.uintp(0)], dtype=np.intp)
    begin = 0  # indexer for final indices
    start = 0  # indexer into right_positions
    for n in range(booleans.size):
        n_ = np.uintp(n)
        size = sizes[n_]
        if not booleans[n_]:
            # stay in sync; always increment
            # when moving to the next iteration
            start += size
            continue
        base = -1
        for _ in range(size):
            start_ = np.uintp(start)
            if not matches[start_]:
                # stay in sync; always increment
                # when moving to the next iteration
                start += 1
                continue
            start_ = np.uintp(start)
            pos = right_positions[start_]
            pos_ = np.uintp(pos)
            r_val = right_index[pos_]
            boolean = base < r_val
            if boolean:
                base = r_val
            start += 1
        begin_ = np.uintp(begin)
        right_indices[begin_] = base
        l_val = left_index[n_]
        left_indices[begin_] = l_val
        begin += 1
    return left_indices, right_indices


@njit(nogil=True, parallel=True)
def _update_search_indices_less_than(
    length: int,
    booleans: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sizes: np.ndarray,
):
    """
    update start positions for '<='
    """
    match = 0
    total = 0
    for n in prange(length):
        _n = np.uintp(n)
        if not booleans[_n]:
            sizes[_n] = 0
            continue
        end = ends[_n]
        l_value = left[_n]
        # adapted from numba/np/array_math.py
        min_idx = starts[_n]
        max_idx = ends[_n]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            idx = np.uintp(mid_idx)
            current_value = right[idx]
            if current_value < l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        if min_idx == end:
            booleans[_n] = False
            sizes[_n] = False
            continue
        booleans[_n] = True
        starts[_n] = min_idx
        size = end - min_idx
        sizes[_n] = size
        total += size
        match += 1
    return (
        starts,
        ends,
        booleans,
        sizes,
        np.asarray([total]),
        np.asarray([match]),
    )


@njit(nogil=True, parallel=True)
def _update_search_indices_less_than_strict(
    length: int,
    booleans: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sizes: np.ndarray,
):
    """
    update start positions for '<'
    """
    match = 0
    total = 0
    for n in prange(length):
        _n = np.uintp(n)
        if not booleans[_n]:
            sizes[_n] = 0
            continue
        end = ends[_n]
        l_value = left[_n]
        # adapted from numba/np/array_math.py
        min_idx = starts[_n]
        max_idx = ends[_n]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            idx = np.uintp(mid_idx)
            current_value = right[idx]
            if current_value <= l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        if min_idx == end:
            booleans[_n] = False
            sizes[_n] = 0
            continue
        idx = np.uintp(min_idx)
        current_value = right[idx]
        if current_value == l_value:
            booleans[_n] = False
            sizes[_n] = 0
            continue
        booleans[_n] = True
        starts[_n] = min_idx
        size = end - min_idx
        sizes[_n] = size
        total += size
        match += 1
    return (
        starts,
        ends,
        booleans,
        sizes,
        np.asarray([total]),
        np.asarray([match]),
    )


@njit(nogil=True, parallel=True)
def _update_search_indices_greater_than(
    length: int,
    booleans: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sizes: np.ndarray,
):
    """
    update end positions for '>='
    """
    match = 0
    total = 0
    for n in prange(length):
        _n = np.uintp(n)
        if not booleans[_n]:
            sizes[_n] = 0
            continue
        start = starts[_n]
        l_value = left[_n]
        # adapted from numba/np/array_math.py
        min_idx = starts[_n]
        max_idx = ends[_n]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            idx = np.uintp(mid_idx)
            current_value = right[idx]
            if current_value > l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        if min_idx == start:
            booleans[_n] = False
            sizes[_n] = 0
            continue
        booleans[_n] = True
        ends[_n] = min_idx
        size = min_idx - start
        sizes[_n] = size
        total += size
        match += 1
    return (
        starts,
        ends,
        booleans,
        sizes,
        np.asarray([total]),
        np.asarray([match]),
    )


@njit(nogil=True, parallel=True)
def _update_search_indices_greater_than_strict(
    length: int,
    booleans: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sizes: np.ndarray,
):
    """
    update end positions for '>'
    """
    match = 0
    total = 0
    for n in prange(length):
        _n = np.uintp(n)
        if not booleans[_n]:
            sizes[_n] = 0
            continue
        start = starts[_n]
        l_value = left[_n]
        # adapted from numba/np/array_math.py
        min_idx = starts[_n]
        max_idx = ends[_n]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            idx = np.uintp(mid_idx)
            current_value = right[idx]
            if current_value >= l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        if min_idx == start:
            booleans[_n] = False
            sizes[_n] = 0
            continue
        index = min_idx - 1
        idx = np.uintp(index)
        current_value = right[idx]
        if current_value == l_value:
            booleans[_n] = False
            sizes[_n] = 0
            continue
        booleans[_n] = True
        ends[_n] = min_idx
        size = min_idx - start
        sizes[_n] = size
        total += size
        match += 1
    return (
        starts,
        ends,
        booleans,
        sizes,
        np.asarray([total]),
        np.asarray([match]),
    )


@njit(nogil=True, parallel=False)
def _get_indices_equi_ge_gt_or_le_lt_join_keep_all(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
    tuples: tuple | None,
    ge_gt: tuple | None,
    le_lt: tuple | None,
):
    """
    Get indices for equi join and at least one >/>=/</<= join
    """
    length = starts.size
    sizes = np.empty(length, dtype=np.intp)
    if ge_gt is not None:
        left_array = ge_gt[0]
        right_array = ge_gt[1]
        op = ge_gt[2][0]
        if op == 0:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_greater_than_strict(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        else:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_greater_than(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        if match[np.uintp(0)] == 0:
            return None, None
    if le_lt is not None:
        left_array = le_lt[0]
        right_array = le_lt[1]
        op = le_lt[2][0]
        if op == 2:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_less_than_strict(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        else:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_less_than(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        if match[np.uintp(0)] == 0:
            return None, None
    total = total[np.uintp(0)]
    if tuples is None:
        left_indices = np.empty(total, dtype=np.intp)
        right_indices = np.empty(total, dtype=np.intp)
        begin = 0
        for n in range(booleans.size):
            n_ = np.uintp(n)
            if not booleans[n_]:
                continue
            start = starts[n_]
            end = ends[n_]
            l_val = left_index[n_]
            for nn in range(start, end):
                nn_ = np.uintp(nn)
                r_val = right_index[nn_]
                begin_ = np.uintp(begin)
                left_indices[begin_] = l_val
                right_indices[begin_] = r_val
                begin += 1
        return left_indices, right_indices
    matches = np.ones(total, dtype=np.bool_)
    counts_array = np.zeros(length, dtype=np.intp)
    return _get_indices_ranges_keep_all(
        booleans=booleans,
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        matches=matches,
        sizes=sizes,
        counts_array=counts_array,
        tuples=tuples,
    )


@njit(nogil=True, parallel=False)
def _get_indices_equi_ge_gt_or_le_lt_join_keep_first(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
    tuples: tuple | None,
    ge_gt: tuple | None,
    le_lt: tuple | None,
):
    """
    Build indices for equi join and range join
    """
    length = starts.size
    sizes = np.empty(length, dtype=np.intp)
    if ge_gt is not None:
        left_array = ge_gt[0]
        right_array = ge_gt[1]
        op = ge_gt[2][0]
        if op == 0:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_greater_than_strict(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        else:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_greater_than(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        if match[np.uintp(0)] == 0:
            return None, None
    if le_lt is not None:
        left_array = le_lt[0]
        right_array = le_lt[1]
        op = le_lt[2][0]
        if op == 2:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_less_than_strict(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        else:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_less_than(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        if match[np.uintp(0)] == 0:
            return None, None
    if tuples is None:
        match = match[np.uintp(0)]
        left_indices = np.empty(match, dtype=np.intp)
        right_indices = np.empty(match, dtype=np.intp)
        begin = 0
        for n in range(length):
            n_ = np.uintp(n)
            if not booleans[n_]:
                continue
            start = starts[n_]
            end = ends[n_]
            base = -1
            for nn in range(start, end):
                nn_ = np.uintp(nn)
                r_value = right_index[nn_]
                if (base < 0) or (base > r_value):
                    base = r_value
            begin_ = np.uintp(begin)
            l_value = left_index[n_]
            left_indices[begin_] = l_value
            right_indices[begin_] = base
            begin += 1
        return left_indices, right_indices
    matches = np.ones(total[np.uintp(0)], dtype=np.bool_)
    counts_array = np.zeros(length, dtype=np.intp)
    return _get_indices_ranges_keep_first(
        booleans=booleans,
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        matches=matches,
        sizes=sizes,
        counts_array=counts_array,
        tuples=tuples,
    )


@njit(nogil=True, parallel=False)
def _get_indices_equi_ge_gt_or_le_lt_join_keep_last(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    booleans: np.ndarray,
    tuples: tuple | None,
    ge_gt: tuple | None,
    le_lt: tuple | None,
):
    """
    Build indices for equi join and range join
    """
    length = starts.size
    sizes = np.empty(length, dtype=np.intp)
    if ge_gt is not None:
        left_array = ge_gt[0]
        right_array = ge_gt[1]
        op = ge_gt[2][0]
        if op == 0:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_greater_than_strict(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        else:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_greater_than(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        if match[np.uintp(0)] == 0:
            return None, None
    if le_lt is not None:
        left_array = le_lt[0]
        right_array = le_lt[1]
        op = le_lt[2][0]
        if op == 2:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_less_than_strict(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        else:
            starts, ends, booleans, sizes, total, match = (
                _update_search_indices_less_than(
                    length=length,
                    booleans=booleans,
                    starts=starts,
                    ends=ends,
                    left=left_array,
                    right=right_array,
                    sizes=sizes,
                )
            )
        if match[np.uintp(0)] == 0:
            return None, None
    if tuples is None:
        match = match[np.uintp(0)]
        left_indices = np.empty(match, dtype=np.intp)
        right_indices = np.empty(match, dtype=np.intp)
        begin = 0
        for n in range(length):
            n_ = np.uintp(n)
            if not booleans[n_]:
                continue
            start = starts[n_]
            end = ends[n_]
            base = -1
            for nn in range(start, end):
                nn_ = np.uintp(nn)
                r_value = right_index[nn_]
                if base < r_value:
                    base = r_value
            begin_ = np.uintp(begin)
            l_value = left_index[n_]
            left_indices[begin_] = l_value
            right_indices[begin_] = base
            begin += 1
        return left_indices, right_indices
    matches = np.ones(total[np.uintp(0)], dtype=np.bool_)
    counts_array = np.zeros(length, dtype=np.intp)
    return _get_indices_ranges_keep_last(
        booleans=booleans,
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        matches=matches,
        sizes=sizes,
        counts_array=counts_array,
        tuples=tuples,
    )


@njit(nogil=True)
def _get_positive_matches_ranges(
    tuples: tuple,
    starts: np.ndarray,
    ends: np.ndarray,
    matches: np.ndarray,
    booleans: np.ndarray,
    sizes: np.ndarray,
    counts_array: np.ndarray,
):
    """
    Compute matching locations for multiple conditions
     Applies if there is a `starts` and `ends`
    """
    length = starts.size
    for _tuple in literal_unroll(tuples):
        left_arr = _tuple[0]
        right_arr = _tuple[1]
        op = _tuple[2]
        is_extension_array = _tuple[3]
        left_booleans = _tuple[4]
        right_booleans = _tuple[5]
        total = 0
        l_counts = 0
        begin = 0
        for n in range(length):
            _n = np.uintp(n)
            size = sizes[_n]
            # stay in sync; always increment
            # this ensures correct selection of match/right_positions
            if not booleans[_n]:
                begin += size
                counts_array[_n] = 0
                continue
            count = 0
            left_val = left_arr[_n]
            l_bool = None
            r_bool = None
            start = starts[_n]
            end = ends[_n]
            for nn in range(start, end):
                begin_ = np.uintp(begin)
                if not matches[begin_]:
                    begin += 1
                    continue
                _nn = np.uintp(nn)
                right_val = right_arr[_nn]
                if op == 5:
                    l_bool = left_booleans[_n]
                    r_bool = right_booleans[_nn]
                    boolean = l_bool or r_bool
                    # pandas' pd.NA uses a different logic from np.nan
                    # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
                    if boolean and is_extension_array:
                        boolean = False
                    elif not boolean:
                        boolean = _compare(left_val, right_val, op)
                else:
                    boolean = _compare(left_val, right_val, op)
                matches[begin_] = boolean
                begin += 1
                boolean_int = int(boolean)
                total += boolean_int
                count += boolean_int
            counts_array[_n] = count
            boolean = count > 0
            booleans[_n] = boolean
            l_counts += int(boolean)
        if total == 0:
            return None, None, None, None, None
    return (
        booleans,
        counts_array,
        matches,
        np.array([total]),
        np.array([l_counts]),
    )


@njit(nogil=True)
def _get_positive_matches_regions(
    tuples: tuple,
    right_positions: np.ndarray,
    sizes: np.ndarray,
    counts_array: np.ndarray,
    booleans: np.ndarray,
    matches: np.ndarray,
):
    """
    Get matching locations for multiple conditions;
    kicks in after getting matches for regions
    """
    length = booleans.size
    for _tuple in literal_unroll(tuples):
        left_arr = _tuple[0]
        right_arr = _tuple[1]
        op = _tuple[2]
        is_extension_array = _tuple[3]
        left_booleans = _tuple[4]
        right_booleans = _tuple[5]
        total = 0  # total number of successful matches
        begin = 0  # tracker for matches insertion/update/extraction
        l_counts = 0  # track successful match per left index
        for n in range(length):
            count = 0
            _n = np.uintp(n)
            size = sizes[_n]
            # stay in sync; always increment
            # this ensures correct selection of match/right_positions
            if not booleans[_n]:
                begin += size
                counts_array[_n] = 0
                continue
            left_val = left_arr[_n]
            for _ in range(size):
                begin_ = np.uintp(begin)
                if not matches[begin_]:
                    # stay in sync; always increment
                    # when moving to the next iteration
                    begin += 1
                    continue
                # select position for right array
                pos = right_positions[begin_]
                pos_ = np.uintp(pos)
                # select actual value to compare
                # with left_val
                right_val = right_arr[pos_]

                l_bool = None
                r_bool = None
                if op == 5:
                    l_bool = left_booleans[_n]
                    r_bool = right_booleans[pos_]
                    boolean = l_bool or r_bool
                    if not boolean:
                        boolean = _compare(left_val, right_val, op)
                    # pandas' pd.NA uses a different logic from np.nan
                    # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
                    elif boolean and is_extension_array:
                        boolean = False
                else:
                    boolean = _compare(left_val, right_val, op)
                matches[begin_] = boolean
                begin += 1
                boolean_int = int(boolean)
                total += boolean_int
                count += boolean_int
            boolean = count > 0
            booleans[_n] = boolean
            counts_array[_n] = count
            l_counts += int(boolean)
    return (
        booleans,
        matches,
        counts_array,
        np.array([total], dtype=np.intp),
        np.array([l_counts], dtype=np.intp),
    )


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
        return left_val == right_val
    return left_val != right_val


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


@njit(nogil=True)
def _numba_get_matches_from_regions(
    left_region: np.ndarray,
    right_region: np.ndarray,
    booleans: np.ndarray,
    maxxes: np.ndarray,
    lengths: np.ndarray,
    sorted_array: np.ndarray,
    positions_array: np.ndarray,
    starts: np.ndarray,
    load_factor: int,
):
    """
    Get indices for non-equi join;
    applies to joins where >/>=/</<= count
    is greater than 1
    """
    # first pass - get actual length
    length = left_region.size
    len_right = right_region.size
    end = len_right - 1
    # keep track of the maxxes array
    # how many cells have actual values?
    maxxes_counter = 0
    # add the last positive region
    # no need to have this checked within an if-else statement
    # in the for loop below
    for indexer in range(len_right - 1, -1, -1):
        region = right_region[np.uintp(indexer)]
        # a negative region indicates there is no match
        # minimum region should be 0
        if region >= 0:
            end = indexer
            break
    # the largest left region position,
    # which should be our starting point,
    # should be less than `end`
    do_not_run = True
    for indexer in range(length):
        _indexer = np.uintp(indexer)
        if not booleans[_indexer]:
            continue
        start = starts[_indexer]
        # python is zero indexing
        # hence the addition of 1
        # e.g iteration from
        # a start of 5 to
        # and end of 8
        # yields 5, 6, 7
        if start >= (end + 1):
            booleans[_indexer] = False
            continue
        do_not_run = False
        break
    if do_not_run:
        return None, None, None, None, None
    zero_index = np.uintp(0)
    maxxes_counter = 1
    maxxes[zero_index] = region
    sorted_array[zero_index, zero_index] = region
    positions_array[zero_index, zero_index] = end
    lengths[zero_index] = 1
    base_end = end
    base_region = region
    # keep track of iterations through the right_region
    # used to determine if the sorted_array should be expanded
    r_count = 0
    # capture total positive matches
    total = 0
    l_count = 0  # how many left_regions actually have a match
    for indexer in range(length):
        _indexer = np.uintp(indexer)
        if not booleans[_indexer]:
            continue
        start = starts[_indexer]
        # in this for-loop section
        # we build the sorted array
        # for the specified range start -> end
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_region[_num]
            # a negative region indicates there is no match
            # minimum region should be 0
            if region < 0:
                r_count += 1
                continue
            posn = maxxes_counter - 1
            posn_ = np.uintp(posn)
            if region > maxxes[posn_]:
                # it is larger than the max in the maxxes array
                # shove it into the last (uninhabited) column
                # example, region is 4, max is 3 and occupies position 0
                # place 4 in position 1, which is unoccupied
                # no need for a binary search
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                # we dont need to compute positions in the first run?
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                # where does region fit in maxxes?
                # e.g region is 4 and maxxes is 3,4,5
                # region fits in position 1 (zero indexing)
                arr = maxxes[: np.uintp(maxxes_counter)]
                posn = _numba_less_than_base(arr=arr, value=region)
                # look for the first column in the sorted_array
                # and insert region 4
                # and also insert the region position
                # into the postions_array
                sorted_array, positions_array, lengths, maxxes = (
                    _numba_sorted_array(
                        sorted_array=sorted_array,
                        positions_array=positions_array,
                        maxxes=maxxes,
                        lengths=lengths,
                        region=region,
                        posn=posn,
                        num=num,
                    )
                )
            r_count += 1
            posn_ = np.uintp(posn)
            # have we exceeded the size of this column?
            check1 = lengths[posn_] == (load_factor * 2)
            # no need to trim if loop == right_region.size
            check2 = r_count < len_right
            # do we need to trim and move data to other columns?
            if check1 & check2:
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
        end = start
        # now we do a binary search
        # for left region in right region
        # 1. find the position in maxxes
        # - this indicates which column in sorted_arrays contains our region
        # 2. search in the specific region for the positions
        # where left_region <= right_region
        l_region = left_region[_indexer]
        max_ind = maxxes_counter - 1
        # l_region has no match in this case:
        if l_region > maxxes[np.uintp(max_ind)]:
            booleans[_indexer] = False
            continue
        arr = maxxes[: np.uintp(maxxes_counter)]
        # position in maxxes
        posn = _numba_less_than_base(arr=arr, value=l_region)
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        len_arr_ = np.uintp(len_arr)
        arr = sorted_array[:len_arr_, posn_]
        # earliest position in sorted array
        _posn = _numba_less_than_base(arr=arr, value=l_region)
        difference = len_arr - _posn
        total += difference
        # step into the remaining columns
        # to get the remaining positions
        for ind in range(posn + 1, maxxes_counter):
            ind_ = np.uintp(ind)
            len_arr = lengths[ind_]
            difference += len_arr
            total += len_arr
        l_count += 1
    if total == 0:
        return None, None, None, None, None
    # second pass - fill arrays with indices
    zero_index = np.uintp(0)
    maxxes_counter = 1
    lengths[zero_index] = 1
    maxxes[zero_index] = base_region
    sorted_array[zero_index, zero_index] = base_region
    positions_array[zero_index, zero_index] = base_end
    end = base_end
    r_count = 0
    counts_array = np.zeros(length, dtype=np.intp)
    right_indices = np.empty(total, dtype=np.intp)
    begin = 0
    for indexer in range(length):
        _indexer = np.uintp(indexer)
        if not booleans[_indexer]:
            continue
        start = starts[_indexer]
        for num in range(start, end):
            _num = np.uintp(num)
            region = right_region[_num]
            if region < 0:
                r_count += 1
                continue
            posn = maxxes_counter - 1
            posn_ = np.uintp(posn)
            if region > maxxes[posn_]:
                len_arr = lengths[posn_]
                len_arr_ = np.uintp(len_arr)
                sorted_array[len_arr_, posn_] = region
                positions_array[len_arr_, posn_] = num
                maxxes[posn_] = region
                lengths[posn_] += 1
            else:
                arr = maxxes[: np.uintp(maxxes_counter)]
                posn = _numba_less_than_base(arr=arr, value=region)
                sorted_array, positions_array, lengths, maxxes = (
                    _numba_sorted_array(
                        sorted_array=sorted_array,
                        positions_array=positions_array,
                        maxxes=maxxes,
                        lengths=lengths,
                        region=region,
                        posn=posn,
                        num=num,
                    )
                )
            r_count += 1
            posn_ = np.uintp(posn)
            check1 = lengths[posn_] == (load_factor * 2)
            check2 = r_count < len_right
            if check1 & check2:
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
        end = start
        # now we do a binary search
        # for left region in right region
        counter = 0
        l_region = left_region[_indexer]
        arr = maxxes[: np.uintp(maxxes_counter)]
        posn = _numba_less_than_base(arr=arr, value=l_region)
        posn_ = np.uintp(posn)
        len_arr = lengths[posn_]
        len_arr_ = np.uintp(len_arr)
        arr = sorted_array[:len_arr_, posn_]
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
        counts_array[_indexer] = counter
    return (
        booleans,
        right_indices,
        counts_array,
        np.array([total], dtype=np.intp),
        np.array([l_count], dtype=np.intp),
    )


@njit(nogil=True)
def _numba_less_than_base(arr: np.ndarray, value: int):
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
        num: position of region in right_region array.
            Inserted into positions_array to keep
            in sync with the region the sorted_array.
    """
    # the sorted array implmentation is an adaptation
    # of grantjenks' sortedcontainers
    posn_ = np.uintp(posn)
    len_arr = lengths[posn_]
    len_arr_ = np.uintp(len_arr)
    # grab the specific column that the region falls into
    arr = sorted_array[:len_arr_, posn_]
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
    maxxes[posn_] = sorted_array[len_arr_, posn_]
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
        num: position of region in right_region array.
            Inserted into positions_array to keep
            in sync with the region the sorted_array.
        load_factor: optimal max length of each column in the sorted array.
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
    max_index = load_factor * 2
    max_index -= 1
    max_index = np.uintp(max_index)
    maxxes[forward] = sorted_array[max_index, current]
    lengths[forward] = load_factor
    sorted_array[:load_factor, forward] = sorted_array[load_factor:, current]
    positions_array[:load_factor, forward] = positions_array[
        load_factor:, current
    ]
    # update the length and maxxes arrays
    lengths[current] = load_factor
    maxxes[current] = sorted_array[np.uintp(load_factor - 1), current]
    maxxes_counter += 1
    return sorted_array, positions_array, lengths, maxxes, maxxes_counter
