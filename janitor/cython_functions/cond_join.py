# cythonised functions for conditional_join

import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_non_monotonic_dual_range_join_keep_all(
    left_index: cython.long[:],
    right_index: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    lookup: cython.long[:],
):
    """
    Build indices for non_monotonic conditions
    """
    # first pass
    # get total number of actual matches
    # this is where start, when resolved with lookup
    # is less than end
    size: cython.long = left_index.shape[0]
    num: cython.Py_ssize_t
    counter: cython.Py_ssize_t
    total: cython.long = 0
    r_value: cython.Py_ssize_t
    lookup_value: cython.long
    start: cython.long
    end: cython.long = right_index.shape[0]
    baseline: cython.long
    compare: cython.bint
    for num in range(size):
        start = starts[num]
        baseline = ends[num]
        for counter in range(start, end):
            r_value = right_index[counter]
            lookup_value = lookup[r_value]
            compare = lookup_value < baseline
            compare = int(compare)
            total += compare
    if not total:
        return None
    # second pass
    # build indices, based on total
    l_index = np.empty(total, dtype="int64")
    l_view: cython.long[:] = l_index
    r_index = np.empty(total, dtype="int64")
    r_view: cython.long[:] = r_index
    begin: cython.Py_ssize_t = 0
    l_value: cython.long
    for num in range(size):
        start = starts[num]
        baseline = ends[num]
        l_value = left_index[num]
        for counter in range(start, end):
            r_value = right_index[counter]
            lookup_value = lookup[r_value]
            compare = lookup_value < baseline
            if not compare:
                continue
            l_view[begin] = l_value
            r_view[begin] = r_value
            begin += 1
    return l_index, r_index


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_non_monotonic_dual_range_join_keep_first(
    left_index: cython.long[:],
    right_index: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    lookup: cython.long[:],
):
    """
    Build indices for non_monotonic conditions
    """
    size: cython.long = left_index.shape[0]
    num: cython.Py_ssize_t
    counter: cython.Py_ssize_t
    l_value: cython.long
    r_value: cython.Py_ssize_t
    min_value: cython.long
    baseline: cython.long
    start: cython.long
    l_index = np.empty(size, dtype="int64")
    l_view: cython.long[:] = l_index
    r_index = np.empty(size, dtype="int64")
    r_view: cython.long[:] = r_index
    begin: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = right_index.shape[0]

    for num in range(size):
        start = starts[num]
        baseline = ends[num]
        l_value = left_index[num]
        min_value = right_index[end - 1]
        for counter in range(start, end):
            r_value = right_index[counter]
            if not (lookup[r_value] < baseline):
                continue
            if r_value < min_value:
                min_value = r_value
        l_view[begin] = l_value
        r_view[begin] = min_value
        begin += 1
    if not begin:
        return None
    # trim if needed
    if begin < size:
        l_view = l_view[:begin]
        r_view = r_view[:begin]
    return l_index, r_index


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_non_monotonic_dual_range_join_keep_last(
    left_index: cython.long[:],
    right_index: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    lookup: cython.long[:],
):
    """
    Build indices for non_monotonic conditions
    """
    size: cython.long = left_index.shape[0]
    num: cython.Py_ssize_t
    counter: cython.Py_ssize_t
    l_value: cython.long
    r_value: cython.Py_ssize_t
    max_value: cython.long
    baseline: cython.long
    start: cython.long
    end: cython.long = right_index.shape[0]
    l_index = np.empty(size, dtype="int64")
    l_view: cython.long[:] = l_index
    r_index = np.empty(size, dtype="int64")
    r_view: cython.long[:] = r_index
    begin: cython.Py_ssize_t = 0

    for num in range(size):
        start = starts[num]
        baseline = ends[num]
        l_value = left_index[num]
        max_value = -1
        for counter in range(start, end):
            r_value = right_index[counter]
            if not (lookup[r_value] < baseline):
                continue
            if r_value > max_value:
                max_value = r_value
        l_view[begin] = l_value
        r_view[begin] = max_value
        begin += 1
    if not begin:
        return None
    # trim if needed
    if begin < size:
        l_view = l_view[:begin]
        r_view = r_view[:begin]
    return l_index, r_index
