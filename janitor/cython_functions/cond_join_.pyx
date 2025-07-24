
import numpy as np
cimport cython

ctypedef fused my_type:
    int
    double
    long long


cdef my_type compare_values(my_type left_value, my_type right_value, int op):
    if op == 0:
        return left_value > right_value
    if op == 1:
        return left_value >= right_value
    if op == 2:
        return left_value < right_value
    if op == 3:
        return left_value <= right_value
    if op == 4:
        return left_value == right_value



@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_non_monotonic_range_join_keep_all(
    long[:] left_index,
    long[:] right_index,
    long[:] starts,
    long[:] ends,
    long[:] lookup,
    conditions:list,
):
    """
    Build indices for non_monotonic conditions
    """
    # first pass
    # get total number of actual matches
    # this is where start, when resolved with lookup
    # is less than end
    cdef long size = left_index.shape[0]
    cdef Py_ssize_t num, number, counter, l_indexer, r_indexer
    cdef long total = 0
    cdef long lookup_index, baseline_index
    cdef my_type l_value, r_value
    cdef my_type[:] left_array, right_array
    cdef int op
    cdef long start
    cdef long end = right_index.shape[0]
    cdef bint compare
    for num in range(size):
        start = starts[num]
        baseline_index = ends[num]
        l_indexer = left_index[num]
        for counter in range(start, end):
            r_indexer = right_index[counter]
            lookup_index = lookup[r_indexer]
            if not (lookup_index < baseline_index):
                continue
            for item in conditions:
                cdef my_type[:] l_view = item[0]
                pass
                # l_view: types[:] = left_array
                # r_view: types[:] = right_array
                # l_value = l_view[l_indexer]
                # r_value = r_view[r_indexer]
                # print(l_value, r_value)
            # for left_array, right_array, op in conditions:
            #     l_value = left_array[l_indexer]
            #     r_value = right_array[r_indexer]
            #     compare = compare_values(
            #         left_value=l_value, right_value=r_value, op=op
            #     )
            #     print(l_value, r_value, op, compare)
            #     pass
            total += 1
    if not total:
        return None
    # # second pass
    # # build indices, based on total
    # l_index = np.empty(total, dtype="int64")
    # l_view = l_index
    # r_index = np.empty(total, dtype="int64")
    # r_view = r_index
    # begin: cython.Py_ssize_t = 0
    # l_value: cython.long
    # for num in range(size):
    #     start = starts[num]
    #     baseline = ends[num]
    #     l_value = left_index[num]
    #     for counter in range(start, end):
    #         r_value = right_index[counter]
    #         lookup_value = lookup[r_value]
    #         compare = lookup_value < baseline
    #         if not compare:
    #             continue
    #         l_view[begin] = l_value
    #         r_view[begin] = r_value
    #         begin += 1
    # return l_index, r_index
