# cythonised functions for conditional_join

import cython
import numpy as np

scalar_types = cython.fused_type(
    cython.schar,
    cython.short,
    cython.int,
    cython.long,
    cython.float,
    cython.double,
)


@cython.cfunc
def compare_values(
    left_value: scalar_types, right_value: scalar_types, op: cython.int
) -> scalar_types:
    compare: scalar_types = 0
    if op == 0:
        compare = left_value > right_value
    elif op == 1:
        compare = left_value >= right_value
    elif op == 2:
        compare = left_value < right_value
    elif op == 3:
        compare = left_value <= right_value
    elif op == 4:
        compare = left_value == right_value
    return compare


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches(
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get positive matches from comparison
    """
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    begin: cython.Py_ssize_t = 0
    any_match: cython.bint = 0
    bools_all: cython.bint = 1
    for num in range(counts):
        check = booleans[num]
        if check == 0:
            bools_all = 0
            continue
        left_value: scalar_types = left_array[num]
        start: cython.long = starts[num]
        end: cython.long = ends[num]
        count: cython.long = 0
        for number in range(start, end):
            right_value: scalar_types = right_array[number]
            compare: scalar_types = compare_values(
                left_value=left_value, right_value=right_value, op=op
            )
            boolean: cython.schar = cython.cast(cython.schar, compare)
            current_bool: cython.schar = matches[begin]
            boolean = current_bool & boolean
            matches[begin] = boolean
            begin += 1
            count += boolean
        update: cython.bint = count > 0
        booleans[num] = update
        counts_array[num] = count
        any_match = any_match | update
    return (
        np.asarray(matches),
        np.asarray(booleans),
        np.asarray(counts_array),
        any_match,
        bools_all,
    )
