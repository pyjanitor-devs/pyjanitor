# cythonised functions for conditional_join - aggregation

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


int_types = cython.fused_type(
    cython.schar,
    cython.short,
    cython.int,
    cython.long,
)

float_types = cython.fused_type(
    cython.float,
    cython.double,
)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges(
    booleans: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute index positions for min value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.bint = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        index = start
        base = 0
        for number in range(start, end):
            current = arr[number]
            if (base == 0) & (current != current):
                previous = current
                index = number
            elif base == 0:
                previous = current
                index = number
                base = 1
            elif current < previous:
                previous = current
                index = number
        indexes[begin] = index
        begin += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges(
    booleans: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute index positions for max value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.bint = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        index = start
        base = 0
        for number in range(start, end):
            current = arr[number]
            if (base == 0) & (current != current):
                previous = current
                index = number
            elif base == 0:
                previous = current
                index = number
                base = 1
            elif current > previous:
                previous = current
                index = number
        indexes[begin] = index
        begin += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges_matches(
    booleans: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute max indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    base: cython.bint = 0
    index: cython.Py_ssize_t = -1
    indexer: cython.Py_ssize_t = 0
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            continue
        start = starts[num]
        end = ends[num]
        index = start
        base = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            current = arr[number]
            if (base == 0) & (current != current):
                previous = current
                index = number
            elif base == 0:
                previous = current
                index = number
                base = 1
            elif current > previous:
                previous = current
                index = number
            begin += 1
        indexes[indexer] = index
        indexer += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges_matches(
    booleans: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute min indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    base: cython.bint = 0
    index: cython.Py_ssize_t = -1
    indexer: cython.Py_ssize_t = 0
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            continue
        start = starts[num]
        end = ends[num]
        index = start
        base = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            current = arr[number]
            if (base == 0) & (current != current):
                previous = current
                index = number
            elif base == 0:
                previous = current
                index = number
                base = 1
            elif current < previous:
                previous = current
                index = number
            begin += 1
        indexes[indexer] = index
        indexer += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_ints(
    booleans: cython.schar[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(total, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        for number in range(start, end):
            current = arr[number]
            final += current
        sums_array[begin] = final
        begin += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_floats(
    booleans: cython.schar[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(total, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        compensation = 0
        for number in range(start, end):
            current = arr[number]
            if current != current:
                continue
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
        sums_array[begin] = final
        begin += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_matches_ints(
    booleans: cython.schar[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(total, dtype=np.int64)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            current = arr[number]
            final += current
            begin += 1
        sums_array[indexer] = final
        indexer += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_matches_floats(
    booleans: cython.schar[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(total, dtype=np.float64)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        compensation = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            current = arr[number]
            if current != current:
                begin += 1
                continue
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
            begin += 1
        sums_array[indexer] = final
        indexer += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_positions_ints(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(total, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            final += current
        sums_array[begin] = final
        begin += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_positions_floats(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(total, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        compensation = 0
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            if current != current:
                continue
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            # gleaned from pandas' cython code
            # handles scenarios where compensation
            # devolves to NAN
            if compensation != compensation:
                compensation = 0
            final = increment
        sums_array[begin] = final
        begin += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_matches_positions_ints(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    index: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(total, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            current = arr[position]
            final += current
            begin += 1
        sums_array[index] = final
        index += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_matches_positions_floats(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    index: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(total, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        compensation = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            current = arr[position]
            if current != current:
                begin += 1
                continue
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
            begin += 1
        sums_array[index] = final
        index += 1
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_positions_ranges(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute index positions for min value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.bint = 0
    index: cython.Py_ssize_t = -1
    begin: cython.Py_ssize_t = 0
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = start
        base = 0
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            if (base == 0) & (current != current):
                previous = current
                index = position
            elif base == 0:
                previous = current
                index = position
                base = 1
            elif current < previous:
                previous = current
                index = position
        indexes[begin] = index
        begin += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_positions_ranges(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    total: cython.long,
):
    """
    compute index positions for max value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.bint = 0
    index: cython.Py_ssize_t = -1
    begin: cython.Py_ssize_t = 0
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = start
        base = 0
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            if (base == 0) & (current != current):
                previous = current
                index = position
            elif base == 0:
                previous = current
                index = position
                base = 1
            elif current > previous:
                previous = current
                index = position
        indexes[begin] = index
        begin += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges_positions_matches(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute max indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.bint = 0
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    l_index: cython.Py_ssize_t = 0
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = start
        base = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            current = arr[position]
            if (base == 0) & (current != current):
                previous = current
                index = position
            elif base == 0:
                previous = current
                index = position
                base = 1
            elif current > previous:
                previous = current
                index = position
            begin += 1
        indexes[l_index] = index
        l_index += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges_positions_matches(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    total: cython.long,
):
    """
    compute min indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.bint = 0
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    l_index: cython.Py_ssize_t = 0
    indexes: cython.long[::1] = np.empty(total, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = start
        base = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            current = arr[position]
            if (base == 0) & (current != current):
                previous = current
                index = position
            elif base == 0:
                previous = current
                index = position
                base = 1
            elif current < previous:
                previous = current
                index = position
            begin += 1
        indexes[l_index] = index
        l_index += 1
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_row_counts_from_ranges_positions(
    booleans: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
):
    """
    Expand sizes to length of booleans
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    count: cython.long = 0
    counts_array: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            counts_array[num] = 0
            continue
        indexer = indexers[num]
        if indexer == -1:
            counts_array[num] = 0
            booleans[num] = 0
            continue
        count = sizes[indexer]
        counts_array[num] = count
    return np.asarray(counts_array), np.asarray(booleans)
