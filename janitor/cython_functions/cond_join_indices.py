# cythonised functions to build indices for conditional_join

import cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_from_ranges_keep_all(
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[::1],
    right_indices: cython.long[::1],
    booleans: cython.schar[:],
):
    """
    Get indices if keep=='all'
    """
    ##### types declaration ######
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = starts.shape[0]
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    l_val: cython.Py_ssize_t
    r_val: cython.Py_ssize_t
    match_index: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    size: cython.Py_ssize_t = 0
    #############################

    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            match_index += size
            continue
        l_val = left_index[num]
        start = starts[num]
        end = ends[num]
        for number in range(start, end):
            if matches[match_index] == 0:
                match_index += 1
                continue
            r_val = right_index[number]
            left_indices[begin] = l_val
            right_indices[begin] = r_val
            begin += 1
            match_index += 1
    return np.asarray(left_indices), np.asarray(right_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_from_ranges_keep_first(
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[::1],
    right_indices: cython.long[::1],
    booleans: cython.schar[:],
):
    """
    Get indices if keep=='first'
    """
    ##### types declaration ######
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = starts.shape[0]
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    l_val: cython.Py_ssize_t
    r_val: cython.Py_ssize_t
    match_index: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    size: cython.Py_ssize_t = 0
    base: cython.long = -1
    #############################

    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            match_index += size
            continue
        start = starts[num]
        end = ends[num]
        base = -1
        for number in range(start, end):
            if matches[match_index] == 0:
                match_index += 1
                continue
            r_val = right_index[number]
            if (base < 0) | (base > r_val):
                base = r_val
            match_index += 1
        l_val = left_index[num]
        right_indices[begin] = base
        left_indices[begin] = l_val
        begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_from_ranges_keep_last(
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[::1],
    right_indices: cython.long[::1],
    booleans: cython.schar[:],
):
    """
    Get indices if keep=='last'
    """
    ##### types declaration ######
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = starts.shape[0]
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    l_val: cython.Py_ssize_t
    r_val: cython.Py_ssize_t
    match_index: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    size: cython.Py_ssize_t = 0
    base: cython.long = -1
    #############################

    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            match_index += size
            continue
        start = starts[num]
        end = ends[num]
        base = -1
        for number in range(start, end):
            if matches[match_index] == 0:
                match_index += 1
                continue
            r_val = right_index[number]
            if base < r_val:
                base = r_val
            match_index += 1
        l_val = left_index[num]
        right_indices[begin] = base
        left_indices[begin] = l_val
        begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_no_ranges_keep_first(
    left_index: cython.long[:],
    right_index: cython.long[:],
    indexers: cython.long[:],
    matches: cython.schar[:],
):
    """
    Compute indices
    """
    ##### types declaration ######
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    len_: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    l_index: cython.Py_ssize_t
    r_index: cython.Py_ssize_t
    r_val: cython.long
    begin_: cython.Py_ssize_t = 0
    booleans: cython.schar[::1] = np.zeros(len_, dtype=np.int8)
    #############################

    # compute total
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_index[num]
        if booleans[l_index] == 1:
            continue
        total += 1
        booleans[l_index] = 1
    # build indices
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    tracker: cython.long[::1] = np.empty(len_, dtype=np.intp)
    for num in range(len_):
        booleans[num] = 0
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_index[num]
        r_index = indexers[num]
        r_val = right_index[r_index]
        if booleans[l_index] == 0:
            tracker[l_index] = begin
            index_left[begin] = l_index
            index_right[begin] = r_val
            booleans[l_index] = 1
            begin += 1
        else:
            begin_ = tracker[l_index]
            if r_val < index_right[begin_]:
                index_right[begin_] = r_val
    return np.asarray(index_left), np.asarray(index_right)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_no_ranges_keep_last(
    left_index: cython.long[:],
    right_index: cython.long[:],
    indexers: cython.long[:],
    matches: cython.schar[:],
):
    """
    Compute indices
    """
    ##### types declaration ######
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    len_: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    l_index: cython.Py_ssize_t
    r_index: cython.Py_ssize_t
    r_val: cython.long
    begin_: cython.Py_ssize_t = 0
    booleans: cython.schar[::1] = np.zeros(len_, dtype=np.int8)
    #############################

    # compute total
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_index[num]
        if booleans[l_index] == 1:
            continue
        total += 1
        booleans[l_index] = 1
    # build indices
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    tracker: cython.long[::1] = np.empty(len_, dtype=np.intp)
    for num in range(len_):
        booleans[num] = 0
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_index[num]
        r_index = indexers[num]
        r_val = right_index[r_index]
        if booleans[l_index] == 0:
            tracker[l_index] = begin
            index_left[begin] = l_index
            index_right[begin] = r_val
            booleans[l_index] = 1
            begin += 1
        else:
            begin_ = tracker[l_index]
            if r_val > index_right[begin_]:
                index_right[begin_] = r_val
    return np.asarray(index_left), np.asarray(index_right)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_matches_positions_all(
    booleans: cython.schar[:],
    matches: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    index_right: cython.long[:],
    left_index: cython.long[::1],
    right_index: cython.long[::1],
):
    """
    Build indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    r_index: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    match_index: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            match_index += size
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        for number in range(start, end):
            if matches[match_index] == 0:
                match_index += 1
                continue
            position = positions[number]
            r_index = index_right[position]
            left_index[begin] = num
            right_index[begin] = r_index
            begin += 1
            match_index += 1
    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_matches_positions_first(
    booleans: cython.schar[:],
    matches: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    index_right: cython.long[:],
    left_index: cython.long[::1],
    right_index: cython.long[::1],
):
    """
    Build indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    r_index: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    match_index: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    base: cython.long = -1
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            match_index += size
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        base = -1
        for number in range(start, end):
            if matches[match_index] == 0:
                match_index += 1
                continue
            position = positions[number]
            if (base < 0) | (base > position):
                base = position
            match_index += 1
        left_index[begin] = num
        r_index = index_right[base]
        right_index[begin] = r_index
        begin += 1
    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_matches_positions_last(
    booleans: cython.schar[:],
    matches: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    index_right: cython.long[:],
    left_index: cython.long[::1],
    right_index: cython.long[::1],
):
    """
    Build indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    r_index: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    match_index: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    base: cython.long = -1
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            match_index += size
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        base = -1
        for number in range(start, end):
            if matches[match_index] == 0:
                match_index += 1
                continue
            position = positions[number]
            if (base < 0) | (base < position):
                base = position
            match_index += 1
        left_index[begin] = num
        r_index = index_right[base]
        right_index[begin] = r_index
        begin += 1

    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_ranges_positions_all(
    booleans: cython.schar[:],
    indexers: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    index_right: cython.long[:],
    left_index: cython.long[::1],
    right_index: cython.long[::1],
):
    """
    Build indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    r_index: cython.Py_ssize_t
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        for number in range(start, end):
            position = positions[number]
            r_index = index_right[position]
            left_index[begin] = num
            right_index[begin] = r_index
            begin += 1
    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_ranges_positions_first(
    booleans: cython.schar[:],
    indexers: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    index_right: cython.long[:],
    left_index: cython.long[::1],
    right_index: cython.long[::1],
):
    """
    Build indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    r_index: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.long = -1
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        base = -1
        for number in range(start, end):
            position = positions[number]
            if (base < 0) | (base > position):
                base = position
        left_index[begin] = num
        r_index = index_right[base]
        right_index[begin] = r_index
        begin += 1
    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_ranges_positions_last(
    booleans: cython.schar[:],
    indexers: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    index_right: cython.long[:],
    left_index: cython.long[::1],
    right_index: cython.long[::1],
):
    """
    Build indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    r_index: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    base: cython.long = -1
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        base = -1
        for number in range(start, end):
            position = positions[number]
            if (base < 0) | (base < position):
                base = position
        left_index[begin] = num
        r_index = index_right[base]
        right_index[begin] = r_index
        begin += 1

    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_fast_path_keep_all(
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[::1],
    right_indices: cython.long[::1],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Build indices for a single equi join,
    and a true range join (sorted on both right columns)
    and there are no other joins
    """
    ######## type declarations #######
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    length: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    l_value: cython.long
    r_value: cython.long
    ####################################

    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        l_value = left_index[num]
        for number in range(start, end):
            left_indices[begin] = l_value
            r_value = right_index[number]
            right_indices[begin] = r_value
            begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_fast_path_keep_first(
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[::1],
    right_indices: cython.long[::1],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Build indices for a single equi join,
    and a true range join (sorted on both right columns)
    and there are no other joins
    """
    ######## type declarations #######
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    length: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    base: cython.long = -1
    l_value: cython.long
    r_value: cython.long
    ####################################

    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        base = -1
        for number in range(start, end):
            r_value = right_index[number]
            if (base < 0) | (base > r_value):
                base = r_value
        l_value = left_index[num]
        left_indices[begin] = l_value
        right_indices[begin] = base
        begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_fast_path_keep_last(
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[::1],
    right_indices: cython.long[::1],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Build indices for a single equi join,
    and a true range join (sorted on both right columns)
    and there are no other joins
    """
    ######## type declarations #######
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    length: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    base: cython.long = -1
    l_value: cython.long
    r_value: cython.long
    ####################################

    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        base = -1
        for number in range(start, end):
            r_value = right_index[number]
            if base < r_value:
                base = r_value
        l_value = left_index[num]
        left_indices[begin] = l_value
        right_indices[begin] = base
        begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)
