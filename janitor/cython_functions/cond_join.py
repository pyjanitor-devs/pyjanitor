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


@cython.exceptval(check=False)
@cython.cfunc
def compare_values(
    left: scalar_types, right: scalar_types, op: cython.int
) -> cython.bint:
    compare: cython.bint = 0
    if op == 0:
        compare = left > right
    elif op == 1:
        compare = left >= right
    elif op == 2:
        compare = left < right
    elif op == 3:
        compare = left <= right
    elif op == 4:
        compare = left == right
    else:
        compare = left != right
    return compare


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches(
    sizes: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get positive matches from comparison
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    # how many left values have actual matches in right
    l_counts: cython.long = 0
    boolean: cython.bint = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    size: cython.long = 0
    count: cython.long = 0
    # total count of actual positives
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            counts_array[num] = 0
            continue
        l_val = left[num]
        start = starts[num]
        end = ends[num]
        count = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            r_val = right[number]
            boolean = compare_values(left=l_val, right=r_val, op=op)
            matches[begin] = boolean
            begin += 1
            total += boolean
            count += boolean
        counts_array[num] = count
        boolean = count > 0
        booleans[num] = boolean
        l_counts += boolean

    return (
        np.asarray(matches),
        np.asarray(booleans),
        np.asarray(counts_array),
        total,
        l_counts,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_ne(
    sizes: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Special situation to get positive matches from comparison,
    where `op==!=`
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    l_counts: cython.long = 0
    boolean: cython.bint = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    l_bool: cython.schar
    r_bool: cython.schar
    size: cython.long = 0
    count: cython.long = 0
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            counts_array[num] = 0
            continue
        l_val = left[num]
        start = starts[num]
        end = ends[num]
        count = 0
        l_bool = left_booleans[num]
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            r_bool = right_booleans[number]
            if (l_bool == 1) | (r_bool == 1):
                boolean = 1
            else:
                r_val = right[number]
                boolean = compare_values(left=l_val, right=r_val, op=op)
            matches[begin] = boolean
            begin += 1
            total += boolean
            count += boolean
        counts_array[num] = count
        boolean = count > 0
        booleans[num] = boolean
        l_counts += boolean

    return (
        np.asarray(matches),
        np.asarray(booleans),
        np.asarray(counts_array),
        total,
        l_counts,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_ne_pandas_array(
    sizes: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Special situation to get positive matches from comparison,
    where `op==!=` - specifically for pandas extension arrays (pd.NA)
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    l_counts: cython.long = 0
    boolean: cython.bint = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    l_bool: cython.schar
    r_bool: cython.schar
    size: cython.long = 0
    count: cython.long = 0
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            counts_array[num] = 0
            continue
        l_val = left[num]
        start = starts[num]
        end = ends[num]
        count = 0
        l_bool = left_booleans[num]
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            r_bool = right_booleans[number]
            # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
            if (l_bool == 1) | (r_bool == 1):
                boolean = 0
            else:
                r_val = right[number]
                boolean = compare_values(left=l_val, right=r_val, op=op)
            matches[begin] = boolean
            begin += 1
            total += boolean
            count += boolean
        counts_array[num] = count
        boolean = count > 0
        booleans[num] = boolean
        l_counts += boolean

    return (
        np.asarray(matches),
        np.asarray(booleans),
        np.asarray(counts_array),
        total,
        l_counts,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_no_ranges(
    op: cython.int,
    left: scalar_types[:],
    right: scalar_types[:],
    right_index: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get positive matches from comparison
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    boolean: cython.bint = 0
    r_index: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            continue
        r_index = right_index[num]
        l_val = left[num]
        r_val = right[r_index]
        boolean = compare_values(left=l_val, right=r_val, op=op)
        booleans[num] = boolean
        total += boolean
    return (
        np.asarray(booleans),
        total,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_no_ranges_ne(
    op: cython.int,
    left: scalar_types[:],
    right: scalar_types[:],
    right_index: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Get positive matches from comparison;
    Applies if op == '!='
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    boolean: cython.bint = 0
    r_index: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    l_bool: cython.schar
    r_bool: cython.schar
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            continue
        r_index = right_index[num]
        l_bool = left_booleans[num]
        r_bool = right_booleans[r_index]
        if (l_bool == 1) | (r_bool == 1):
            boolean = 1
        else:
            l_val = left[num]
            r_val = right[r_index]
            boolean = compare_values(left=l_val, right=r_val, op=op)
        booleans[num] = boolean
        total += boolean

    return (
        np.asarray(booleans),
        total,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_no_ranges_ne_pandas_array(
    op: cython.int,
    left: scalar_types[:],
    right: scalar_types[:],
    right_index: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Get positive matches from comparison;
    Applies if op == '!=' and pandas array(pd.NA)
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    boolean: cython.bint = 0
    r_index: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    l_bool: cython.schar
    r_bool: cython.schar
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            continue
        r_index = right_index[num]
        l_bool = left_booleans[num]
        r_bool = right_booleans[r_index]
        # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
        if (l_bool == 1) | (r_bool == 1):
            boolean = 0
        else:
            l_val = left[num]
            r_val = right[r_index]
            boolean = compare_values(left=l_val, right=r_val, op=op)
        booleans[num] = boolean
        total += boolean

    return (
        np.asarray(booleans),
        total,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_less_than_strict(
    left: scalar_types[:],
    right: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `>=` condition
    """
    ######## type declarations #######
    num: cython.Py_ssize_t
    mid_idx: cython.Py_ssize_t
    length: cython.Py_ssize_t = left.shape[0]
    end: cython.Py_ssize_t
    size: cython.long = 0
    min_idx: cython.Py_ssize_t
    max_idx: cython.Py_ssize_t
    l_value: scalar_types
    current_value: scalar_types
    match: cython.long = 0
    total: cython.long = 0
    ####################################

    for num in range(length):
        if booleans[num] == 0:
            sizes[num] = 0
            continue
        end = ends[num]
        l_value = left[num]
        # adapted from numba/np/array_math.py
        min_idx = starts[num]
        max_idx = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            current_value = right[mid_idx]
            if current_value <= l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        if min_idx == end:
            booleans[num] = 0
            sizes[num] = 0
            continue
        current_value = right[min_idx]
        if current_value == l_value:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        starts[num] = min_idx
        size = end - min_idx
        sizes[num] = size
        total += size
        match += 1
    return (
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(booleans),
        np.asarray(sizes),
        total,
        match,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_less_than(
    left: scalar_types[:],
    right: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `>=` condition
    """
    ######## type declarations #######
    num: cython.Py_ssize_t
    mid_idx: cython.Py_ssize_t
    length: cython.Py_ssize_t = left.shape[0]
    end: cython.Py_ssize_t
    size: cython.long = 0
    min_idx: cython.Py_ssize_t
    max_idx: cython.Py_ssize_t
    l_value: scalar_types
    current_value: scalar_types
    match: cython.long = 0
    total: cython.long = 0
    ####################################

    for num in range(length):
        if booleans[num] == 0:
            sizes[num] = 0
            continue
        end = ends[num]
        l_value = left[num]
        # adapted from numba/np/array_math.py
        min_idx = starts[num]
        max_idx = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            current_value = right[mid_idx]
            if current_value < l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        if min_idx == end:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        starts[num] = min_idx
        size = end - min_idx
        sizes[num] = size
        total += size
        match += 1
    return (
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(booleans),
        np.asarray(sizes),
        total,
        match,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_greater_than_strict(
    left: scalar_types[:],
    right: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `>=` condition
    """
    ######## type declarations #######
    num: cython.Py_ssize_t
    mid_idx: cython.Py_ssize_t
    length: cython.Py_ssize_t = left.shape[0]
    start: cython.Py_ssize_t = 0
    size: cython.long = 0
    min_idx: cython.Py_ssize_t
    max_idx: cython.Py_ssize_t
    l_value: scalar_types
    current_value: scalar_types
    match: cython.long = 0
    total: cython.long = 0
    ####################################

    for num in range(length):
        if booleans[num] == 0:
            sizes[num] = 0
            continue
        start = starts[num]
        l_value = left[num]
        # adapted from numba/np/array_math.py
        min_idx = starts[num]
        max_idx = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            current_value = right[mid_idx]
            if current_value >= l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        if min_idx == start:
            booleans[num] = 0
            sizes[num] = 0
            continue
        mid_idx = min_idx - 1
        current_value = right[mid_idx]
        if current_value == l_value:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        ends[num] = min_idx
        size = min_idx - start
        sizes[num] = size
        total += size
        match += 1
    return (
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(booleans),
        np.asarray(sizes),
        total,
        match,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_greater_than(
    left: scalar_types[:],
    right: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `>=` condition
    """
    ######## type declarations #######
    num: cython.Py_ssize_t
    mid_idx: cython.Py_ssize_t
    length: cython.Py_ssize_t = left.shape[0]
    start: cython.Py_ssize_t = 0
    size: cython.long = 0
    min_idx: cython.Py_ssize_t
    max_idx: cython.Py_ssize_t
    l_value: scalar_types
    current_value: scalar_types
    match: cython.long = 0
    total: cython.long = 0
    ####################################

    for num in range(length):
        if booleans[num] == 0:
            sizes[num] = 0
            continue
        start = starts[num]
        l_value = left[num]
        # adapted from numba/np/array_math.py
        min_idx = starts[num]
        max_idx = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx = min_idx + ((max_idx - min_idx) >> 1)
            current_value = right[mid_idx]
            if current_value > l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        if min_idx == start:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        ends[num] = min_idx
        size = min_idx - start
        sizes[num] = size
        total += size
        match += 1
    return (
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(booleans),
        np.asarray(sizes),
        total,
        match,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def check_monotonicity_per_range(
    starts: cython.long[:],
    ends: cython.long[:],
    arr: scalar_types[:],
    booleans: cython.schar[:],
):
    """
    set true/false flag per window for monotonic_increasing
    """
    #### type declarations #####
    length: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    lenght: cython.Py_ssize_t = arr.shape[0]
    matches: cython.schar[::1] = np.empty(length, dtype=np.int8)
    bools: cython.schar[::1] = np.zeros(lenght, dtype=np.int8)
    current: scalar_types
    next_start: cython.Py_ssize_t
    tracker: cython.schar
    val: scalar_types
    all_true: cython.schar = 1
    #######################

    for num in range(length):
        if booleans[num] == 0:
            continue
        start = starts[num]
        end = ends[num]
        if bools[start] == 1:
            matches[num] = 1
            continue
        current = arr[start]
        next_start = start + 1
        bools[start] = 1
        tracker = 1
        for number in range(next_start, end):
            val = arr[number]
            if current > val:
                tracker = 0
                all_true = 0
                break
            current = val
        matches[num] = tracker
    return all_true


def reorder_positions(
    len_uniques: cython.Py_ssize_t,
    positions: cython.Py_ssize_t[:],
) -> tuple:
    """
    reorder positions based on uniques.
    variant of counting sort
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = positions.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    val: cython.Py_ssize_t = 0
    value: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    counts_array: cython.Py_ssize_t[::1] = np.zeros(len_uniques, dtype=np.intp)
    starts_: cython.Py_ssize_t[::1] = np.empty(len_uniques, dtype=np.intp)
    starts: cython.Py_ssize_t[::1] = np.empty(len_uniques, dtype=np.intp)
    ends: cython.Py_ssize_t[::1] = np.empty(len_uniques, dtype=np.intp)
    ordered_positions: cython.Py_ssize_t[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    # get counts
    for num in range(length):
        val = positions[num]
        counts_array[val] += 1
    starts_[0] = 0
    starts[0] = 0
    val = 0
    # build cumsum of starting indices
    # per unique index
    for num in range(1, len_uniques):
        number = num - 1
        value = counts_array[number]
        val += value
        starts_[num] = val
        starts[num] = val
    # reorder positions
    # all 0s followed by 1s ...
    # here the actual positions will be captured
    for num in range(length):
        val = positions[num]
        begin = starts_[val]
        ordered_positions[begin] = num
        starts_[val] += 1
    # build ends
    # so we have start, end pair
    for num in range(len_uniques):
        val = counts_array[num]
        value = starts[num]
        value += val
        ends[num] = value
    return (
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(counts_array),
        np.asarray(ordered_positions),
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_ranges_positions(
    sizes: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
):
    """
    Get positive matches from comparison
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    # how many left values have actual matches in right
    l_counts: cython.long = 0
    boolean: cython.bint = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    size: cython.long = 0
    count: cython.long = 0
    # total count of actual positives
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            counts_array[num] = 0
            continue
        l_val = left[num]
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        count = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            r_val = right[position]
            boolean = compare_values(left=l_val, right=r_val, op=op)
            matches[begin] = boolean
            begin += 1
            total += boolean
            count += boolean
        counts_array[num] = count
        boolean = count > 0
        booleans[num] = boolean
        l_counts += boolean
    return (
        np.asarray(matches),
        np.asarray(booleans),
        np.asarray(counts_array),
        total,
        l_counts,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_ranges_positions_ne(
    sizes: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
):
    """
    Special situation to get positive matches from comparison,
    where `op==!=`
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    l_counts: cython.long = 0
    boolean: cython.bint = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    l_bool: cython.schar
    r_bool: cython.schar
    size: cython.long = 0
    count: cython.long = 0
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            counts_array[num] = 0
            continue
        l_val = left[num]
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        l_bool = left_booleans[num]
        count = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            r_bool = right_booleans[position]
            if (l_bool == 1) | (r_bool == 1):
                boolean = 1
            else:
                r_val = right[position]
                boolean = compare_values(left=l_val, right=r_val, op=op)
            matches[begin] = boolean
            begin += 1
            total += boolean
            count += boolean
        counts_array[num] = count
        boolean = count > 0
        booleans[num] = boolean
        l_counts += boolean

    return (
        np.asarray(matches),
        np.asarray(booleans),
        np.asarray(counts_array),
        total,
        l_counts,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_ranges_positions_ne_pandas_array(
    sizes: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
):
    """
    Special situation to get positive matches from comparison,
    where `op==!=` - specifically for pandas extension arrays (pd.NA)
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    l_counts: cython.long = 0
    boolean: cython.bint = 0
    start: cython.Py_ssize_t = 0
    end: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    l_bool: cython.schar
    r_bool: cython.schar
    size: cython.long = 0
    count: cython.long = 0
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            counts_array[num] = 0
            continue
        l_val = left[num]
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        l_bool = left_booleans[num]
        count = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            r_bool = right_booleans[position]
            # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
            if (l_bool == 1) | (r_bool == 1):
                boolean = 0
            else:
                r_val = right[position]
                boolean = compare_values(left=l_val, right=r_val, op=op)
            matches[begin] = boolean
            begin += 1
            total += boolean
            count += boolean
        counts_array[num] = count
        boolean = count > 0
        booleans[num] = boolean
        l_counts += boolean

    return (
        np.asarray(matches),
        np.asarray(booleans),
        np.asarray(counts_array),
        total,
        l_counts,
    )
