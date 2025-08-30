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
        size = sizes[num]
        if booleans[num] == 0:
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
        size = sizes[num]
        if booleans[num] == 0:
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
        size = sizes[num]
        if booleans[num] == 0:
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
    left_index: cython.long[:],
    right_index: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get positive matches from comparison
    """
    #### type declarations #####
    lengths: cython.Py_ssize_t = left_index.shape[0]
    num: cython.Py_ssize_t = 0
    boolean: cython.bint = 0
    l_index: cython.Py_ssize_t = 0
    r_index: cython.Py_ssize_t = 0
    l_val: scalar_types
    r_val: scalar_types
    total: cython.long = 0
    #######################
    for num in range(lengths):
        if booleans[num] == 0:
            continue
        l_index = left_index[num]
        r_index = right_index[num]
        l_val = left[l_index]
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
    left_index: cython.long[:],
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
    lengths: cython.Py_ssize_t = left_index.shape[0]
    num: cython.Py_ssize_t = 0
    boolean: cython.bint = 0
    l_index: cython.Py_ssize_t = 0
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
        l_index = left_index[num]
        r_index = right_index[num]
        l_bool = left_booleans[l_index]
        r_bool = right_booleans[r_index]
        if (l_bool == 1) | (r_bool == 1):
            boolean = 1
        else:
            l_val = left[l_index]
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
    left_index: cython.long[:],
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
    lengths: cython.Py_ssize_t = left_index.shape[0]
    num: cython.Py_ssize_t = 0
    boolean: cython.bint = 0
    l_index: cython.Py_ssize_t = 0
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
        l_index = left_index[num]
        r_index = right_index[num]
        l_bool = left_booleans[l_index]
        r_bool = right_booleans[r_index]
        # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
        if (l_bool == 1) | (r_bool == 1):
            boolean = 0
        else:
            l_val = left[l_index]
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
def get_row_count_no_ranges(
    counts_array: cython.long[::1],
    left_index: cython.long[:],
    left_indices: cython.long[:],
    matches: cython.schar[:],
):
    """
    Compute row count
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = left_indices.shape[0]
    num: cython.Py_ssize_t = 0
    l_val: cython.Py_ssize_t
    l_index: cython.Py_ssize_t
    ##########################################
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_indices[num]
        l_val = left_index[l_index]
        counts_array[l_val] += 1
    return np.asarray(counts_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_no_ranges_keep_all(
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[:],
    right_indices: cython.long[:],
    matches: cython.schar[:],
):
    """
    Compute indices
    """
    ##### types declaration ######
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    begin: cython.Py_ssize_t = 0
    l_index: cython.Py_ssize_t
    r_index: cython.Py_ssize_t
    l_val: cython.Py_ssize_t
    r_val: cython.Py_ssize_t
    #############################

    # compute length of final indices
    for num in range(length):
        if matches[num] == 0:
            continue
        total += 1
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    for num in range(length):
        if begin == total:
            break
        if matches[num] == 0:
            continue
        l_index = left_indices[num]
        l_val = left_index[l_index]
        r_index = right_indices[num]
        r_val = right_index[r_index]
        index_left[begin] = l_val
        index_right[begin] = r_val
        begin += 1
    return np.asarray(index_left), np.asarray(index_right)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_no_ranges_keep_first(
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[:],
    right_indices: cython.long[:],
    matches: cython.schar[:],
):
    """
    Compute indices
    """
    ##### types declaration ######
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    begin: cython.Py_ssize_t = 0
    l_index: cython.Py_ssize_t
    r_index: cython.Py_ssize_t
    l_val: cython.Py_ssize_t
    r_val: cython.Py_ssize_t
    base_index: cython.Py_ssize_t = -1
    begin_: cython.Py_ssize_t = 0
    current: cython.Py_ssize_t
    #############################

    # compute total
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_indices[num]
        if base_index != l_index:
            total += 1
            base_index = l_index
    # build indices
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    base_index = -1
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_indices[num]
        l_val = left_index[l_index]
        r_index = right_indices[num]
        r_val = right_index[r_index]
        if base_index != l_index:
            index_left[begin] = l_val
            index_right[begin] = r_val
            base_index = l_index
            begin += 1
        else:
            begin_ = begin - 1
            current = index_right[begin_]
            if current > r_val:
                index_right[begin_] = r_val
    return np.asarray(index_left), np.asarray(index_right)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_no_ranges_keep_last(
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_indices: cython.long[:],
    right_indices: cython.long[:],
    matches: cython.schar[:],
):
    """
    Compute indices
    """
    ##### types declaration ######
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    begin: cython.Py_ssize_t = 0
    l_index: cython.Py_ssize_t
    r_index: cython.Py_ssize_t
    l_val: cython.Py_ssize_t
    r_val: cython.Py_ssize_t
    base_index: cython.Py_ssize_t = -1
    begin_: cython.Py_ssize_t = 0
    current: cython.Py_ssize_t
    #############################

    # compute total
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_indices[num]
        if base_index != l_index:
            total += 1
            base_index = l_index
    # build indices
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    base_index = -1
    for num in range(length):
        if matches[num] == 0:
            continue
        l_index = left_indices[num]
        l_val = left_index[l_index]
        r_index = right_indices[num]
        r_val = right_index[r_index]
        if base_index != l_index:
            index_left[begin] = l_val
            index_right[begin] = r_val
            base_index = l_index
            begin += 1
        else:
            begin_ = begin - 1
            current = index_right[begin_]
            if current < r_val:
                index_right[begin_] = r_val
    return np.asarray(index_left), np.asarray(index_right)


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
            if (base < 0) or (base > r_val):
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
def build_indices_equi_range_join_only_fast_path_keep_all(
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
def build_indices_equi_range_join_only_fast_path_keep_first(
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
            if (base < 0) or (base > r_value):
                base = r_value
        l_value = left_index[num]
        left_indices[begin] = l_value
        right_indices[begin] = base
        begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_equi_range_join_only_fast_path_keep_last(
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
