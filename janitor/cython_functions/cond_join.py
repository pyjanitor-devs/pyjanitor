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


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges(
    booleans: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute index positions for min value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current < previous:
                previous = current
                index = number
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute index positions for min value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    mask: cython.schar
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            mask = nulls_mask[number]
            if mask == 1:
                continue
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current < previous:
                previous = current
                index = number
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges(
    booleans: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute index positions for max value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current > previous:
                previous = current
                index = number
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute index positions for max value
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    mask: cython.schar
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            mask = nulls_mask[number]
            if mask == 1:
                continue
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current > previous:
                previous = current
                index = number
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges_matches_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute max indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            mask = nulls_mask[number]
            if mask == 1:
                begin += 1
                continue
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current > previous:
                previous = current
                index = number
            begin += 1
        indexes[num] = index
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
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current > previous:
                previous = current
                index = number
            begin += 1
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges_matches_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute min indices
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            mask = nulls_mask[number]
            if mask == 1:
                begin += 1
                continue
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current < previous:
                previous = current
                index = number
            begin += 1
        indexes[num] = index
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
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            indexes[num] = -1
            continue
        start = starts[num]
        end = ends[num]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            current = arr[number]
            if index == -1:
                previous = current
                index = number
            elif current < previous:
                previous = current
                index = number
            begin += 1
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_counts_from_ranges_matches_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute counts
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    count: cython.long = 0
    counts_array: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            counts_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        count = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            mask = nulls_mask[number]
            if mask == 1:
                begin += 1
                continue
            count += 1
            begin += 1
        counts_array[num] = count
    return np.asarray(counts_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_counts_from_ranges_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute counts
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    count: cython.long = 0
    counts_array: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            counts_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        count = 0
        for number in range(start, end):
            mask = nulls_mask[number]
            if mask == 1:
                continue
            count += 1
        counts_array[num] = count
    return np.asarray(counts_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_matches_ints_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            sums_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            mask = nulls_mask[number]
            if mask == 1:
                begin += 1
                continue
            current = arr[number]
            final += current
            begin += 1
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_matches_floats_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            sums_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        compensation = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            mask = nulls_mask[number]
            if mask == 1:
                begin += 1
                continue
            current = arr[number]
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
            begin += 1
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_ints_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        for number in range(start, end):
            mask = nulls_mask[number]
            if mask == 1:
                continue
            current = arr[number]
            final += current
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_floats_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        compensation = 0
        for number in range(start, end):
            mask = nulls_mask[number]
            if mask == 1:
                continue
            current = arr[number]
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_ints(
    booleans: cython.schar[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        for number in range(start, end):
            current = arr[number]
            final += current
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_floats(
    booleans: cython.schar[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        start = starts[num]
        end = ends[num]
        final = 0
        compensation = 0
        for number in range(start, end):
            current = arr[number]
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
        sums_array[num] = final
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
    size: cython.long = 0
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            sums_array[num] = 0
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
        sums_array[num] = final
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
    size: cython.long = 0
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        size = sizes[num]
        if booleans[num] == 0:
            begin += size
            sums_array[num] = 0
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
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
            begin += 1
        sums_array[num] = final
    return np.asarray(sums_array)


def reorder_positions(
    len_uniques: cython.Py_ssize_t, positions: cython.Py_ssize_t[:]
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
def get_sums_from_ranges_matches_ints_positions_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    indexers: cython.long[:],
    positions: cython.long[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            sums_array[num] = 0
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
            mask = nulls_mask[position]
            if mask == 1:
                begin += 1
                continue
            current = arr[number]
            final += current
            begin += 1
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_matches_floats_positions_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            sums_array[num] = 0
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
            mask = nulls_mask[position]
            if mask == 1:
                begin += 1
                continue
            current = arr[position]
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
            begin += 1
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_ints_positions_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: int_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
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
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        for number in range(start, end):
            position = positions[number]
            mask = nulls_mask[position]
            if mask == 1:
                continue
            current = arr[position]
            final += current
        sums_array[num] = final
    return np.asarray(sums_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_sums_from_ranges_floats_positions_nulls(
    booleans: cython.schar[:],
    nulls_mask: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    arr: float_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute sums for floats using kahan summation
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        compensation = 0
        for number in range(start, end):
            position = positions[number]
            mask = nulls_mask[position]
            if mask == 1:
                continue
            current = arr[position]
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
        sums_array[num] = final
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
):
    """
    compute sums for integers
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            final += current
        sums_array[num] = final
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
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            sums_array[num] = 0
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        final = 0
        compensation = 0
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            # gleaned from pandas' cython code
            # handles scenarios where compensation
            # devolves to NAN
            if compensation != compensation:
                compensation = 0
            final = increment
        sums_array[num] = final
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
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    final: cython.long = 0
    current: int_types
    sums_array: cython.long[::1] = np.empty(length, dtype=np.int64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            sums_array[num] = 0
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
        sums_array[num] = final
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
):
    """
    compute sums for floats using kahan summation
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
    size: cython.long = 0
    final: cython.double = 0
    current: float_types
    compensation: float_types = 0
    increment: float_types
    difference: float_types
    sums_array: cython.double[::1] = np.empty(length, dtype=np.float64)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            sums_array[num] = 0
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
            difference = current - compensation
            increment = final + difference
            compensation = (increment - final) - difference
            if compensation != compensation:
                compensation = 0
            final = increment
            begin += 1
        sums_array[num] = final
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
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current < previous:
                previous = current
                index = position
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges_positions_nulls(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
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
    mask: cython.schar
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            position = positions[number]
            mask = nulls_mask[position]
            if mask == 1:
                continue
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current < previous:
                previous = current
                index = position
        indexes[num] = index
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
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            position = positions[number]
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current > previous:
                previous = current
                index = position
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges_positions_nulls(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
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
    mask: cython.schar
    current: scalar_types
    previous: scalar_types
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            position = positions[number]
            mask = nulls_mask[position]
            if mask == 1:
                continue
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current > previous:
                previous = current
                index = position
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_max_from_ranges_matches_positions_nulls(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
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
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            mask = nulls_mask[position]
            if mask == 1:
                begin += 1
                continue
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current > previous:
                previous = current
                index = position
            begin += 1
        indexes[num] = index
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
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current > previous:
                previous = current
                index = position
            begin += 1
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_min_from_ranges_matches_positions_nulls(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    nulls_mask: cython.schar[:],
    arr: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
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
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            mask = nulls_mask[position]
            if mask == 1:
                begin += 1
                continue
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current < previous:
                previous = current
                index = position
            begin += 1
        indexes[num] = index
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
    current: scalar_types
    previous: scalar_types
    size: cython.long = 0
    index: cython.Py_ssize_t = -1
    indexes: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            indexes[num] = -1
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        index = -1
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            current = arr[position]
            if index == -1:
                previous = current
                index = position
            elif current < previous:
                previous = current
                index = position
            begin += 1
        indexes[num] = index
    return np.asarray(indexes)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_counts_from_ranges_matches_positions_nulls(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    nulls_mask: cython.schar[:],
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
):
    """
    compute counts
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    size: cython.long = 0
    count: cython.long = 0
    counts_array: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            size = sizes[num]
            begin += size
            counts_array[num] = 0
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        count = 0
        for number in range(start, end):
            if matches[begin] == 0:
                begin += 1
                continue
            position = positions[number]
            mask = nulls_mask[position]
            mask = not mask
            count += mask
            begin += 1
        counts_array[num] = count
    return np.asarray(counts_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_counts_from_ranges_positions_nulls(
    booleans: cython.schar[:],
    positions: cython.long[:],
    indexers: cython.long[:],
    nulls_mask: cython.schar[:],
    starts: cython.long[:],
    ends: cython.long[:],
):
    """
    compute counts
    """
    ###### types declaration  #################
    length: cython.Py_ssize_t = booleans.shape[0]
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    indexer: cython.Py_ssize_t = 0
    position: cython.Py_ssize_t = 0
    mask: cython.schar
    start: cython.Py_ssize_t
    end: cython.Py_ssize_t
    count: cython.long = 0
    counts_array: cython.long[::1] = np.empty(length, dtype=np.intp)
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            counts_array[num] = 0
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        count = 0
        for number in range(start, end):
            position = positions[number]
            mask = nulls_mask[position]
            mask = not mask
            count += mask
        counts_array[num] = count
    return np.asarray(counts_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_row_counts_from_ranges_positions(
    booleans: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
):
    """
    compute sizes
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


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_from_ranges_positions(
    booleans: cython.schar[:],
    indexers: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
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
    ##########################################
    for num in range(length):
        if booleans[num] == 0:
            continue
        indexer = indexers[num]
        start = starts[indexer]
        end = ends[indexer]
        for number in range(start, end):
            position = positions[number]
            left_index[begin] = num
            right_index[begin] = position
            begin += 1
    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_from_ranges_matches_positions_keep_all(
    booleans: cython.schar[:],
    matches: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
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
            left_index[begin] = num
            right_index[begin] = position
            begin += 1
            match_index += 1
    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_from_ranges_matches_positions_keep_first(
    booleans: cython.schar[:],
    matches: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
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
            if (base < 0) or (base > position):
                base = position
            match_index += 1
        left_index[begin] = num
        right_index[begin] = base
        begin += 1
    return np.asarray(left_index), np.asarray(right_index)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_from_ranges_matches_positions_keep_last(
    booleans: cython.schar[:],
    matches: cython.schar[:],
    indexers: cython.long[:],
    sizes: cython.long[:],
    positions: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
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
            if (base < 0) or (base < position):
                base = position
            match_index += 1
        left_index[begin] = num
        right_index[begin] = base
        begin += 1

    return np.asarray(left_index), np.asarray(right_index)


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
