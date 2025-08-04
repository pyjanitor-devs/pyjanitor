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
    else:
        compare = left_value != right_value
    return compare


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches(
    start_indices: cython.long[:],
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
    any_match: cython.bint = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: scalar_types = left_array[num]
        start: cython.long = starts[num]
        end: cython.long = ends[num]
        count: cython.long = 0
        begin: cython.Py_ssize_t = start_indices[num]
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
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_ne_pandas_array(
    start_indices: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Special situation to get positive matches from comparison,
    where `op==!=` - specifically for pandas extension arrays (pd.NA)
    """
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    any_match: cython.bint = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: scalar_types = left_array[num]
        start: cython.long = starts[num]
        end: cython.long = ends[num]
        count: cython.long = 0
        begin: cython.Py_ssize_t = start_indices[num]
        l_bool: cython.schar = left_booleans[num]
        for number in range(start, end):
            r_bool: cython.schar = right_booleans[number]
            # pandas' pd.NA uses a different logic from np.nan
            # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
            if (l_bool == 1) | (r_bool == 1):
                boolean: cython.schar = 0
            else:
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
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_ne(
    start_indices: cython.long[:],
    starts: cython.long[:],
    ends: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Special situation to get positive matches from comparison,
    where `op==!=`
    """
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    any_match: cython.bint = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: scalar_types = left_array[num]
        start: cython.long = starts[num]
        end: cython.long = ends[num]
        count: cython.long = 0
        begin: cython.Py_ssize_t = start_indices[num]
        l_bool: cython.schar = left_booleans[num]
        for number in range(start, end):
            r_bool: cython.schar = right_booleans[number]
            if (l_bool == 1) | (r_bool == 1):
                boolean: cython.schar = 1
            else:
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
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_no_ranges(
    op: cython.int,
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    booleans: cython.schar[:],
):
    """
    Get positive matches from comparison
    """
    counts: cython.Py_ssize_t = left_array.shape[0]
    num: cython.Py_ssize_t
    count_exact_matches: cython.long = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: scalar_types = left_array[num]
        right_value: scalar_types = right_array[num]
        compare: scalar_types = compare_values(
            left_value=left_value, right_value=right_value, op=op
        )
        boolean: cython.schar = cython.cast(cython.schar, compare)
        current_bool: cython.schar = booleans[num]
        boolean = current_bool & boolean
        booleans[num] = boolean
        if boolean == 0:
            continue
        count_exact_matches += 1

    return (
        np.asarray(booleans),
        count_exact_matches,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_no_ranges_ne_pandas_array(
    op: cython.int,
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Special situation to get positive matches from comparison
    where `op==!=` - specific to pandas extension arrays (pd.NA)
    """
    counts: cython.Py_ssize_t = left_array.shape[0]
    num: cython.Py_ssize_t
    count_exact_matches: cython.long = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: scalar_types = left_array[num]
        right_value: scalar_types = right_array[num]
        l_bool: cython.schar = left_booleans[num]
        r_bool: cython.schar = right_booleans[num]
        if (l_bool == 1) | (r_bool == 1):
            boolean: cython.schar = 0
        else:
            compare: scalar_types = compare_values(
                left_value=left_value, right_value=right_value, op=op
            )
            boolean: cython.schar = cython.cast(cython.schar, compare)
        current_bool: cython.schar = booleans[num]
        boolean = current_bool & boolean
        booleans[num] = boolean
        if boolean == 0:
            continue
        count_exact_matches += 1

    return (
        np.asarray(booleans),
        count_exact_matches,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_positive_matches_no_ranges_ne(
    op: cython.int,
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
):
    """
    Special situation to get positive matches from comparison
    where `op==!=`
    """
    counts: cython.Py_ssize_t = left_array.shape[0]
    num: cython.Py_ssize_t
    count_exact_matches: cython.long = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: scalar_types = left_array[num]
        right_value: scalar_types = right_array[num]
        l_bool: cython.schar = left_booleans[num]
        r_bool: cython.schar = right_booleans[num]
        if (l_bool == 1) | (r_bool == 1):
            boolean: cython.schar = 1
        else:
            compare: scalar_types = compare_values(
                left_value=left_value, right_value=right_value, op=op
            )
            boolean: cython.schar = cython.cast(cython.schar, compare)
        current_bool: cython.schar = booleans[num]
        boolean = current_bool & boolean
        booleans[num] = boolean
        if boolean == 0:
            continue
        count_exact_matches += 1

    return (
        np.asarray(booleans),
        count_exact_matches,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_keep_all(
    starts: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    starts_indices: cython.long[:],
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_array: cython.long[::1],
    right_array: cython.long[::1],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get indices if keep=='all'
    """
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    begin: cython.Py_ssize_t = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: cython.long = left_index[num]
        start: cython.long = starts[num]
        counter: cython.Py_ssize_t = starts_indices[num]
        count: cython.long = 0
        array_count: cython.long = counts_array[num]
        size: cython.long = sizes[num]
        for number in range(size):
            if count == array_count:
                break
            match_index: cython.Py_ssize_t = counter + number
            any_match: cython.schar = matches[match_index]
            if any_match == 0:
                continue
            r_index: cython.Py_ssize_t = start + number
            right_value: cython.long = right_index[r_index]
            left_array[begin] = left_value
            right_array[begin] = right_value
            begin += 1
            count += 1
    return np.asarray(left_array), np.asarray(right_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_keep_first(
    starts: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    starts_indices: cython.long[:],
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_array: cython.long[::1],
    right_array: cython.long[::1],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get indices if keep=='first'
    """
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    begin: cython.Py_ssize_t = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: cython.long = left_index[num]
        start: cython.long = starts[num]
        counter: cython.Py_ssize_t = starts_indices[num]
        count: cython.long = 0
        array_count: cython.long = counts_array[num]
        size: cython.long = sizes[num]
        base: cython.long = -1
        compare: cython.bint = 0
        for number in range(size):
            if count == array_count:
                break
            match_index: cython.Py_ssize_t = counter + number
            any_match: cython.schar = matches[match_index]
            if any_match == 0:
                continue
            r_index: cython.Py_ssize_t = start + number
            right_value: cython.long = right_index[r_index]
            compare = (base < 0) or (base > right_value)
            if compare == 1:
                base = right_value
            count += 1
        left_array[begin] = left_value
        right_array[begin] = base
        begin += 1
    return np.asarray(left_array), np.asarray(right_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_keep_last(
    starts: cython.long[:],
    sizes: cython.long[:],
    matches: cython.schar[:],
    starts_indices: cython.long[:],
    left_index: cython.long[:],
    right_index: cython.long[:],
    left_array: cython.long[::1],
    right_array: cython.long[::1],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get indices if keep=='last'
    """
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    begin: cython.Py_ssize_t = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        left_value: cython.long = left_index[num]
        start: cython.long = starts[num]
        counter: cython.Py_ssize_t = starts_indices[num]
        count: cython.long = 0
        array_count: cython.long = counts_array[num]
        size: cython.long = sizes[num]
        base: cython.long = -1
        compare: cython.bint = 0
        for number in range(size):
            if count == array_count:
                break
            match_index: cython.Py_ssize_t = counter + number
            any_match: cython.schar = matches[match_index]
            if any_match == 0:
                continue
            r_index: cython.Py_ssize_t = start + number
            right_value: cython.long = right_index[r_index]
            compare = base < right_value
            if compare == 1:
                base = right_value
            count += 1
        left_array[begin] = left_value
        right_array[begin] = base
        begin += 1
    return np.asarray(left_array), np.asarray(right_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_less_than_strict(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `<` condition
    """
    len_left: cython.long = left_array.shape[0]
    new_starts = np.empty(len_left, dtype="int64")
    starts_view: cython.long[::1] = new_starts
    match: cython.long = 0
    total: cython.long = 0
    num: cython.Py_ssize_t
    for num in range(len_left):
        check: cython.schar = booleans[num]
        if check == 0:
            sizes[num] = 0
            continue
        end: cython.Py_ssize_t = ends[num]
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = starts[num]
        max_idx: cython.Py_ssize_t = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value <= l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        boolean: cython.bint = min_idx == end
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        current_value: scalar_types = right_array[min_idx]
        compare: scalar_types = current_value == l_value
        boolean: cython.bint = cython.cast(cython.bint, compare)
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        starts_view[num] = min_idx
        size: cython.long = end - min_idx
        sizes[num] = size
        total += size
        match += 1
    return (
        new_starts,
        np.asarray(booleans),
        np.asarray(sizes),
        total,
        match,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_less_than(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `<=` condition
    """
    len_left: cython.long = left_array.shape[0]
    new_starts = np.empty(len_left, dtype="int64")
    starts_view: cython.long[::1] = new_starts
    num: cython.Py_ssize_t
    match: cython.long = 0
    total: cython.long = 0
    for num in range(len_left):
        check: cython.schar = booleans[num]
        if check == 0:
            sizes[num] = 0
            continue
        end: cython.Py_ssize_t = ends[num]
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = starts[num]
        max_idx: cython.Py_ssize_t = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value < l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        boolean: cython.bint = min_idx == end
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        starts_view[num] = min_idx
        size: cython.long = end - min_idx
        sizes[num] = size
        match += 1
        total += size
    return (
        new_starts,
        np.asarray(booleans),
        np.asarray(sizes),
        total,
        match,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_greater_than_strict(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `>` condition
    """
    len_left: cython.long = left_array.shape[0]
    new_ends = np.zeros(len_left, dtype="int64")
    ends_view: cython.long[::1] = new_ends
    num: cython.Py_ssize_t
    match: cython.long = 0
    total: cython.long = 0
    for num in range(len_left):
        check: cython.schar = booleans[num]
        if check == 0:
            sizes[num] = 0
            continue
        start: cython.Py_ssize_t = starts[num]
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = starts[num]
        max_idx: cython.Py_ssize_t = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value >= l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        boolean: cython.bint = min_idx == start
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        index: cython.Py_ssize_t = min_idx - 1
        current_value: scalar_types = right_array[index]
        compare: scalar_types = current_value == l_value
        boolean: cython.bint = cython.cast(cython.bint, compare)
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        ends_view[num] = min_idx
        size: cython.long = min_idx - start
        sizes[num] = size
        total += size
        match += 1
    return new_ends, np.asarray(booleans), np.asarray(sizes), total, match


@cython.boundscheck(False)
@cython.wraparound(False)
def update_search_indices_greater_than(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `>=` condition
    """
    len_left: cython.long = left_array.shape[0]
    new_ends = np.zeros(len_left, dtype="int64")
    ends_view: cython.long[::1] = new_ends
    num: cython.Py_ssize_t
    match: cython.long = 0
    total: cython.long = 0
    for num in range(len_left):
        check: cython.schar = booleans[num]
        if check == 0:
            sizes[num] = 0
            continue
        start: cython.Py_ssize_t = starts[num]
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = starts[num]
        max_idx: cython.Py_ssize_t = ends[num]
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value > l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        boolean: cython.bint = min_idx == start
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        ends_view[num] = min_idx
        size: cython.long = min_idx - start
        sizes[num] = size
        total += size
        match += 1
    return new_ends, np.asarray(booleans), np.asarray(sizes), total, match


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
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    length: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    for num in range(length):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        start: cython.Py_ssize_t = starts[num]
        end: cython.Py_ssize_t = ends[num]
        l_value: cython.long = left_index[num]
        for number in range(start, end):
            left_indices[begin] = l_value
            r_value: cython.long = right_index[number]
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
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    length: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    for num in range(length):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        start: cython.Py_ssize_t = starts[num]
        end: cython.Py_ssize_t = ends[num]
        base = -1
        for number in range(start, end):
            r_value: cython.long = right_index[number]
            compare: cython.bint = (base < 0) or (base > r_value)
            if compare == 1:
                base = r_value
        l_value: cython.long = left_index[num]
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
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    length: cython.Py_ssize_t = left_index.shape[0]
    begin: cython.Py_ssize_t = 0
    for num in range(length):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        start: cython.Py_ssize_t = starts[num]
        end: cython.Py_ssize_t = ends[num]
        base = -1
        for number in range(start, end):
            r_value: cython.long = right_index[number]
            compare: cython.bint = base < r_value
            if compare == 1:
                base = r_value
        l_value: cython.long = left_index[num]
        left_indices[begin] = l_value
        right_indices[begin] = base
        begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)
