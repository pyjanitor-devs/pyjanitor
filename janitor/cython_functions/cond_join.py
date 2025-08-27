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
) -> scalar_types:
    compare: scalar_types = 0
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
def get_positive_matches_ranges(
    starts: cython.long[:],
    ends: cython.long[:],
    sizes: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    counts_array: cython.long[:],
    booleans: cython.schar[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
    is_extension_array: cython.bint,
):
    """
    Compute matching locations for multiple conditions
     Applies if there is a `starts` and `ends`
    """
    length: cython.long = starts.size
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    total: cython.long = 0
    l_counts: cython.long = 0
    begin: cython.Py_ssize_t = 0
    bool_: cython.schar
    for num in range(length):
        size = sizes[num]
        check: cython.schar = booleans[num]
        if check == 0:
            begin += size
            counts_array[num] = 0
            continue
        count: cython.long = 0
        l_val: scalar_types = left[num]
        start: cython.Py_ssize_t = starts[num]
        end: cython.Py_ssize_t = ends[num]
        for number in range(start, end):
            check: cython.schar = matches[begin]
            if check == 0:
                begin += 1
                continue
            r_val: scalar_types = right[number]
            bool_: cython.schar = 0
            if op == 5:
                l_bool: cython.schar = left_booleans[num]
                r_bool: cython.schar = right_booleans[number]
                boolean: cython.bint = l_bool | r_bool
                check: cython.bint = cython.cast(
                    cython.bint, is_extension_array
                )
                check = boolean & check
                if boolean == 0:
                    compare: scalar_types = compare_values(
                        left=l_val, right=r_val, op=op
                    )
                    bool_ = cython.cast(cython.schar, compare)
                # pandas' pd.NA uses a different logic from np.nan
                # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
                elif check == 1:
                    bool_ = 0
                elif boolean == 1:
                    bool_ = 1
            else:
                compare: scalar_types = compare_values(
                    left=l_val, right=r_val, op=op
                )
                bool_ = cython.cast(cython.schar, compare)
            matches[begin] = bool_
            begin += 1
            bool_int = cython.cast(cython.long, bool_)
            total += bool_int
            count += bool_int
        counts_array[num] = count
        boolean: cython.bint = count > 0
        bool_: cython.schar = cython.cast(cython.schar, boolean)
        booleans[num] = bool_
        bool_int = cython.cast(cython.long, bool_)
        l_counts += bool_int
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
    left_index: cython.long[:],
    right_index: cython.long[:],
    op: cython.int,
    matches: cython.schar[:],
    left: scalar_types[:],
    right: scalar_types[:],
    left_booleans: cython.schar[:],
    right_booleans: cython.schar[:],
    is_extension_array: cython.bint,
):
    """
    Compute matching locations for multiple conditions
     Applies if there is no `starts` and `ends`
    """
    length: cython.long = left_index.size
    num: cython.Py_ssize_t
    l_index: cython.Py_ssize_t
    r_index: cython.Py_ssize_t
    total: cython.long = 0
    bool_: cython.schar
    for num in range(length):
        l_index = left_index[num]
        r_index = right_index[num]
        check: cython.schar = matches[num]
        if check == 0:
            continue
        l_val: scalar_types = left[l_index]
        r_val: scalar_types = right[r_index]
        if op == 5:
            l_bool: cython.schar = left_booleans[l_index]
            r_bool: cython.schar = right_booleans[r_index]
            boolean: cython.bint = l_bool | r_bool
            check: cython.bint = cython.cast(cython.bint, is_extension_array)
            check = boolean & check
            if boolean == 0:
                compare: scalar_types = compare_values(
                    left=l_val, right=r_val, op=op
                )
                bool_ = cython.cast(cython.schar, compare)
            # pandas' pd.NA uses a different logic from np.nan
            # https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
            elif check == 1:
                bool_ = 0
            elif boolean == 1:
                bool_ = 1
        else:
            compare: scalar_types = compare_values(
                left=l_val, right=r_val, op=op
            )
            bool_ = cython.cast(cython.schar, compare)
        bool_int = cython.cast(cython.long, bool_)
        matches[num] = bool_
        total += bool_int
    return (
        np.asarray(matches),
        total,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def get_row_count_no_ranges(
    counts_array: cython.long[::1],
    left_indices: cython.long[:],
    matches: cython.schar[:],
):
    """
    Compute row count
    """
    length: cython.Py_ssize_t = left_indices.shape[0]
    num: cython.Py_ssize_t
    for num in range(length):
        check: cython.schar = matches[num]
        if check == 0:
            continue
        l_index = left_indices[num]
        counts_array[l_index] += 1
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
    # compute total
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    for num in range(length):
        check: cython.schar = matches[num]
        if check == 0:
            continue
        total += 1
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    begin: cython.Py_ssize_t = 0
    for num in range(length):
        if begin == total:
            break
        check: cython.schar = matches[num]
        if check == 0:
            continue
        l_index: cython.Py_ssize_t = left_indices[num]
        l_val: cython.Py_ssize_t = left_index[l_index]
        r_index: cython.Py_ssize_t = right_indices[num]
        r_val: cython.Py_ssize_t = right_index[r_index]
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
    # compute total
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    base_index: cython.Py_ssize_t = -1
    for num in range(length):
        check: cython.schar = matches[num]
        if check == 0:
            continue
        l_index: cython.Py_ssize_t = left_indices[num]
        if base_index != l_index:
            total += 1
            base_index = l_index
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    begin: cython.Py_ssize_t = 0
    base_index = -1
    for num in range(length):
        check: cython.schar = matches[num]
        if check == 0:
            continue
        l_index: cython.Py_ssize_t = left_indices[num]
        l_val: cython.Py_ssize_t = left_index[l_index]
        r_index: cython.Py_ssize_t = right_indices[num]
        r_val: cython.Py_ssize_t = right_index[r_index]
        if base_index != l_index:
            index_left[begin] = l_val
            index_right[begin] = r_val
            base_index = l_index
            begin += 1
        else:
            begin_: cython.Py_ssize_t = begin - 1
            current: cython.Py_ssize_t = index_right[begin_]
            bool_: cython.bint = current > r_val
            if bool_ == 1:
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
    # compute total
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    length: cython.Py_ssize_t = matches.shape[0]
    base_index: cython.Py_ssize_t = -1
    for num in range(length):
        check: cython.schar = matches[num]
        if check == 0:
            continue
        l_index: cython.Py_ssize_t = left_indices[num]
        if base_index != l_index:
            total += 1
            base_index = l_index
    index_left: cython.long[::1] = np.empty(total, dtype=np.intp)
    index_right: cython.long[::1] = np.empty(total, dtype=np.intp)
    begin: cython.Py_ssize_t = 0
    base_index = -1
    for num in range(length):
        check: cython.schar = matches[num]
        if check == 0:
            continue
        l_index: cython.Py_ssize_t = left_indices[num]
        l_val: cython.Py_ssize_t = left_index[l_index]
        r_index: cython.Py_ssize_t = right_indices[num]
        r_val: cython.Py_ssize_t = right_index[r_index]
        if base_index != l_index:
            index_left[begin] = l_val
            index_right[begin] = r_val
            base_index = l_index
            begin += 1
        else:
            begin_: cython.Py_ssize_t = begin - 1
            current: cython.Py_ssize_t = index_right[begin_]
            bool_: cython.bint = current < r_val
            if bool_ == 1:
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
    counts_array: cython.long[:],
    booleans: cython.schar[:],
):
    """
    Get indices if keep=='all'
    """
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            continue
        val: cython.long = left_index[num]
        size: cython.Py_ssize_t = counts_array[num]
        for _ in range(size):
            left_indices[begin] = val
            begin += 1

    num: cython.Py_ssize_t = 0
    begin: cython.Py_ssize_t = 0
    match_index: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    for num in range(counts):
        start: cython.Py_ssize_t = starts[num]
        end: cython.Py_ssize_t = ends[num]
        check: cython.schar = booleans[num]
        if check == 0:
            size: cython.Py_ssize_t = sizes[num]
            match_index += size
            continue
        for number in range(start, end):
            check: cython.schar = matches[match_index]
            if check == 0:
                match_index += 1
                continue
            val: cython.long = right_index[number]
            right_indices[begin] = val
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
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    begin: cython.Py_ssize_t = 0
    match_index: cython.Py_ssize_t = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            size: cython.Py_ssize_t = sizes[num]
            match_index += size
            continue
        start: cython.long = starts[num]
        end: cython.long = ends[num]
        base: cython.long = -1
        for number in range(start, end):
            check: cython.schar = matches[match_index]
            if check == 0:
                match_index += 1
                continue
            r_val: cython.long = right_index[number]
            check: cython.bint = (base < 0) or (base > r_val)
            if check == 1:
                base = r_val
            match_index += 1
        l_val: cython.long = left_index[num]
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
    counts: cython.Py_ssize_t = starts.shape[0]
    num: cython.Py_ssize_t
    number: cython.Py_ssize_t
    begin: cython.Py_ssize_t = 0
    match_index: cython.Py_ssize_t = 0
    for num in range(counts):
        check: cython.schar = booleans[num]
        if check == 0:
            size: cython.Py_ssize_t = sizes[num]
            match_index += size
            continue
        start: cython.long = starts[num]
        end: cython.long = ends[num]
        base: cython.long = -1
        for number in range(start, end):
            check: cython.schar = matches[match_index]
            if check == 0:
                match_index += 1
                continue
            r_val: cython.long = right_index[number]
            check: cython.bint = base < r_val
            if check == 1:
                base = r_val
            match_index += 1
        l_val: cython.long = left_index[num]
        right_indices[begin] = base
        left_indices[begin] = l_val
        begin += 1
    return np.asarray(left_indices), np.asarray(right_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_search_indices_less_than_strict_first_run(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Get search indices for a `<` condition
    """
    len_left: cython.long = left_array.shape[0]
    len_right: cython.long = right_array.shape[0]
    match: cython.long = 0
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    for num in range(len_left):
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = 0
        max_idx: cython.Py_ssize_t = len_right
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value <= l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        boolean: cython.bint = min_idx == len_right
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
        booleans[num] = 1
        starts[num] = min_idx
        ends[num] = len_right
        size: cython.long = len_right - min_idx
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
def get_search_indices_less_than_first_run(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Get search indices for a `<=` condition
    """
    len_left: cython.long = left_array.shape[0]
    len_right: cython.long = right_array.shape[0]
    match: cython.long = 0
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    for num in range(len_left):
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = 0
        max_idx: cython.Py_ssize_t = len_right
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value < l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        boolean: cython.bint = min_idx == len_right
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        starts[num] = min_idx
        ends[num] = len_right
        size: cython.long = len_right - min_idx
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
def get_search_indices_strictly_equal_first_run(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Get search indices for a `<=` condition
    """
    len_left: cython.long = left_array.shape[0]
    len_right: cython.long = right_array.shape[0]
    match: cython.long = 0
    total: cython.long = 0
    num: cython.Py_ssize_t = 0
    for num in range(len_left):
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = 0
        max_idx: cython.Py_ssize_t = len_right
        # get the starts
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value < l_value:
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        boolean: cython.bint = min_idx == len_right
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        current_value: scalar_types = right_array[min_idx]
        compare: scalar_types = current_value == l_value
        boolean: cython.bint = cython.cast(cython.bint, compare)
        if boolean != 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        starts[num] = min_idx
        # get the ends
        max_idx: cython.Py_ssize_t = len_right
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value > l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        ends[num] = min_idx
        size: cython.long = ends[num] - starts[num]
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
def get_search_indices_greater_than_strict_first_run(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Get search indices for a `>` condition
    """
    len_left: cython.long = left_array.shape[0]
    len_right: cython.long = right_array.shape[0]
    num: cython.Py_ssize_t
    match: cython.long = 0
    total: cython.long = 0
    for num in range(len_left):
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = 0
        max_idx: cython.Py_ssize_t = len_right
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value >= l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        boolean: cython.bint = min_idx == 0
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
        starts[num] = 0
        ends[num] = min_idx
        sizes[num] = min_idx
        total += min_idx
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
def get_search_indices_greater_than_first_run(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Get search indices for a `>=` condition
    """
    len_left: cython.long = left_array.shape[0]
    len_right: cython.long = right_array.shape[0]
    num: cython.Py_ssize_t
    match: cython.long = 0
    total: cython.long = 0
    for num in range(len_left):
        l_value: scalar_types = left_array[num]
        # adapted from numba/np/array_math.py
        min_idx: cython.Py_ssize_t = 0
        max_idx: cython.Py_ssize_t = len_right
        while min_idx < max_idx:
            # to avoid overflow
            mid_idx: cython.Py_ssize_t = min_idx + ((max_idx - min_idx) >> 1)
            current_value: scalar_types = right_array[mid_idx]
            if current_value > l_value:
                max_idx = mid_idx
            else:
                min_idx = mid_idx + 1
        boolean: cython.bint = min_idx == 0
        if boolean == 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        starts[num] = 0
        ends[num] = min_idx
        sizes[num] = min_idx
        total += min_idx
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
        starts[num] = min_idx
        size: cython.long = end - min_idx
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
        starts[num] = min_idx
        size: cython.long = end - min_idx
        sizes[num] = size
        match += 1
        total += size
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
def update_search_indices_strictly_equal_min(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `==` condition
    """
    len_left: cython.long = left_array.shape[0]
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
        current_value: scalar_types = right_array[min_idx]
        compare: scalar_types = current_value == l_value
        boolean: cython.bint = cython.cast(cython.bint, compare)
        if boolean != 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        starts[num] = min_idx
        size: cython.long = end - min_idx
        sizes[num] = size
        match += 1
        total += size
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
def update_search_indices_strictly_equal_max(
    left_array: scalar_types[:],
    right_array: scalar_types[:],
    starts: cython.long[:],
    ends: cython.long[:],
    booleans: cython.schar[:],
    sizes: cython.long[:],
):
    """
    Update search indices for a `==` condition
    """
    len_left: cython.long = left_array.shape[0]
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
        index: cython.Py_ssize_t = min_idx - 1
        current_value: scalar_types = right_array[index]
        compare: scalar_types = current_value == l_value
        boolean: cython.bint = cython.cast(cython.bint, compare)
        if boolean != 1:
            booleans[num] = 0
            sizes[num] = 0
            continue
        booleans[num] = 1
        ends[num] = min_idx
        size: cython.long = min_idx - start
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
        ends[num] = min_idx
        size: cython.long = min_idx - start
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
        ends[num] = min_idx
        size: cython.long = min_idx - start
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
    length: cython.Py_ssize_t = starts.shape[0]
    matches: cython.schar[::1] = np.empty(length, dtype=np.int8)
    lenght: cython.Py_ssize_t = arr.shape[0]
    bools: cython.schar[::1] = np.zeros(lenght, dtype=np.int8)
    num: cython.Py_ssize_t = 0
    number: cython.Py_ssize_t = 0
    all_true: cython.schar = 1
    for num in range(length):
        if booleans[num] == 0:
            continue
        start: cython.Py_ssize_t = starts[num]
        end: cython.Py_ssize_t = ends[num]
        if bools[start] == 1:
            matches[num] = 1
            continue
        current: scalar_types = arr[start]
        next_start: cython.Py_ssize_t = start + 1
        bool_: cython.bint = 0
        bools[start] = 1
        tracker: cython.schar = 1
        for number in range(next_start, end):
            val: scalar_types = arr[number]
            bool_ = current > val
            if bool_ == 1:
                tracker = 0
                all_true = 0
                break
            current = val
        matches[num] = tracker
    return all_true
