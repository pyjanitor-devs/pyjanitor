"""Various Functions powered by Numba"""

import numpy as np
import pandas as pd
from janitor.functions.utils import (
    _generic_func_cond_join,
    _JoinOperator,
    less_than_join_types,
    greater_than_join_types,
)
from numba import njit, prange
from pandas.api.types import is_extension_array_dtype, is_datetime64_dtype


def _numba_equi_join(df, right, eqs, ge_gt, le_lt):
    """
    Compute indices when an equi join is present.
    """
    # the logic is to delay searching for actual matches
    # while reducing the search space
    # to get the smallest possible search area
    # this serves as an alternative to pandas' hash join
    # and in some cases,
    # usually for many to many joins,
    # can offer significant performance improvements.
    # it relies on binary searches, within the groups,
    # and relies on the fact that sorting ensures the first
    # two columns from the right dataframe are in ascending order
    # per group - this gives us the opportunity to
    # only do a linear search, within the groups,
    # for the last column (if any)
    # (the third column is applicable only for range joins)
    # Example :
    #     df1:
    #    id  value_1
    # 0   1        2
    # 1   1        5
    # 2   1        7
    # 3   2        1
    # 4   2        3
    # 5   3        4
    #
    #
    #  df2:
    #    id  value_2A  value_2B
    # 0   1         0         1
    # 1   1         3         5
    # 2   1         7         9
    # 3   1        12        15
    # 4   2         0         1
    # 5   2         2         4
    # 6   2         3         6
    # 7   3         1         3
    #
    #
    # join condition ->
    # ('id', 'id', '==') &
    # ('value_1', 'value_2A','>') &
    # ('value_1', 'value_2B', '<')
    #
    #
    # note how for df2, id and value_2A
    # are sorted per group
    # the third column (relevant for range join)
    # may or may not be sorted per group
    # (the group is determined by the values of the id column)
    # and as such, we do a linear search in that space, per group
    #
    # first we get the slice boundaries based on id -> ('id', 'id', '==')
    # value     start       end
    #  1         0           4
    #  1         0           4
    #  1         0           4
    #  2         4           7
    #  2         4           7
    #  3         7           8
    #
    # next step is to get the slice end boundaries,
    # based on the greater than condition
    # -> ('value_1', 'value_2A', '>')
    # the search will be within each boundary
    # so for the first row, value_1 is 2
    # the boundary search will be between 0, 4
    # for the last row, value_1 is 4
    # and its boundary search will be between 7, 8
    # since value_2A is sorted per group,
    # a binary search is employed
    # value     start       end      value_1   new_end
    #  1         0           4         2         1
    #  1         0           4         5         2
    #  1         0           4         7         2
    #  2         4           7         1         4
    #  2         4           7         3         6
    #  3         7           8         4         8
    #
    # next step is to get the start boundaries,
    # based on the less than condition
    # -> ('value_1', 'value_2B', '<')
    # note that we have new end boundaries,
    # and as such, our boundaries will use that
    # so for the first row, value_1 is 2
    # the boundary search will be between 0, 1
    # for the 5th row, value_1 is 3
    # and its boundary search will be between 4, 6
    # for value_2B, which is the third column
    # sinc we are not sure whether it is sorted or not,
    # a cumulative max array is used,
    # to get the earliest possible slice start
    # value     start       end      value_1   new_start   new_end
    #  1         0           4         2         -1           1
    #  1         0           4         5         -1           2
    #  1         0           4         7         -1           2
    #  2         4           7         1         -1           5
    #  2         4           7         3         5            6
    #  3         7           8         4         -1           8
    #
    # if there are no matches, boundary is reported as -1
    # from above, we can see that our search space
    # is limited to just 5, 6
    # we can then search for actual matches
    # 	id	value_1	id	value_2A	value_2B
    # 	2	  3	    2	   2	       4
    #
    left_column, right_column, _ = eqs
    left_arr = df[left_column]._values
    right_arr = right[right_column]._values
    left_index = df.index._values
    right_index = right.index._values
    slice_starts = right_arr.searchsorted(left_arr, side="left")
    slice_ends = right_arr.searchsorted(left_arr, side="right")
    # check if there is a search space
    # this also lets us know if there are equi matches
    keep_rows = slice_starts < slice_ends
    if not keep_rows.any():
        return None
    if not keep_rows.all():
        left_index = left_index[keep_rows]
        slice_starts = slice_starts[keep_rows]
        slice_ends = slice_ends[keep_rows]

    ge_arr1 = None
    ge_arr2 = None
    ge_strict = None
    if ge_gt:
        left_column, right_column, op = ge_gt
        ge_arr1 = df.loc[left_index, left_column]._values
        ge_arr2 = right[right_column]._values
        if is_extension_array_dtype(ge_arr1):
            array_dtype = ge_arr1.dtype.numpy_dtype
            ge_arr1 = ge_arr1.astype(array_dtype)
            ge_arr2 = ge_arr2.astype(array_dtype)
        if is_datetime64_dtype(ge_arr1):
            ge_arr1 = ge_arr1.view(np.int64)
            ge_arr2 = ge_arr2.view(np.int64)
        ge_strict = True if op == ">" else False

    le_arr1 = None
    le_arr2 = None
    le_strict = None
    if le_lt:
        left_column, right_column, op = le_lt
        le_arr1 = df.loc[left_index, left_column]._values
        le_arr2 = right[right_column]._values
        if is_extension_array_dtype(le_arr1):
            array_dtype = le_arr1.dtype.numpy_dtype
            le_arr1 = le_arr1.astype(array_dtype)
            le_arr2 = le_arr2.astype(array_dtype)
        if is_datetime64_dtype(le_arr1):
            le_arr1 = le_arr1.view(np.int64)
            le_arr2 = le_arr2.view(np.int64)
        le_strict = True if op == "<" else False

    if le_lt and ge_gt:
        left_index, right_index = _numba_equi_join_range_join(
            left_index,
            right_index,
            slice_starts,
            slice_ends,
            ge_arr1,
            ge_arr2,
            ge_strict,
            le_arr1,
            le_arr2,
            le_strict,
        )

    elif le_lt:
        left_index, right_index = _numba_equi_le_join(
            left_index,
            right_index,
            slice_starts,
            slice_ends,
            le_arr1,
            le_arr2,
            le_strict,
        )

    else:
        left_index, right_index = _numba_equi_ge_join(
            left_index,
            right_index,
            slice_starts,
            slice_ends,
            ge_arr1,
            ge_arr2,
            ge_strict,
        )

    if left_index is None:
        return None

    return left_index, right_index


@njit(cache=True, parallel=True)
def _numba_equi_le_join(
    left_index,
    right_index,
    slice_starts,
    slice_ends,
    le_arr1,
    le_arr2,
    le_strict,
):
    """
    Get indices for an equi join
    and a less than join
    """
    length = left_index.size
    starts = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    # sizes array is used to track the exact
    # number of matches per slice_start
    sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        l1 = le_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = le_arr2[slice_start:slice_end]
        start = np.searchsorted(r1, l1, side="left")
        if start < r1.size:
            if le_strict and (l1 == r1[start]):
                start = np.searchsorted(r1, l1, side="right")
        if start == r1.size:
            start = -1
            starts[num] = start
            counts += 1
            booleans[num] = False
        else:
            starts[num] = slice_start + start
            sizes[num] = r1.size - start
    if counts == length:
        return None, None
    if counts > 0:
        left_index = left_index[booleans]
        starts = starts[booleans]
        slice_ends = slice_ends[booleans]
        sizes = sizes[booleans]

    slice_starts = starts
    starts = None
    # build the left and right indices
    # idea is to populate the indices
    # based on the number of matches
    # per slice_start and ensure alignment with the
    # slice_start during the iteration
    cum_sizes = np.cumsum(sizes)
    starts = np.empty(slice_ends.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = cum_sizes[:-1]
    r_index = np.empty(cum_sizes[-1], dtype=np.intp)
    l_index = np.empty(cum_sizes[-1], dtype=np.intp)
    for num in prange(slice_ends.size):
        start = starts[num]
        r_ind = slice_starts[num]
        l_ind = left_index[num]
        width = sizes[num]
        if width == 1:
            r_index[start] = right_index[r_ind]
            l_index[start] = l_ind
        else:
            for n in range(width):
                indexer = start + n
                r_index[indexer] = right_index[r_ind + n]
                l_index[indexer] = l_ind

    return l_index, r_index


@njit(cache=True, parallel=True)
def _numba_equi_ge_join(
    left_index,
    right_index,
    slice_starts,
    slice_ends,
    ge_arr1,
    ge_arr2,
    ge_strict,
):
    """
    Get indices for an equi join
    and a greater than join
    """

    length = left_index.size
    ends = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    # sizes array is used to track the exact
    # number of matches per slice_start
    sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        l1 = ge_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = ge_arr2[slice_start:slice_end]
        end = np.searchsorted(r1, l1, side="right")
        if end > 0:
            if ge_strict and (l1 == r1[end - 1]):
                end = np.searchsorted(r1, l1, side="left")
        if end == 0:
            end = -1
            ends[num] = end
            counts += 1
            booleans[num] = False
        else:
            ends[num] = slice_start + end
            sizes[num] = end
    if counts == length:
        return None, None
    if counts > 0:
        left_index = left_index[booleans]
        ends = ends[booleans]
        slice_starts = slice_starts[booleans]
        sizes = sizes[booleans]
    slice_ends = ends
    ends = None
    # build the left and right indices
    # idea is to populate the indices
    # based on the number of matches
    # per slice_start and ensure alignment with the
    # slice_start during the iteration
    cum_sizes = np.cumsum(sizes)
    starts = np.empty(slice_ends.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = cum_sizes[:-1]
    r_index = np.empty(cum_sizes[-1], dtype=np.intp)
    l_index = np.empty(cum_sizes[-1], dtype=np.intp)
    for num in prange(slice_ends.size):
        start = starts[num]
        r_ind = slice_starts[num]
        l_ind = left_index[num]
        width = sizes[num]
        if width == 1:
            r_index[start] = right_index[r_ind]
            l_index[start] = l_ind
        else:
            for n in range(width):
                indexer = start + n
                r_index[indexer] = right_index[r_ind + n]
                l_index[indexer] = l_ind

    return l_index, r_index


@njit(cache=True, parallel=True)
def _numba_equi_join_range_join(
    left_index,
    right_index,
    slice_starts,
    slice_ends,
    ge_arr1,
    ge_arr2,
    ge_strict,
    le_arr1,
    le_arr2,
    le_strict,
):
    """
    Get indices for an equi join
    and a range join
    """
    length = left_index.size
    ends = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    counts = 0
    for num in prange(length):
        l1 = ge_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = ge_arr2[slice_start:slice_end]
        end = np.searchsorted(r1, l1, side="right")
        if end > 0:
            if ge_strict and (l1 == r1[end - 1]):
                end = np.searchsorted(r1, l1, side="left")
        if end == 0:
            end = -1
            ends[num] = end
            counts += 1
            booleans[num] = False
        else:
            ends[num] = slice_start + end
    if counts == length:
        return None, None
    # cumulative array
    # used to find the first possible match
    # for the less than section below
    max_arr = np.empty_like(le_arr2)
    counter = 0  # are all groups monotonic increasing?
    for num in prange(length):
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = le_arr2[slice_start:slice_end]
        start = r1[0]
        max_arr[slice_start] = start
        if r1.size > 1:
            for n in range(1, r1.size):
                new_value = r1[n]
                check = start < new_value
                if check:
                    start = new_value
                else:
                    counter += 1
                max_arr[slice_start + n] = start

    if counts > 0:
        left_index = left_index[booleans]
        le_arr1 = le_arr1[booleans]
        ends = ends[booleans]
        slice_starts = slice_starts[booleans]
    slice_ends = ends
    ends = None

    length = left_index.size
    starts = np.empty(length, dtype=np.intp)
    booleans = np.ones(length, dtype=np.bool_)
    if counter == 0:
        sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        l1 = le_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = max_arr[slice_start:slice_end]
        start = np.searchsorted(r1, l1, side="left")
        if start < r1.size:
            if le_strict and (l1 == r1[start]):
                start = np.searchsorted(r1, l1, side="right")
        if start == r1.size:
            start = -1
            starts[num] = start
            counts += 1
            booleans[num] = False
        else:
            starts[num] = slice_start + start
            if counter == 0:
                sizes[num] = r1.size - start
    if counts == length:
        return None, None
    if counts > 0:
        left_index = left_index[booleans]
        le_arr1 = le_arr1[booleans]
        starts = starts[booleans]
        slice_ends = slice_ends[booleans]
        if counter == 0:
            sizes = sizes[booleans]

    slice_starts = starts
    starts = None

    # no need to run a comparison
    # since all groups are monotonic increasing
    # simply create left and right indices
    if counter == 0:
        cum_sizes = np.cumsum(sizes)
        starts = np.empty(slice_ends.size, dtype=np.intp)
        starts[0] = 0
        starts[1:] = cum_sizes[:-1]
        r_index = np.empty(cum_sizes[-1], dtype=np.intp)
        l_index = np.empty(cum_sizes[-1], dtype=np.intp)
        for num in prange(slice_ends.size):
            start = starts[num]
            r_ind = slice_starts[num]
            l_ind = left_index[num]
            width = sizes[num]
            if width == 1:
                r_index[start] = right_index[r_ind]
                l_index[start] = l_ind
            else:
                for n in range(width):
                    indexer = start + n
                    r_index[indexer] = right_index[r_ind + n]
                    l_index[indexer] = l_ind

        return l_index, r_index

    # get exact no of rows for left and right index
    # sizes is used to track the exact
    # number of matches per slice_start
    sizes = np.empty(slice_starts.size, dtype=np.intp)
    counts = 0
    for num in prange(slice_ends.size):
        l1 = le_arr1[num]
        start = slice_starts[num]
        end = slice_ends[num]
        r1 = le_arr2[start:end]
        internal_count = 0
        if le_strict:
            for n in range(r1.size):
                check = l1 < r1[n]
                internal_count += check
                counts += check
        else:
            for n in range(r1.size):
                check = l1 <= r1[n]
                internal_count += check
                counts += check
        sizes[num] = internal_count
    # populate the left and right index
    # idea is to populate the indices
    # based on the number of matches
    # per slice_start and ensure alignment with the
    # slice_start during the iteration
    r_index = np.empty(counts, dtype=np.intp)
    l_index = np.empty(counts, dtype=np.intp)
    starts = np.empty(sizes.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = np.cumsum(sizes)[:-1]
    for num in prange(sizes.size):
        l_ind = left_index[num]
        l1 = le_arr1[num]
        starter = slice_starts[num]
        slicer = slice(starter, slice_ends[num])
        r1 = le_arr2[slicer]
        start = starts[num]
        counter = sizes[num]
        if le_strict:
            for n in range(r1.size):
                if not counter:
                    break
                check = l1 < r1[n]
                if not check:
                    continue
                l_index[start] = l_ind
                r_index[start] = right_index[starter + n]
                counter -= 1
                start += 1
        else:
            for n in range(r1.size):
                if not counter:
                    break
                check = l1 <= r1[n]
                if not check:
                    continue
                l_index[start] = l_ind
                r_index[start] = right_index[starter + n]
                counter -= 1
                start += 1
    return l_index, r_index


def _numba_dual_join(df: pd.DataFrame, right: pd.DataFrame, pair: list):
    """
    # https://www.scitepress.org/papers/2018/68268/68268.pdf
    An alternative to the _range_indices algorithm
    and more generalised - it covers any pair of non equi joins
    in >, >=, <, <=.
    Returns a tuple of left and right indices.
    """
    # https://www.scitepress.org/papers/2018/68268/68268.pdf

    # summary:
    # get regions for first and second conditions in the pair
    # (l_col1, r_col1, op1), (l_col2, r_col2, op2)
    # the idea is that r_col1 should always be ahead of the
    # appropriate value from lcol1; same applies to l_col2 & r_col2.
    # if the operator is in less than join types
    # the l_col should be in ascending order
    # if in greater than join types, l_col should be
    # in descending order
    # Example :
    #     df1:
    #    id  value_1
    # 0   1        2
    # 1   1        5
    # 2   1        7
    # 3   2        1
    # 4   2        3
    # 5   3        4
    #
    #
    #  df2:
    #    id  value_2A  value_2B
    # 0   1         0         1
    # 1   1         3         5
    # 2   1         7         9
    # 3   1        12        15
    # 4   2         0         1
    # 5   2         2         4
    # 6   2         3         6
    # 7   3         1         3
    #
    #
    # ('value_1', 'value_2A','>'), ('value_1', 'value_2B', '<')
    # for the first pair, since op is greater than
    # 'value_1' is sorted in descending order
    #  our pairing should be :
    # value  source      region number
    # 12   value_2A       0
    # 7    value_2A       1
    # 7    value_1        2
    # 5    value_1        2
    # 4    value_1        2
    # 3    value_2A       2
    # 3    value_2A       2
    # 3    value_1        3
    # 2    value_2A       3
    # 2    value_1        4
    # 1    value_2A       4
    # 1    value_1        5
    # 0    value_2A       5
    # 0    value_2A       5
    #
    # note that 7 for value_2A is not matched with 7 of value_1
    # because it is >, not >=, hence the different region numbers
    # looking at the output above, we can safely discard regions 0 and 1
    # since they do not have any matches with value_1
    # for the second pair, since op is <, value_1 is sorted
    # in ascending order, and our pairing should be:
    #   value    source    region number
    #     1    value_2B       0
    #     1    value_2B       1
    #     1    value_1        2
    #     2    value_1        2
    #     3    value_2B       2
    #     3    value_1        3
    #     4    value_2B       3
    #     4    value_1        4
    #     5    value_2B       4
    #     5    value_1        5
    #     6    value_2B       5
    #     7    value_1        6
    #     9    value_2B       6
    #     15   value_2B       6
    #
    # from the above we can safely discard regions 0 and 1, since there are
    # no matches with value_1 ... note that the index for regions 0 and 1
    # coincide with the index for region 5 values in value_2A(0, 0);
    # as such those regions will be discarded.
    # Similarly, the index for regions 0 and 1 of value_2A(12, 7)
    # coincide with the index for regions 6 for value_2B(9, 15);
    # these will be discarded as well.
    # let's create a table of the regions, paired with the index
    #
    #
    #  value_1 :
    ###############################################
    # index-->  2  1  5  4  0  3
    # pair1-->  2  2  2  3  4  5
    # pair2-->  6  5  4  3  2  2
    ###############################################
    #
    #
    # value_2A, value_2B
    ##############################################
    # index --> 1  6  5  7
    # pair1 --> 2  2  3  4
    # pair2 --> 4  5  3  2
    ##############################################
    #
    # To find matching indices, the regions from value_1 must be less than
    # or equal to the regions in value_2A/2B.
    # pair1 <= pair1 and pair2 <= pair2
    # Starting from the highest region in value_1
    # 5 in pair1 is not less than any in value_2A/2B, so we discard
    # 4 in pair1 is matched to 4 in pair1 of value_2A/2B
    # we look at the equivalent value in pair2 for 4, which is 2
    # 2 matches 2 in pair 2, so we have a match -> (0, 7)
    # 3 in pair 1 from value_1 matches 3 and 4 in pair1 for value_2A/2B
    # next we compare the equivalent value from pair2, which is 3
    # 3 matches only 3 in value_2A/2B, so our only match is  -> (4, 5)
    # next is 2 (we have 3 2s in value_1 for pair1)
    # they all match 2, 2, 3, 4 in pair1 of value_2A/2B
    # compare the first equivalent in pair2 -> 4
    # 4 matches only 4, 5 in pair2 of value_2A/2B
    # ->(5, 1), (5, 6)
    # the next equivalent is -> 5
    # 5 matches only 5 in pair2 of value_2A/2B
    # -> (1, 6)
    # the last equivalent is -> 6
    # 6 has no match in pair2 of value_2A/2B, so we discard
    # our final matching indices for the left and right pairs
    #########################################################
    # left_index      right_index
    #     0              7
    #     4              5
    #     5              1
    #     5              6
    #     1              6
    ########################################################
    # and if we index the dataframes, we should get the output below:
    #################################
    #    value_1  value_2A  value_2B
    # 0        2         1         3
    # 1        5         3         6
    # 2        3         2         4
    # 3        4         3         5
    # 4        4         3         6
    ################################
    mapping = {">=": "<=", ">": "<", "<=": ">=", "<": ">"}
    left_indices = []
    right_indices = []
    left_regions = []
    right_regions = []
    left_index = df.index
    right_index = right.index
    for num, pairing in enumerate(pair):
        left_on, right_on, op = pairing
        if num:
            left_c = df.loc[left_index, left_on]
            right_c = right.loc[right_index, right_on]
        else:
            left_c = df[left_on]
            right_c = right[right_on]
        left_on, right_on, op = pairing
        # the flip is because the left Series
        # is the control point
        # and we can easily get the regions back
        # by keeping track and using the logic
        # explained below
        outcome = _generic_func_cond_join(
            left=right_c,
            right=left_c,
            op=mapping[op],
            multiple_conditions=True,
            keep="all",
        )
        if not outcome:
            return None
        right_index, left_index, search_indices = outcome

        if op in greater_than_join_types:
            left_index = left_index[::-1]
            search_indices = left_index.size - search_indices
        # logic for computing regions
        # relies on binary search
        # subtract 1 from search indices
        # to align it with the lowest value for left region
        # say we had 2, 3, 5, 8, for the left region
        # and 6 for the right region
        # and is a < operation
        # a binary search returns 3
        # subtracting 1, yields 2,
        # which pairs it correctly with 5
        # since 2, 3, 5 are less than 6
        # if it was a > operation
        # first a subtraction from len(left) -> 4 - 3
        # which yields 1
        # flipping the left region in descending order
        # -> 8, 3, 5 ,2
        # subtract 1 from the search index yields 0
        # which correctly pairs with 8,
        # since 8 is the first closest number greater than 6
        # from here on we can compute the regions
        right_region, indexer = pd.factorize(search_indices - 1, sort=True)
        length = indexer[-1] + 1
        if length < left_index.size:
            left_index = left_index[:length]
        left_region = np.full(shape=length, dtype=np.intp, fill_value=-1)
        indexer_pos = np.arange(indexer.size)
        left_region[indexer] = indexer_pos
        bools = left_region == -1
        # spool through to fill up empty points
        # with the immediate next region number
        if bools.any():
            fill_pos = bools.nonzero()[0]
            fill_pos = indexer.searchsorted(fill_pos, side="left")
            left_region[bools] = indexer_pos[fill_pos]
        left_indices.append(left_index)
        right_indices.append(right_index)
        left_regions.append(left_region)
        right_regions.append(right_region)

    def _realign(indices, regions):
        """
        Realign the indices and regions
        obtained from _get_regions.
        """
        # we've got two regions, since we have two pairs
        # there might be region numbers missing from either
        # or misalignment of the arrays
        # this function ensures the regions are properly aligned
        arr1, arr2 = indices
        region1, region2 = regions
        # arr2 is used as the reference point
        # because we are certain that at the very least
        # it has the same items as arr1, but not more
        indexer = pd.Index(arr2).get_indexer(arr1)
        mask = indexer == -1
        if mask.any():
            arr1 = arr1[~mask]
            region1 = region1[~mask]
            indexer = indexer[~mask]
        region2 = region2[indexer]
        return arr1, region1, region2

    left_index, l_table1, l_table2 = _realign(left_indices, left_regions)
    right_index, r_table1, r_table2 = _realign(right_indices, right_regions)

    del (
        left_indices,
        left_regions,
        right_indices,
        right_regions,
        left_region,
        right_region,
    )

    # we'll be running a for-loop/binary search
    # in r_table2 to get positions
    # where l_table2 is less than or equal to r_table2
    # sorting here allows us to search in the first level
    # for positions where l_table1 <= r_table1
    # efficiently with a binary search
    if not pd.Series(r_table1).is_monotonic_increasing:
        indexer = np.lexsort((r_table2, r_table1))
        right_index, r_table1, r_table2 = (
            right_index[indexer],
            r_table1[indexer],
            r_table2[indexer],
        )

    indexer = None

    positions = r_table1.searchsorted(l_table1, side="left")
    # anything equal to the length of r_table1
    # implies exclusion
    bools = positions < r_table1.size
    if not bools.any():
        return None
    if not bools.all():
        positions = positions[bools]
        left_index = left_index[bools]
        l_table2 = l_table2[bools]

    if pd.Series(r_table2).is_monotonic_decreasing:
        # our work here is significantly easier
        # as we can run a binary search to find exact positions
        # which is better than running a linear search on every entry
        # to check where l_table2 <= r_table2
        ends = r_table2.size - r_table2[::-1].searchsorted(
            l_table2, side="left"
        )
        keep_rows = positions < ends
        if not keep_rows.all():
            positions = positions[keep_rows]
            ends = ends[keep_rows]
            left_index = left_index[keep_rows]
        if ((ends - positions) == 1).all():
            return left_index, right_index[positions]
        return _get_indices_single(
            l_index=left_index,
            r_index=right_index,
            counts=ends - positions,
            starts=positions,
            ends=ends,
        )

    # find the maximum from the bottom upwards
    # if value from l_table2 is greater than the maximum
    # there is no point searching within that space
    max_arr = np.maximum.accumulate(r_table2[::-1])[::-1]
    bools = l_table2 > max_arr[positions]
    # there is no match
    if bools.all():
        return None
    if bools.any():
        positions = positions[~bools]
        left_index = left_index[~bools]
        l_table2 = l_table2[~bools]
    # this is where numba comes in
    # to improve performance for the for-loop
    return _get_indices_dual(
        left_index, l_table2, right_index, r_table2, positions, max_arr
    )


def _numba_single_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    keep: str,
) -> tuple:
    """Return matching indices for single non-equi join."""

    outcome = _generic_func_cond_join(
        left=left,
        right=right,
        op=op,
        multiple_conditions=True,
        keep=keep,
    )

    if (outcome is None) or (op == _JoinOperator.NOT_EQUAL.value):
        return outcome

    left_index, right_index, search_indices = outcome

    if op in greater_than_join_types:
        starts = np.zeros(shape=search_indices.size, dtype=np.int8)
        ends = search_indices
        counts = search_indices
    else:
        ends = np.full(
            shape=search_indices.size,
            dtype=np.intp,
            fill_value=right_index.size,
        )
        starts = search_indices
        counts = ends - starts
    if keep == "all":
        return _get_indices_single(
            l_index=left_index,
            r_index=right_index,
            counts=counts,
            starts=starts,
            ends=ends,
        )

    if (
        (keep == "first")
        and (op in less_than_join_types)
        and pd.Series(right_index).is_monotonic_increasing
    ):
        return left_index, right_index[search_indices]
    if (
        (keep == "last")
        and (op in greater_than_join_types)
        and pd.Series(right_index).is_monotonic_increasing
    ):
        return left_index, right_index[search_indices - 1]
    if keep == "first":
        right_index = _numba_single_non_equi_keep_first(
            right_index, starts, ends
        )
    else:
        right_index = _numba_single_non_equi_keep_last(
            right_index, starts, ends
        )
    return left_index, right_index


@njit(parallel=True)
def _numba_single_non_equi_keep_first(
    right_index: np.ndarray, starts, ends
) -> np.ndarray:
    """
    Generate all indices when keep = `first`
    Applies only to >, >= , <, <= operators.
    """
    r_index = np.empty(starts.size, np.intp)
    for num in prange(starts.size):
        indexer = slice(starts[num], ends[num])
        r_index[num] = right_index[indexer].min()
    return r_index


@njit(parallel=True)
def _numba_single_non_equi_keep_last(
    right_index: np.ndarray, starts, ends
) -> np.ndarray:
    """
    Generate all indices when keep = `last`
    Applies only to >, >= , <, <= operators.
    """
    r_index = np.empty(starts.size, np.intp)
    for num in prange(starts.size):
        indexer = slice(starts[num], ends[num])
        r_index[num] = right_index[indexer].max()
    return r_index


@njit(cache=True, parallel=True)
def _get_indices_single(l_index, r_index, counts, starts, ends):
    """ "Compute indices when starts and ends are already known"""
    lengths = np.cumsum(counts)
    left_index = np.empty(lengths[-1], np.intp)
    right_index = np.empty(lengths[-1], np.intp)
    start_indices = np.empty(lengths.size, np.intp)
    start_indices[0] = 0
    start_indices[1:] = lengths[:-1]
    for num in prange(lengths.size):
        start = start_indices[num]
        width = counts[num]
        l_indexer = slice(start, start + width)
        left_index[l_indexer] = l_index[num]
        r_indexer = slice(starts[num], ends[num])
        right_index[l_indexer] = r_index[r_indexer]

    return left_index, right_index


@njit(cache=True, parallel=True)
def _get_indices_dual(
    l_index, l_table2, r_index, r_table2, positions, max_arr
):
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins (pairs).
    """
    length = r_index.size
    counts = np.empty(positions.size, dtype=np.intp)
    ends = np.empty(positions.size, dtype=np.intp)

    # first let's get the total number of matches
    for num in prange(l_index.size):
        l2 = l_table2[num]
        pos = positions[num]
        # get the first point where l2
        # is greater than the cumulative max
        # that will serve as the range
        # (pos, pos_end)
        # within which to search for actual matches

        # no point checking the first position
        # as we already know that l2 <= r_table2[pos]
        # what happens below?
        # max_arr is a cumulative max in decreasing order
        # e.g max_arr -> [4 3 3 2 1 1 1]
        # slice it from pos+1, since we have established
        # that l2 <= r_table2[pos]
        # -> [3 3 2 1 1 1]
        # flip it into increasing order
        # -> [1 1 1 2 3 3]
        # making it easy to do a binary search
        # say l2 = 4
        # binary search returns the lowest possible point
        # which is 6
        # subtract from the actual r_table2 length (7)
        # -> 7 - 6 = 1
        # now we have our (pos_start, pos_end) -> (0, 1)
        # our search space is reduced, allowing us to do
        # less work with the comparison operation
        sliced = max_arr[slice(pos + 1, None)][::-1]
        pos_end = np.searchsorted(sliced, l2, side="left")
        pos_end = length - pos_end
        ends[num] = pos_end
        # get the total number of exact matches
        # for l2
        end = 0
        for ind in range(pos, pos_end):
            out = l2 <= r_table2[ind]
            end += out
        counts[num] = end

    start_indices = np.cumsum(counts)
    starts = np.empty(counts.size, dtype=np.intp)
    starts[0] = 0
    starts[1:] = start_indices[:-1]

    # create left and right indexes
    left_index = np.empty(start_indices[-1], dtype=np.intp)
    right_index = np.empty(start_indices[-1], dtype=np.intp)
    start_indices = None

    for num in prange(l_index.size):
        pos = positions[num]
        pos_end = ends[num]
        l2 = l_table2[num]
        l3 = l_index[num]
        start = starts[num]
        counter = counts[num]
        for ind in range(pos, pos_end):
            if not counter:
                break
            if r_table2[ind] < l2:
                continue
            left_index[start] = l3
            right_index[start] = r_index[ind]
            start += 1
            counter -= 1

    return left_index, right_index
