"""Various Functions powered by Numba"""

from typing import Union

import numpy as np
import pandas as pd
from numba import njit, prange
from pandas.api.types import (
    is_datetime64_dtype,
    is_extension_array_dtype,
)

from janitor.functions.utils import (
    _generic_func_cond_join,
    greater_than_join_types,
    less_than_join_types,
)


def _convert_to_numpy(left: np.ndarray, right: np.ndarray) -> tuple:
    """
    Ensure array is a numpy array.
    """
    if is_extension_array_dtype(left):
        array_dtype = left.dtype.numpy_dtype
        left = left.astype(array_dtype)
        right = right.astype(array_dtype)
    if is_datetime64_dtype(left):
        left = left.view(np.int64)
        right = right.view(np.int64)
    return left, right


def _get_regions_equi(
    left: pd.Index,
    right: pd.Index,
    left_index: np.ndarray,
    right_index: np.ndarray,
) -> Union[tuple[np.ndarray], None]:
    """
    Generate regions for an equi join.

    Returns indices and regions.
    """
    # it is assumed to be a many-to-many join
    # if it isnt, then using Pandas' one-to-one
    # or many-to-one via the use_numba=False
    # argument should offer better performance
    left_positions, left_uniques = left.factorize()
    right_positions, right_uniques = right.factorize()
    *_, right_indexer = left_uniques.join(
        right_uniques, how="left", sort=False, return_indexers=True
    )
    # left and right are the same
    # and have the same order
    # e.g pd.Index([1,2,3]) vs pd.Index([1,2,3])
    if right_indexer is None:
        indexer = np.arange(left_uniques.size)
        left_region = indexer[left_positions]
        right_region = indexer[right_positions]
        return left_index, right_index, left_region, right_region
    booleans = right_indexer == -1
    # absolutely no match
    if booleans.all():
        return None
    # left and right are not the same
    # pd.Index([1,2,3]) vs pd.Index([2,1])
    if booleans.any():
        booleans = ~booleans
        left_region = booleans.cumsum() - 1
        left_region = left_region[left_positions]
        mask = booleans[left_positions]
        left_region = left_region[mask]
        left_index = left_index[mask]
        right_region = np.full_like(right, fill_value=-1)
        right_indexer = right_indexer[booleans]
        right_region[right_indexer] = np.arange(right_indexer.size)
        right_region = right_region[right_positions]
        mask = right_region != -1
        right_region = right_region[mask]
        right_index = right_index[mask]
        return left_region, right_region, left_index, right_index
    # no -1s
    # but left and right may not be in the same order
    # e.g pd.Index([1,2,3]) vs pd.Index([3,1,2])
    indexer = np.arange(left_uniques.size)
    left_region = indexer[left_positions]
    right_region = np.empty(right_uniques.size, dtype=np.intp)
    right_region[right_indexer] = indexer
    right_region = right_region[right_positions]
    return left_index, right_index, left_region, right_region


def _numba_equii_join(df, right, equi_conditions, non_equi_conditions):
    left_indices = []
    left_regions = []
    right_indices = []
    right_regions = []
    left_index = slice(None)
    right_index = slice(None)

    # for left_on, right_on, op in non_equi_conditions:
    #     result = _get_regions_non_equi(
    #         left=df.loc[left_index, left_on],
    #         right=right.loc[right_index, right_on],
    #         op=op,
    #     )
    #     if result is None:
    #         return None
    #     (
    #         left_index,
    #         right_index,
    #         left_region,
    #         right_region,
    #     ) = result

    #     left_index = pd.Index(left_index)
    #     right_index = pd.Index(right_index)
    #     left_indices.append(left_index)
    #     right_indices.append(right_index)
    #     left_regions.append(left_region)
    #     right_regions.append(right_region)

    if len(equi_conditions) == 1:
        left_on, right_on, _ = equi_conditions[0]
        left_arr = df.loc[left_index, left_on]
        any_nulls = left_arr.isna()
        if any_nulls.all():
            return None
        if any_nulls.any():
            left_arr = left_arr[~any_nulls]
        right_arr = right.loc[right_index, right_on]
        any_nulls = right_arr.isna()
        if any_nulls.all():
            return None
        if any_nulls.any():
            right_arr = right_arr[~any_nulls]
        left_index = left_arr.index._values
        right_index = right_arr.index._values
        left_arr = pd.Index(left_arr)
        right_arr = pd.Index(right_arr)
    else:
        left_on, right_on, _ = zip(*equi_conditions)
        left_arr = df.loc[left_index, list(left_on)]
        any_nulls = left_arr.isna().any(axis=1)
        if any_nulls.all():
            return None
        if any_nulls.any(axis=None):
            left_arr = left_arr[~any_nulls]
        right_arr = right.loc[right_index, list(right_on)]
        any_nulls = right_arr.isna().any(axis=1)
        if any_nulls.all():
            return None
        if any_nulls.any(axis=None):
            right_arr = right_arr[~any_nulls]
        left_index = left_arr.index._values
        left_arr = pd.MultiIndex.from_frame(left_arr)
        right_index = right_arr.index._values
        right_arr = pd.MultiIndex.from_frame(right_arr)

    result = _get_regions_equi(
        left=left_arr,
        right=right_arr,
        left_index=left_index,
        right_index=right_index,
    )
    return result
    if result is None:
        return None
    (
        left_index,
        right_index,
        left_region,
        right_region,
    ) = result

    left_index = pd.Index(left_index)
    right_index = pd.Index(right_index)
    left_indices.append(left_index)
    right_indices.append(right_index)
    left_regions.append(left_region)
    right_regions.append(right_region)

    left_index, left_regions = _align_indices_and_regions(
        indices=left_indices, regions=left_regions, sort=False
    )
    right_index, right_regions = _align_indices_and_regions(
        indices=right_indices, regions=right_regions, sort=False
    )

    # return left_regions, right_regions

    left_regions = np.column_stack(left_regions)
    right_regions = np.column_stack(right_regions)

    # return left_regions, right_regions
    left_indices = None
    right_indices = None
    left_arr = left_regions[:, 0]
    right_arr = right_regions[:, 0]
    starts = right_arr.searchsorted(left_arr, side="left")
    ends = right_arr.searchsorted(left_arr, side="right")

    grp = pd.DataFrame(right_regions[:, 1:]).groupby(
        right_regions[:, 0], sort=False
    )
    grp = grp.transform("max").to_numpy(copy=False)
    grp = grp[starts]
    booleans = left_regions[:, 1:] > grp
    booleans = booleans.any(axis=1)
    if booleans.any(axis=None):
        booleans = ~booleans
        left_regions = left_regions[booleans]
        starts = starts[booleans]
        ends = ends[booleans]
        left_index = left_index[booleans]
    if (ends - starts).max() == 1:
        # no need for a comparision
        return left_index, right_index[starts]
    # return left_regions,right_regions, grp,starts,ends
    from pprint import pprint

    pprint([left_regions, right_regions, starts, ends, grp[:, 0]])
    # return left_regions, right_regions, starts, ends
    return _get_indices_equi(
        left_regions=left_regions[:, 1:],
        right_regions=right_regions[:, 1:],
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        max_arr=grp[:, 0],
    )


@njit()
def _get_indices_equi(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    max_arr: np.ndarray,
):
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for equi joins.
    """
    nrows = max_arr.max() + 1
    count_indices = np.empty(starts.size, dtype=np.intp)
    total_length = 0
    previous_start = -1
    ncols = right_regions.shape[1]
    total_length = 0
    for num in range(starts.size):
        start = starts[num]
        end = ends[num]
        arr_max = max_arr[start]
        counter = 0
        if previous_start != start:
            value_counts = np.zeros(nrows, dtype=np.intp)
            positions = np.zeros((nrows, end - start), dtype=np.intp)
            for n in range(start, end):
                region = right_regions[n, 0]
                value_counts[region] += 1
                positions[region, value_counts[region] - 1] = n
        previous_start = start
        for nn in range(left_regions[num, 0], arr_max + 1):
            counts = value_counts[nn]
            if not counts:
                continue
            for sz in range(counts):
                pos = positions[nn, sz]
                status = 1
                for col in range(1, ncols):
                    l_value = left_regions[num, col]
                    r_value = right_regions[pos, col]
                    if l_value > r_value:
                        status = 0
                        break
                counter += status
                total_length += status
        count_indices[num] = counter
    start_indices = np.zeros(starts.size, dtype=np.intp)
    start_indices[1:] = np.cumsum(count_indices)[:-1]
    l_index = np.empty(total_length, dtype=np.intp)
    r_index = np.empty(total_length, np.intp)

    for num in range(starts.size):
        start = starts[num]
        end = ends[num]
        arr_max = max_arr[start]
        size = count_indices[num]
        l_ind = left_index[num]
        indexer = start_indices[num]
        if previous_start != start:
            value_counts = np.zeros(nrows, dtype=np.intp)
            positions = np.zeros((nrows, end - start), dtype=np.intp)
            for n in range(start, end):
                region = right_regions[n, 0]
                value_counts[region] += 1
                positions[region, value_counts[region] - 1] = n
        previous_start = start
        for nn in range(left_regions[num, 0], arr_max + 1):
            if not size:
                break
            counts = value_counts[nn]
            if not counts:
                continue
            for sz in range(counts):
                pos = positions[nn, sz]
                status = 1
                for col in range(1, ncols):
                    l_value = left_regions[num, col]
                    r_value = right_regions[pos, col]
                    if l_value > r_value:
                        status = 0
                        break
                if status:
                    l_index[indexer] = l_ind
                    r_index[indexer] = right_index[pos]
                    indexer += 1
                    size -= 1
                if not size:
                    break
    return l_index, r_index

    return (
        left_regions,
        right_regions,
        starts,
        ends,
        count_indices,
        total_length,
    )
    #         for n in range(start,end):
    #             r_region = right_regions[n, 1]
    #             value_counts[r_region] += 1
    #     #     status = 1
    #     #     for col in range(1, ncols):
    #     #         l_value = left_regions[num, col]
    #     #         r_value = right_regions[n, col]
    #     #         if l_value > r_value:
    #     #             status = 0
    #     #             break
    #     #     counter += status
    #     #     total_length += status
    #     # count_indices[num] = counter
    #     # counts += (counter > 0)
    # return counts, count_indices, total_length
    # if counts < starts.size:
    #     indices = np.empty(counts, dtype=np.intp)
    #     positions = np.empty(counts, dtype=np.intp)
    #     indexer = 0
    #     for num in range(starts.size):
    #         val = count_indices[num]
    #         if not val:
    #             continue
    #         indices[indexer] = val
    #         positions[indexer] = num
    #         indexer += 1
    #         if indexer == counts:
    #             break
    # else:
    #     indices = count_indices[:]
    # start_indices = np.zeros(counts, dtype=np.intp)
    # start_indices[1:] = np.cumsum(indices)[:-1]
    # l_index = np.empty(total_length, dtype=np.intp)
    # r_index = np.empty(total_length, dtype=np.intp)
    # for num in range(counts):
    #     indexer = start_indices[num]
    #     pos = positions[num]
    #     size = count_indices[pos]
    #     start = starts[pos]
    #     end = ends[pos]
    #     l_ind = left_index[pos]
    #     for n in range(start,end):
    #         status = 1
    #         for col in range(1, ncols):
    #             l_value = left_regions[pos, col]
    #             r_value = right_regions[n, col]
    #             if l_value > r_value:
    #                 status = 0
    #                 break
    #         if status:
    #             l_index[indexer] = l_ind
    #             r_index[indexer] = right_index[n]
    #             indexer += 1
    #             size -= 1
    #         if not size:
    #             break

    # return  l_index, r_index


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
    # steal some perf here within the binary search
    # search for uniques
    # and later index them with left_positions
    left_positions, left_arr = df[left_column].factorize(sort=False)
    right_arr = right[right_column]._values
    left_index = df.index._values
    right_index = right.index._values
    slice_starts = right_arr.searchsorted(left_arr, side="left")
    slice_starts = slice_starts[left_positions]
    slice_ends = right_arr.searchsorted(left_arr, side="right")
    slice_ends = slice_ends[left_positions]
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
        ge_arr1, ge_arr2 = _convert_to_numpy(left=ge_arr1, right=ge_arr2)
        ge_strict = True if op == ">" else False

    le_arr1 = None
    le_arr2 = None
    le_strict = None
    if le_lt:
        left_column, right_column, op = le_lt
        le_arr1 = df.loc[left_index, left_column]._values
        le_arr2 = right[right_column]._values
        le_arr1, le_arr2 = _convert_to_numpy(left=le_arr1, right=le_arr2)
        le_strict = True if op == "<" else False

    if le_lt and ge_gt:
        group = right.groupby(eqs[1])[le_lt[1]]
        # is the last column (le_lt) monotonic increasing?
        # fast path if it is
        all_monotonic_increasing = all(
            arr.is_monotonic_increasing for _, arr in group
        )
        if all_monotonic_increasing:
            cum_max_arr = le_arr2[:]
        else:
            cum_max_arr = group.cummax()._values
            if is_extension_array_dtype(cum_max_arr):
                array_dtype = cum_max_arr.dtype.numpy_dtype
                cum_max_arr = cum_max_arr.astype(array_dtype)
            if is_datetime64_dtype(cum_max_arr):
                cum_max_arr = cum_max_arr.view(np.int64)

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
            all_monotonic_increasing,
            cum_max_arr,
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


@njit(parallel=True)
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


@njit(parallel=True)
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


@njit(parallel=True)
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
    all_monotonic_increasing,
    cum_max_arr,
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
            counts += 1
            booleans[num] = False
        else:
            ends[num] = slice_start + end
    if counts == length:
        return None, None

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
    if all_monotonic_increasing:
        sizes = np.empty(length, dtype=np.intp)
    counts = 0
    for num in prange(length):
        l1 = le_arr1[num]
        slice_start = slice_starts[num]
        slice_end = slice_ends[num]
        r1 = cum_max_arr[slice_start:slice_end]
        start = np.searchsorted(r1, l1, side="left")
        if start < r1.size:
            if le_strict and (l1 == r1[start]):
                start = np.searchsorted(r1, l1, side="right")
        if start == r1.size:
            counts += 1
            booleans[num] = False
        else:
            starts[num] = slice_start + start
            if all_monotonic_increasing:
                sizes[num] = r1.size - start
    if counts == length:
        return None, None
    if counts > 0:
        left_index = left_index[booleans]
        le_arr1 = le_arr1[booleans]
        starts = starts[booleans]
        slice_ends = slice_ends[booleans]
        if all_monotonic_increasing:
            sizes = sizes[booleans]

    slice_starts = starts
    starts = None

    # no need to run a comparison
    # since all groups are monotonic increasing
    # simply create left and right indices
    if all_monotonic_increasing:
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


def _numba_single_non_equi_join(
    left: pd.Series, right: pd.Series, op: str, keep: str
) -> tuple:
    """Return matching indices for single non-equi join."""
    if (op == "!=") or (keep != "all"):
        return _generic_func_cond_join(
            left=left, right=right, op=op, multiple_conditions=False, keep=keep
        )

    outcome = _generic_func_cond_join(
        left=left, right=right, op=op, multiple_conditions=True, keep="all"
    )
    if outcome is None:
        return None
    left_index, right_index, starts = outcome
    if op in less_than_join_types:
        counts = right_index.size - starts
    else:
        counts = starts[:]
        starts = np.zeros(starts.size, dtype=np.intp)
    return _get_indices_monotonic_non_equi(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        counts=counts,
    )


def _numba_multiple_non_equi_join(
    df: pd.DataFrame, right: pd.DataFrame, gt_lt: list
):
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
    left_indices = []
    left_regions = []
    right_indices = []
    right_regions = []
    left_index = slice(None)
    right_index = slice(None)
    for left_on, right_on, op in gt_lt[::-1]:
        result = _get_regions_non_equi(
            left=df.loc[left_index, left_on],
            right=right.loc[right_index, right_on],
            op=op,
        )
        if result is None:
            return None
        (
            left_index,
            right_index,
            left_region,
            right_region,
        ) = result
        left_index = pd.Index(left_index)
        right_index = pd.Index(right_index)
        left_indices.append(left_index)
        right_indices.append(right_index)
        left_regions.append(left_region)
        right_regions.append(right_region)

    left_index, left_regions = _align_indices_and_regions(
        indices=left_indices, regions=left_regions
    )
    right_index, right_regions = _align_indices_and_regions(
        indices=right_indices, regions=right_regions
    )

    left_regions = np.column_stack(left_regions)
    right_regions = np.column_stack(right_regions)
    left_region1 = left_regions[:, 0]
    right_region1 = right_regions[:, 0]
    starts = right_region1.searchsorted(left_region1)
    booleans = starts == right_regions.shape[0]
    if booleans.all(axis=None):
        return None
    if booleans.any(axis=None):
        booleans = ~booleans
        starts = starts[booleans]
        left_index = left_index[booleans]
        left_regions = left_regions[booleans]
    # exclude points where the left_region is greater than
    # the max right_region at the search index
    # there is no point keeping those points,
    # since the left region should be <= right region
    # no need to include the first columns in the check,
    # since that is already sorted above with the binary search
    left_regions = left_regions[:, 1:]
    right_regions = right_regions[:, 1:]
    cum_max_arr = np.maximum.accumulate(right_regions[::-1])[::-1]
    booleans = left_regions > cum_max_arr[starts]
    booleans = booleans.any(axis=1)
    if booleans.all(axis=None):
        return None
    if booleans.any(axis=None):
        booleans = ~booleans
        left_regions = left_regions[booleans]
        left_index = left_index[booleans]
        starts = starts[booleans]
    if len(gt_lt) == 2:
        left_region = left_regions[:, 0]
        right_region = right_regions[:, 0]
        is_monotonic = False
        ser = pd.Series(right_region)
        # there is a fast path if is_monotonic is True
        # this is handy especially when there are only
        # two join conditions
        # we can get the matches via binary search
        if ser.is_monotonic_increasing:
            is_monotonic = True
            ends = right_region.searchsorted(left_region, side="left")
            starts = np.maximum(starts, ends)
            ends = right_index.size
        elif ser.is_monotonic_decreasing:
            is_monotonic = True
            ends = right_region[::-1].searchsorted(left_region, side="left")
            ends = right_region.size - ends
        else:
            # get the max end, beyond which left_region is > right_region
            cum_max_arr = cum_max_arr[:, 0]
            # the lowest left_region will have the largest coverage
            # that gives us the end beyond which no left region is <= right_region
            ends = cum_max_arr[::-1].searchsorted(
                left_region.min(), side="left"
            )
            ends = right_region.size - ends
            right_index = right_index[:ends]
            right_region = right_region[:ends]
        if (ends - starts).max() == 1:
            # no point running a comparison op
            # if the width is all 1
            return left_index, right_index[starts]
        if is_monotonic:
            return _get_indices_monotonic_non_equi(
                left_index=left_index,
                right_index=right_index,
                starts=starts,
                counts=ends - starts,
            )
        right_index = right_index[right_region.argsort(kind="stable")]
        return _get_indices_dual_non_monotonic_non_equi(
            left_region=left_region,
            right_region=right_region,
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            max_arr=cum_max_arr,
        )
    # idea here is to iterate on the region
    # that has the lowest number of comparisions
    # for each left region in the right region
    # the region with the lowest number of comparisons
    # is moved to the front of the array
    # within this region, once we get a match,
    # we check the other regions on that same row
    # if they match, we keep the row, if not we discard
    # and move on to the next one
    indices = cum_max_arr[starts] - left_regions
    indices = indices.sum(axis=0)
    indices = indices.argsort()
    left_regions = left_regions[:, indices]
    right_regions = right_regions[:, indices]
    cum_max_arr = cum_max_arr[:, indices[0]]
    max_freq = pd.Series(right_regions[:, 0]).value_counts().iloc[0]
    return _get_indices_multiple_non_equi(
        left_regions=left_regions,
        right_regions=right_regions,
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=right_index.size,
        max_arr=cum_max_arr,
        max_freq=max_freq,
    )


def _get_regions_non_equi(
    left: pd.Series,
    right: pd.Series,
    op: str,
) -> tuple:
    """
    Get the regions for `left` and `right`.
    Strictly for non-equi joins,
    specifically the  >, >= , <, <= operators.

    Returns a tuple of indices and regions.
    """
    outcome = _generic_func_cond_join(
        left=left,
        right=right,
        op=op,
        multiple_conditions=True,
        keep="all",
    )
    if not outcome:
        return None
    left_index, right_index, search_indices = outcome
    if op in greater_than_join_types:
        right_index = right_index[::-1]
        search_indices = right_index.size - search_indices
    right_region = np.zeros(right_index.size, dtype=np.intp)
    right_region[search_indices] = 1
    right_region = right_region.cumsum() - 1
    left_region = right_region[search_indices]
    right_region = right_region[search_indices.min() :]
    right_index = right_index[search_indices.min() :]
    return (
        left_index,
        right_index,
        left_region,
        right_region,
    )


def _align_indices_and_regions(indices, regions, sort=True):
    """
    align the indices and regions
    obtained from _get_regions_non_equi.

    A single index is returned, with the regions
    properly aligned with the index.
    """
    *other_indices, index = indices
    *other_regions, region = regions
    # the first region should be sorted
    # which comes in handy during iteration
    # sorting is not required for an equi join region
    # since it is already sorted
    if sort and not pd.Series(region).is_monotonic_increasing:
        sorter = region.argsort()
        index = index[sorter]
        region = region[sorter]
    outcome = [region]
    for _index, _region in zip(other_indices, other_regions):
        indexer = _index.get_indexer(index)
        booleans = indexer == -1
        if booleans.any():
            indexer = indexer[~booleans]
        _region = _region[indexer]
        outcome.append(_region)
    return index._values, outcome


@njit(parallel=True)
def _get_indices_monotonic_non_equi(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple:
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where at most, only two join conditions are present.
    This is a fast path, compared to
    _get_indices_dual_non_monotonic_non_equi.
    """
    # compute the starting positions in the left index
    # e.g if counts is [2, 4, 6]
    # the starting positions in the left_index(l_index)
    # will be 0, 2, 6, 12
    cumulative_starts = np.cumsum(counts)
    startss = np.zeros(starts.size, dtype=np.intp)
    startss[1:] = cumulative_starts[:-1]
    l_index = np.empty(cumulative_starts[-1], dtype=np.intp)
    r_index = np.empty(cumulative_starts[-1], dtype=np.intp)
    for num in prange(startss.size):
        ind = startss[num]
        size = counts[num]
        l_ind = left_index[num]
        r_indexer = starts[num]
        for n in range(size):
            indexer = ind + n
            l_index[indexer] = l_ind
            r_index[indexer] = right_index[r_indexer + n]
    return l_index, r_index


@njit()
def _get_indices_dual_non_monotonic_non_equi(
    left_region: np.ndarray,
    right_region: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    max_arr: np.ndarray,
    ends: int,
):
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where only two join conditions are present.
    """
    # two step pass
    # first pass gets the length of the final indices
    # uses a variant of counting sort
    # to get the number of matches in the right region
    # per entry in the left region
    value_counts = np.zeros(right_region.max() + 1, dtype=np.intp)
    count_indices = np.empty(starts.size, dtype=np.intp)
    total_length = 0
    previous_start = ends
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[num]
        end = min(previous_start, ends)
        # maximum value at the same position
        # as left_region in right_region
        arr_max = max_arr[start]
        for n in range(start, end):
            r_region = right_region[n]
            value_counts[r_region] += 1
        previous_start = start
        counter = 0
        # search here should include the maximum value
        # within the search space
        # hence the +1 addition
        for nn in range(left_region[num], arr_max + 1):
            counter += value_counts[nn]
        total_length += counter
        count_indices[num] = counter
    # second pass populates the final indices with actual values
    # again, we use a variant of counting sort here
    # this time, we use the cumulative sum of value_counts
    # to get the right_index positions
    # since we already have our right_index sorted
    start_indices = np.zeros(starts.size, dtype=np.intp)
    start_indices[1:] = np.cumsum(count_indices)[:-1]
    l_index = np.empty(total_length, dtype=np.intp)
    r_index = np.empty(total_length, np.intp)
    value_counts = np.cumsum(value_counts)
    counts = np.zeros(right_region.max() + 1, dtype=np.intp)
    previous_start = ends
    for num in range(starts.size - 1, -1, -1):
        start = starts[num]
        end = min(previous_start, ends)
        arr_max = max_arr[start]
        for n in range(start, end):
            r_region = right_region[n]
            counts[r_region] += 1
        previous_start = start
        l_region = left_region[num]
        l_ind = left_index[num]
        indexer = start_indices[num]
        size = count_indices[num] - 1
        for region in range(l_region, arr_max + 1):
            counter = counts[region]
            if not counter:
                continue
            for nnn in range(1, counter + 1):
                pos = value_counts[region] - nnn
                r_ind = right_index[pos]
                l_index[indexer + size] = l_ind
                r_index[indexer + size] = r_ind
                size -= 1
    return l_index, r_index


@njit()
def _get_indices_multiple_non_equi(
    left_regions: np.ndarray,
    right_regions: np.ndarray,
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: int,
    max_arr: np.ndarray,
    max_freq: int,
):
    """
    Retrieves the matching indices
    for the left and right regions.
    Strictly for non-equi joins,
    where the join conditions are more than two.
    """
    # it uses a variant of counting sort,
    # same as _get_indices_dual_non_monotonic_non_equi
    # but with an extra check
    # on the remaining regions per row
    # to ensure they match
    nrows = max_arr.max() + 1
    value_counts = np.zeros(nrows, dtype=np.intp)
    positions = np.zeros((nrows, max_freq), dtype=np.intp)
    count_indices = np.empty(starts.size, dtype=np.intp)
    total_length = 0
    previous_start = ends
    ncols = right_regions.shape[1]
    # positions -----> [0 1 2 3 4 5]
    # left_region ---> [0 1 1 5 5 6]
    # right_region --> [7 7 7 7 7 7]
    # note how the last value in the left_region(6)
    # is less than the previous value(position 4) in the right_region (7)
    # there is no need to search from 5 to 7 for position 4
    # since that has already been captured in the iteration
    # for 6 to 7 (position 5)
    # the iterative search should not capture more than once
    # and only capture relevant values
    # in short, restrict the search space as much as permissible
    # search starts from the bottom/max
    for num in range(starts.size - 1, -1, -1):
        start = starts[num]
        end = min(previous_start, ends)
        # maximum value at the same position
        # as left_region in right_region
        arr_max = max_arr[start]
        for n in range(start, end):
            r_region = right_regions[n, 0]
            value_counts[r_region] += 1
            positions[r_region, value_counts[r_region] - 1] = n
        previous_start = start
        counter = 0
        # search here should include the maximum value
        # within the search space
        # hence the +1 addition
        for nn in range(left_regions[num, 0], arr_max + 1):
            counts = value_counts[nn]
            if not counts:
                continue
            for sz in range(counts):
                pos = positions[nn, sz]
                status = 1
                # check the remaining regions for this row
                # and ensure left_region<=right_region
                # for each one
                for col in range(1, ncols):
                    l_value = left_regions[num, col]
                    r_value = right_regions[pos, col]
                    if l_value > r_value:
                        status = 0
                        break
                counter += status
                total_length += status
        count_indices[num] = counter
    # second pass populates the final indices with actual values
    start_indices = np.zeros(starts.size, dtype=np.intp)
    start_indices[1:] = np.cumsum(count_indices)[:-1]
    l_index = np.empty(total_length, dtype=np.intp)
    r_index = np.empty(total_length, np.intp)
    value_counts = np.zeros(nrows, dtype=np.intp)
    previous_start = ends

    for num in range(starts.size - 1, -1, -1):
        start = starts[num]
        end = min(previous_start, ends)
        arr_max = max_arr[start]
        for n in range(start, end):
            r_region = right_regions[n, 0]
            value_counts[r_region] += 1
            positions[r_region, value_counts[r_region] - 1] = n
        previous_start = start
        size = count_indices[num]
        l_ind = left_index[num]
        indexer = start_indices[num]
        for nn in range(left_regions[num, 0], arr_max + 1):
            if not size:
                break
            counts = value_counts[nn]
            if not counts:
                continue
            for sz in range(counts):
                pos = positions[nn, sz]
                status = 1
                for col in range(1, ncols):
                    l_value = left_regions[num, col]
                    r_value = right_regions[pos, col]
                    if l_value > r_value:
                        status = 0
                        break
                if status:
                    l_index[indexer] = l_ind
                    r_index[indexer] = right_index[pos]
                    indexer += 1
                    size -= 1
                if not size:
                    break

    return l_index, r_index
