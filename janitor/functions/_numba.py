"""Various Functions powered by Numba"""

import numpy as np
import pandas as pd
from enum import Enum
from janitor.functions.utils import _convert_to_numpy_array
from numba import njit, prange


class _KeepTypes(Enum):
    """
    List of keep types for conditional_join.
    """

    ALL = "all"
    FIRST = "first"
    LAST = "last"


def _numba_single_join(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    keep: str,
    op_code: int,
) -> tuple:
    """Return matching indices for single non-equi join."""
    if op_code == -1:
        # for the not equal join, we combine indices
        # from strictly less than and strictly greater than indices
        # as well as indices for nulls, if any
        left_nulls, right_nulls = _numba_not_equal_indices(left, right)
        dummy = np.array([], dtype=int)
        result = _numba_less_than_indices(left, right)
        if result is None:
            lt_left = dummy
            lt_right = dummy
        else:
            lt_left, lt_right = _numba_generate_indices_ne(
                *result, strict, keep, op_code=1
            )
        result = _numba_greater_than_indices(left, right)
        if result is None:
            gt_left = dummy
            gt_right = dummy
        else:
            gt_left, gt_right = _numba_generate_indices_ne(
                *result, strict, keep, op_code=0
            )
        left = np.concatenate([lt_left, gt_left, left_nulls])
        right = np.concatenate([lt_right, gt_right, right_nulls])
        if (not left.size) & (not right.size):
            return None
        if keep == _KeepTypes.ALL.value:
            return left, right
        indexer = np.argsort(left)
        left, pos = np.unique(left[indexer], return_index=True)
        if keep == _KeepTypes.FIRST.value:
            right = np.minimum.reduceat(right[indexer], pos)
        else:
            right = np.maximum.reduceat(right[indexer], pos)
        return left, right

    if op_code == 1:
        result = _numba_less_than_indices(left, right)
    else:
        result = _numba_greater_than_indices(left, right)
    if result is None:
        return None
    result = _get_regions(*result, strict, op_code)
    if result is None:
        return None
    left_index, right_index, left_region, right_region = result
    right_index, right_region = _prep_numba_sort_right(
        right_index, right_region
    )
    if keep == _KeepTypes.ALL.value:
        return _numba_single_non_equi(
            left_index, right_index, left_region, right_region
        )
    return _numba_single_non_equi_keep_first_last(
        left_index, right_index, left_region, right_region, keep
    )


def _numba_generate_indices_ne(
    left: np.ndarray,
    left_index: np.ndarray,
    right: np.ndarray,
    right_index: np.ndarray,
    strict: bool,
    keep: str,
    op_code: int,
) -> tuple:
    """
    Generate indices within a not equal join,
    for either greater or less than.
    if op_code is 1, that is a less than operation,
    if op_code is 0, that is a greater than operation.
    """
    dummy = np.array([], dtype=int)
    result = _get_regions(
        left, left_index, right, right_index, strict, op_code
    )
    if result is None:
        return dummy, dummy
    left_index, right_index, left_region, right_region = result
    right_index, right_region = _prep_numba_sort_right(
        right_index, right_region
    )
    if keep == _KeepTypes.ALL.value:
        return _numba_single_non_equi(
            left_index, right_index, left_region, right_region
        )
    return _numba_single_non_equi_keep_first_last(
        left_index, right_index, left_region, right_region, keep
    )


def _prep_numba_sort_right(right_index, right_region):
    """
    Sort right_index and right_region for fast searching
    via binary search.
    """
    bools = np.all(right_region[1:] >= right_region[:-1])
    if not bools:
        indexer = np.lexsort((right_index, right_region))
        right_region = right_region[indexer]
        right_index = right_index[indexer]
    return right_index, right_region


def _numba_not_equal_indices(left_c: pd.Series, right_c: pd.Series) -> tuple:
    """
    Preparatory function for _numba_single_join
    This retrieves the indices for nulls, if any.
    """

    dummy = np.array([], dtype=int)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    any_left_nulls = left_c.isna()
    any_right_nulls = right_c.isna()
    if any_left_nulls.any():
        l1_nulls = left_c.index[any_left_nulls]
        l1_nulls = l1_nulls._values
        r1_nulls = right_c.index
        # avoid NAN duplicates
        if any_right_nulls.any():
            r1_nulls = r1_nulls[~any_right_nulls]
        r1_nulls = r1_nulls._values
        nulls_count = l1_nulls.size
        # blow up nulls to match length of right
        l1_nulls = np.tile(l1_nulls, r1_nulls.size)
        # ensure length of right matches left
        if nulls_count > 1:
            r1_nulls = np.repeat(r1_nulls, nulls_count)
    if any_right_nulls.any():
        r2_nulls = right_c.index[any_right_nulls]
        r2_nulls = r2_nulls._values
        l2_nulls = left_c.index
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)
    l1_nulls = np.concatenate([l1_nulls, l2_nulls])
    r1_nulls = np.concatenate([r1_nulls, r2_nulls])

    return l1_nulls, r1_nulls


def _numba_less_than_indices(
    left: pd.Series,
    right: pd.Series,
) -> tuple:
    """
    Preparatory function for _numba_single_join
    """

    if left.min() > right.max():
        return None
    any_nulls = pd.isna(left)
    if any_nulls.all():
        return None
    if any_nulls.any():
        left = left[~any_nulls]
    if not left.is_monotonic_increasing:
        left = left.sort_values(kind="stable", ascending=True)
    any_nulls = pd.isna(right)
    if any_nulls.all():
        return None
    if any_nulls.any():
        right = right[~any_nulls]
    any_nulls = None
    left_index = left.index._values
    left = left._values
    right_index = right.index._values
    right = right._values
    left, right = _convert_to_numpy_array(left, right)
    return left, left_index, right, right_index


def _numba_greater_than_indices(
    left: pd.Series,
    right: pd.Series,
) -> tuple:
    """
    Preparatory function for _numba_single_join
    """
    if left.max() < right.min():
        return None

    any_nulls = pd.isna(left)
    if any_nulls.all():
        return None
    if any_nulls.any():
        left = left[~any_nulls]
    if not left.is_monotonic_decreasing:
        left = left.sort_values(kind="stable", ascending=False)
    any_nulls = pd.isna(right)
    if any_nulls.all():
        return None
    if any_nulls.any():
        right = right[~any_nulls]

    any_nulls = None
    left_index = left.index._values
    left = left._values
    right_index = right.index._values
    right = right._values

    left, right = _convert_to_numpy_array(left, right)
    return left, left_index, right, right_index


@njit(parallel=True)
def _numba_single_non_equi(
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_region: np.ndarray,
    right_region: np.ndarray,
) -> tuple:
    """
    Generate all indices when keep = `all`.
    Applies only to >, >= , <, <= operators.
    """
    length = left_index.size
    len_right = right_index.size
    counter = 0
    positions = np.empty(length, np.intp)
    # compute the exact length of the new indices
    for num in prange(length):
        val = left_region[num]
        val = np.searchsorted(right_region, val)
        counter += len_right - val
        positions[num] = val
    l_index = np.empty(counter, np.intp)
    r_index = np.empty(counter, np.intp)
    starts = np.empty(length, np.intp)
    # capture the starts and ends for each sub range
    starts[0] = 0
    starts[1:] = np.cumsum(len_right - positions)[:-1]
    # build the actual indices
    for num in prange(length):
        val = left_index[num]
        pos = positions[num]
        posn = starts[num]
        for ind in range(pos, len_right):
            r_index[posn] = right_index[ind]
            l_index[posn] = val
            posn += 1
    return l_index, r_index


@njit(parallel=True)
def _numba_single_non_equi_keep_first_last(
    left_index: np.ndarray,
    right_index: np.ndarray,
    left_region: np.ndarray,
    right_region: np.ndarray,
    keep: str,
) -> tuple:
    """
    Generate all indices when keep = `first` or `last`
    Applies only to >, >= , <, <= operators.
    """
    length = left_index.size
    positions = np.empty(length, np.intp)

    for num in prange(length):
        val = left_region[num]
        val = np.searchsorted(right_region, val)
        positions[num] = val

    l_index = np.empty(length, np.intp)
    r_index = np.empty(length, np.intp)

    len_right = right_index.size
    for num in prange(length):
        val = left_index[num]
        pos = positions[num]
        base_val = right_index[pos]
        for ind in range(pos + 1, len_right):
            value = right_index[ind]
            if keep == "first":
                bool_scalar = value < base_val
            else:
                bool_scalar = value > base_val
            if bool_scalar:
                base_val = value
        l_index[num] = val
        r_index[num] = base_val
    return l_index, r_index


@njit()
def _get_regions(
    left_c: np.ndarray,
    left_index: np.ndarray,
    right_c: np.ndarray,
    right_index: np.ndarray,
    strict: int,
    op_code: int,
) -> tuple:
    """
    Get the regions where left_c and right_c converge.
    Strictly for non-equi joins,
    specifically  -->  >, >= , <, <= operators.
    """
    # The idea is to group values within regions.
    # An example:
    # left_array: [2, 5, 7]
    # right_array: [0, 3, 7]
    # if the join is left_array <= right_array
    # we should have pairs (0), (2,3),(5,7),(5,7)
    # since no value is less than 0, we can discard that
    # our final regions should be
    #  (2,3) --->  0
    #  (5,7) --->  1
    #  (7,7) --->  1
    #  based on the regions, we can see that any value in
    #  region 0 will be less than 1 ---> 2 <= 3, 7
    #  region 1 values are the end ---> 5 <= 7 & 7 <= 7
    #  if the join is left_array >= right_array
    #  then the left_array is sorted in descending order
    #  and the final pairs should be :
    #  (7, 7), (5, 3), (2, 0)
    # our final regions should be
    #  (7,7) --->  0
    #  (5,3) --->  1
    #  (2,0) --->  2
    #  based on the regions, we can see that any value in
    #  region 0 will be greater than 1 and 2 ---> 7 >= 7, 5, 0
    #  region 1 values will be greater than 2 ---> 5 >= 3, 0
    #  region 2 values are the end ----> 2 >= 0
    #  this concept becomes more relevant when two non equi conditions
    #  are present ---> l1 < r1 & l2 > r2
    #  For two non equi conditions, the matches are where
    #  the regions from group A (l1 < r1)
    #  are also lower than the regions from group B (l2 > r2)
    #  This implementation is based on the algorithm outlined here:
    #  https://www.scitepress.org/papers/2018/68268/68268.pdf
    indices = _search_indices(left_c, right_c, strict, op_code)
    left_region = np.empty(left_c.size, dtype=np.intp)
    left_region[:] = -1
    max_indices = indices.max() + 1
    if max_indices < left_index.size:
        left_region = left_region[:max_indices]
        left_index = left_index[:max_indices]
    mask = indices == -1
    if mask.all():
        return None
    if mask.any():
        right_index = right_index[~mask]
        indices = indices[~mask]
    left_region[indices] = 1
    mask = left_region == 1
    left_region[mask] = np.arange(len(set(indices)))
    start = left_region[-1]
    # this is where we spool out the region numbers
    for num in range(left_region.size - 1, -1, -1):
        if left_region[num] != -1:
            start = left_region[num]
        else:
            left_region[num] = start
    right_region = left_region[indices]
    return left_index, right_index, left_region, right_region


@njit(parallel=True)
def _search_indices(
    left_c: np.ndarray, right_c: np.ndarray, strict: int, op_code: int
) -> np.ndarray:
    """
    Get search indices for non-equi joins
    """
    indices = np.empty(right_c.size, dtype=np.intp)
    for num in prange(right_c.size):
        value = right_c[num]
        if strict:
            high = _searchsorted_left(left_c, value, op_code)
        else:
            high = _searchsorted_right(left_c, value, op_code)
        indices[num] = high

    return indices


@njit()
def _searchsorted_left(arr: np.ndarray, value: int, op_code: int) -> int:
    """
    Modification of Python's bisect_left function.
    Used to get the relevant region
    within the _get_regions function.
    """
    high = len(arr)
    low = 0
    while low < high:
        mid = (low + high) // 2
        if op_code:
            check = arr[mid] < value
        else:
            check = arr[mid] > value
        if check:
            low = mid + 1
        else:
            high = mid
    return high - 1


@njit()
def _searchsorted_right(arr: np.ndarray, value: int, op_code: int) -> int:
    """
    Modification of Python's bisect_right function.
    Used to get the relevant region
    within the _get_regions function.
    """
    high = len(arr)
    low = 0
    while low < high:
        mid = (low + high) // 2
        if op_code:
            check = value < arr[mid]
        else:
            check = value > arr[mid]
        if check:
            high = mid
        else:
            low = mid + 1
    return low - 1
