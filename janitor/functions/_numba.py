"""Various Functions powered by Numba"""

import numpy as np
import pandas as pd
from enum import Enum
from janitor.functions.utils import _convert_to_numpy_array
from numba import njit, prange as loop_range


class _KeepTypes(Enum):
    """
    List of keep types for conditional_join.
    """

    ALL = "all"
    FIRST = "first"
    LAST = "last"


def _numba_single_join(left_c, right_c, strict, keep, op_code):
    """Return matching indices for single non-equi join."""
    if op_code == -1:
        left_nulls, right_nulls = _numba_not_equal_indices(left_c, right_c)
        dummy = np.array([], dtype=int)
        result = _numba_less_than_indices(left_c, right_c)
        if result is None:
            lt_left = dummy
            lt_right = dummy
        else:
            lt_left, lt_right = _numba_generate_indices_ne(
                *result, strict, keep, op_code=1
            )
        result = _numba_greater_than_indices(left_c, right_c)
        if result is None:
            gt_left = dummy
            gt_right = dummy
        else:
            gt_left, gt_right = _numba_generate_indices_ne(
                *result, strict, keep, op_code=0
            )
        left_c = np.concatenate([lt_left, gt_left, left_nulls])
        right_c = np.concatenate([lt_right, gt_right, right_nulls])
        if (not left_c.size) & (not right_c.size):
            return None
        if keep == _KeepTypes.ALL.value:
            return left_c, right_c
        indexer = np.argsort(left_c)
        left_c, pos = np.unique(left_c[indexer], return_index=True)
        if keep == _KeepTypes.FIRST.value:
            right_c = np.minimum.reduceat(right_c[indexer], pos)
        else:
            right_c = np.maximum.reduceat(right_c[indexer], pos)
        return left_c, right_c

    if op_code == 1:
        result = _numba_less_than_indices(left_c, right_c)
    else:
        result = _numba_greater_than_indices(left_c, right_c)
    if result is None:
        return None
    result = _get_regions(*result, strict, op_code)
    if result is None:
        return None
    if keep == _KeepTypes.ALL.value:
        return _numba_single_non_equi(*result)
    left_index, right_index, left_region, right_region = result
    right_index, right_region = _prep_numba_first_last(
        right_index, right_region
    )
    return _numba_single_non_equi_keep_first_last(
        left_index, right_index, left_region, right_region, keep
    )


def _numba_generate_indices_ne(
    left_c, left_index, right_c, right_index, strict, keep, op_code
):
    """
    Generate indices for either greater or less than.
    """
    dummy = np.array([], dtype=int)
    result = _get_regions(
        left_c, left_index, right_c, right_index, strict, op_code
    )
    if result is None:
        return dummy, dummy
    if keep == _KeepTypes.ALL.value:
        return _numba_single_non_equi(*result)
    left_index, right_index, left_region, right_region = result
    right_index, right_region = _prep_numba_first_last(
        right_index, right_region
    )
    return _numba_single_non_equi_keep_first_last(
        left_index, right_index, left_region, right_region, keep
    )


def _prep_numba_first_last(right_index, right_region):
    """
    Preparatory function if keep = 'first'/'last'
    """
    if not pd.Series(right_region).is_monotonic_increasing:
        indexer = np.lexsort((right_index, right_region))
        right_region = right_region[indexer]
        right_index = right_index[indexer]
    return right_index, right_region


def _numba_not_equal_indices(left_c: pd.Series, right_c: pd.Series) -> tuple:
    """
    Preparatory function for _numba_single_join
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
    left_c: pd.Series,
    right_c: pd.Series,
) -> tuple:
    """
    Preparatory function for _numba_single_join
    """

    if left_c.min() > right_c.max():
        return None
    any_nulls = pd.isna(left_c)
    if any_nulls.all():
        return None
    if any_nulls.any():
        left_c = left_c[~any_nulls]
    if not left_c.is_monotonic_increasing:
        left_c = left_c.sort_values(kind="stable", ascending=True)
    any_nulls = pd.isna(right_c)
    if any_nulls.all():
        return None
    if any_nulls.any():
        right_c = right_c[~any_nulls]
    any_nulls = None
    left_index = left_c.index._values
    left_c = left_c._values
    right_index = right_c.index._values
    right_c = right_c._values
    left_c, right_c = _convert_to_numpy_array(left_c, right_c)
    return left_c, left_index, right_c, right_index


def _numba_greater_than_indices(
    left_c: pd.Series,
    right_c: pd.Series,
) -> tuple:
    """
    Preparatory function for _numba_single_join
    """
    if left_c.max() < right_c.min():
        return None

    any_nulls = pd.isna(left_c)
    if any_nulls.all():
        return None
    if any_nulls.any():
        left_c = left_c[~any_nulls]
    if not left_c.is_monotonic_decreasing:
        left_c = left_c.sort_values(kind="stable", ascending=False)
    any_nulls = pd.isna(right_c)
    if any_nulls.all():
        return None
    if any_nulls.any():
        right_c = right_c[~any_nulls]

    any_nulls = None
    left_index = left_c.index._values
    left_c = left_c._values
    right_index = right_c.index._values
    right_c = right_c._values

    left_c, right_c = _convert_to_numpy_array(left_c, right_c)
    return left_c, left_index, right_c, right_index


@njit(parallel=True)
def _numba_single_non_equi(left_index, right_index, left_region, right_region):
    """
    Generate all indices when keep = `all`.
    Applies only to >, >= , <, <= operators.
    """
    length = right_index.size
    counter = 0
    positions = np.empty(length, np.intp)
    # compute the exact length of the new indices
    for num in loop_range(length):
        val = right_region[num]
        # left region is always sorted, take advantage of that
        val = np.searchsorted(left_region, val, side="right")
        counter += val
        positions[num] = val
    l_index = np.empty(counter, np.intp)
    r_index = np.empty(counter, np.intp)
    starts = np.empty(length, np.intp)
    # capture the starts and ends for each sub range
    starts[0] = 0
    starts[1:] = np.cumsum(positions)[:-1]
    # build the actual indices
    for num in loop_range(length):
        val = right_index[num]
        pos = positions[num]
        posn = starts[num]
        for ind in range(pos):
            l_index[posn] = left_index[ind]
            r_index[posn] = val
            posn += 1
    return l_index, r_index


@njit(parallel=True)
def _numba_single_non_equi_keep_first_last(
    left_index, right_index, left_region, right_region, keep
):
    """
    Generate all indices when keep = `first` or `last`
    Applies only to >, >= , <, <= operators.
    """
    length = left_index.size
    positions = np.empty(length, np.intp)

    for num in loop_range(length):
        val = left_region[num]
        val = np.searchsorted(right_region, val)
        positions[num] = val

    l_index = np.empty(length, np.intp)
    r_index = np.empty(length, np.intp)

    len_right = right_index.size
    for num in loop_range(length):
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
):
    """
    Get the regions where left_c and right_c converge.
    Strictly for non-equi joins,
    specifically  -->  >, >= , <, <= operators.
    """
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
):
    """
    Get search indices for non-equi joins
    """
    indices = np.empty(right_c.size, dtype=np.intp)
    for num in loop_range(right_c.size):
        value = right_c[num]
        if strict:
            high = _searchsorted_left(left_c, value, op_code)
        else:
            high = _searchsorted_right(left_c, value, op_code)
        indices[num] = high

    return indices


@njit()
def _searchsorted_left(arr, value, op_code):
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
def _searchsorted_right(arr, value, op_code):
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
