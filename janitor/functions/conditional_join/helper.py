from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _null_checks_cond_join(left: pd.Series, right: pd.Series) -> tuple | None:
    """
    Checks for nulls in the arrays before conducting binary search.

    Relevant to _less_than_indices and _greater_than_indices
    """
    any_nulls = left.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        left = left[~any_nulls]
    any_nulls = right.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        right = right[~any_nulls]
    any_nulls = any_nulls.any()
    right_is_sorted = right.is_monotonic_increasing
    if not right_is_sorted:
        right = right.sort_values(kind="stable")

    left_index = left.index._values
    left = left._values
    right_index = right.index._values
    right = right._values

    return left, right, left_index, right_index, right_is_sorted, any_nulls


def _equal_indices(
    left: pd.Series, right: pd.Series, return_ragged_arrays: bool
) -> tuple | None:
    """
    Use binary search to get indices where left
    is equal to right.

    A tuple of integer indexes
    for left and right is returned.
    """
    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome
    # steal some perf here within the binary search
    # search for uniques
    # and later index them with left_positions
    # it is assumed that users will only reach for this
    # if the data is reasonably duplicated; if not
    # pd.merge is superb especially if it's a one-to-one
    # or one-to-many
    positions, left = pd.factorize(left, sort=False)
    if return_ragged_arrays:
        starts = right.searchsorted(left, side="left")
        starts = starts[positions]
        ends = right.searchsorted(left, side="right")
        ends = ends[positions]
        booleans = starts < ends
        if not booleans.any():
            return None
        if not booleans.all():
            left_index = left_index[booleans]
            starts = starts[booleans]
            ends = ends[booleans]
        right = [slice(start, end) for start, end in zip(starts, ends)]
        if right_is_sorted & (not any_nulls):
            return left_index, right
        right = [right_index[slicer] for slicer in right]
        return left_index, right
    # necessary step to remove non matches in right
    # vital to ensuring correct output in numba_equi_join
    # when building the regions
    booleans = pd.Index(left).get_indexer(right) != -1
    if not booleans.any():
        return None
    if not booleans.all():
        right_index = right_index[booleans]
        right = right[booleans]
    starts = right.searchsorted(left, side="left")
    starts = starts[positions]
    ends = right.searchsorted(left, side="right")
    ends = ends[positions]
    booleans = starts < ends
    if not booleans.any():
        return None
    if not booleans.all():
        left_index = left_index[booleans]
        starts = starts[booleans]
    return left_index, right_index, starts


def _less_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    multiple_conditions: bool,
    keep: str,
    return_ragged_arrays: bool,
) -> tuple | None:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.

    A tuple of integer indexes
    for left and right is returned.
    """

    # no point going through all the hassle
    if left.min() > right.max():
        return None

    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome

    search_indices = right.searchsorted(left, side="left")
    # if any of the positions in `search_indices`
    # is equal to the length of `right_keys`
    # that means the respective position in `left`
    # has no values from `right` that are less than
    # or equal, and should therefore be discarded
    len_right = right.size
    booleans = search_indices < len_right

    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]

    # the idea here is that if there are any equal values
    # shift to the right to the immediate next position
    # that is not equal
    if strict:
        booleans = left == right[search_indices]
        # replace positions where rows are equal
        # with positions from searchsorted('right')
        # positions from searchsorted('right') will never
        # be equal and will be the furthermost in terms of position
        # example : right -> [2, 2, 2, 3], and we need
        # positions where values are not equal for 2;
        # the furthermost will be 3, and searchsorted('right')
        # will return position 3.
        if booleans.any():
            replacements = right.searchsorted(left, side="right")
            # now we can safely replace values
            # with strictly less than positions
            search_indices = np.where(booleans, replacements, search_indices)
        # check again if any of the values
        # have become equal to length of right
        # and get rid of them
        booleans = search_indices < len_right

        if not booleans.all():
            left_index = left_index[booleans]
            search_indices = search_indices[booleans]

        if not search_indices.size:
            return None
    if multiple_conditions:
        return left_index, right_index, search_indices
    if right_is_sorted & (keep == "last"):
        indexer = np.empty_like(search_indices)
        indexer[:] = len_right - 1
        return left_index, right_index[indexer]
    if right_is_sorted & (keep == "first") & any_nulls:
        return left_index, right_index[search_indices]
    if right_is_sorted & (keep == "first"):
        return left_index, search_indices
    if return_ragged_arrays & right_is_sorted & (not any_nulls):
        right = [slice(ind, len_right) for ind in search_indices]
        return left_index, right
    right = [right_index[ind:len_right] for ind in search_indices]
    if return_ragged_arrays:
        return left_index, right
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(len_right - search_indices)
    return left, right


def _greater_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    multiple_conditions: bool,
    keep: str,
    return_ragged_arrays: bool,
) -> tuple | None:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.

    if multiple_conditions is False, a tuple of integer indexes
    for left and right is returned;
    else a tuple of the index for left, right, as well
    as the positions of left in right is returned.
    """

    # quick break, avoiding the hassle
    if left.max() < right.min():
        return None

    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome
    search_indices = right.searchsorted(left, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0 (less than 1), it implies that
    # left[position] is not greater than any value
    # in right
    booleans = search_indices > 0
    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]

    # the idea here is that if there are any equal values
    # shift downwards to the immediate next position
    # that is not equal
    if strict:
        booleans = left == right[search_indices - 1]
        # replace positions where rows are equal with
        # searchsorted('left');
        # this works fine since we will be using the value
        # as the right side of a slice, which is not included
        # in the final computed value
        if booleans.any():
            replacements = right.searchsorted(left, side="left")
            # now we can safely replace values
            # with strictly greater than positions
            search_indices = np.where(booleans, replacements, search_indices)
        # any value less than 1 should be discarded
        # since the lowest value for binary search
        # with side='right' should be 1
        booleans = search_indices > 0
        if not booleans.all():
            left_index = left_index[booleans]
            search_indices = search_indices[booleans]

        if not search_indices.size:
            return None
    if multiple_conditions:
        return left_index, right_index, search_indices
    if right_is_sorted & (keep == "first"):
        indexer = np.zeros_like(search_indices)
        return left_index, right_index[indexer]
    if right_is_sorted & (keep == "last") & any_nulls:
        return left_index, right_index[search_indices - 1]
    if right_is_sorted & (keep == "last"):
        return left_index, search_indices - 1
    if return_ragged_arrays & right_is_sorted & (not any_nulls):
        right = [slice(0, ind) for ind in search_indices]
        return left_index, right
    right = [right_index[:ind] for ind in search_indices]
    if return_ragged_arrays:
        return left_index, right
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(search_indices)
    return left, right


def _not_equal_indices(
    left: pd.Series, right: pd.Series, keep: str
) -> tuple | None:
    """
    Use binary search to get indices where
    `left` is exactly  not equal to `right`.

    It is a combination of strictly less than
    and strictly greater than indices.

    A tuple of integer indexes for left and right
    is returned.
    """

    dummy = np.array([], dtype=int)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    any_left_nulls = left.isna()
    any_right_nulls = right.isna()
    if any_left_nulls.any():
        l1_nulls = left.index[any_left_nulls.array]
        l1_nulls = l1_nulls.to_numpy(copy=False)
        r1_nulls = right.index
        # avoid NAN duplicates
        if any_right_nulls.any():
            r1_nulls = r1_nulls[~any_right_nulls.array]
        r1_nulls = r1_nulls.to_numpy(copy=False)
        nulls_count = l1_nulls.size
        # blow up nulls to match length of right
        l1_nulls = np.tile(l1_nulls, r1_nulls.size)
        # ensure length of right matches left
        if nulls_count > 1:
            r1_nulls = np.repeat(r1_nulls, nulls_count)
    if any_right_nulls.any():
        r2_nulls = right.index[any_right_nulls.array]
        r2_nulls = r2_nulls.to_numpy(copy=False)
        l2_nulls = left.index
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)

    l1_nulls = np.concatenate([l1_nulls, l2_nulls])
    r1_nulls = np.concatenate([r1_nulls, r2_nulls])

    outcome = _less_than_indices(
        left,
        right,
        strict=True,
        multiple_conditions=False,
        keep=keep,
        return_ragged_arrays=False,
    )

    if outcome is None:
        lt_left = dummy
        lt_right = dummy
    else:
        lt_left, lt_right = outcome

    outcome = _greater_than_indices(
        left,
        right,
        strict=True,
        multiple_conditions=False,
        keep=keep,
        return_ragged_arrays=False,
    )

    if outcome is None:
        gt_left = dummy
        gt_right = dummy
    else:
        gt_left, gt_right = outcome

    left = np.concatenate([lt_left, gt_left, l1_nulls])
    right = np.concatenate([lt_right, gt_right, r1_nulls])

    if (not left.size) & (not right.size):
        return None
    return _keep_output(keep, left, right)


def _generic_func_cond_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    multiple_conditions: bool,
    keep: str,
    return_ragged_arrays: bool = False,
) -> tuple:
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices,
    or _not_equal_indices).
    """
    strict = False

    if op in {
        _JoinOperator.GREATER_THAN.value,
        _JoinOperator.LESS_THAN.value,
        _JoinOperator.NOT_EQUAL.value,
    }:
        strict = True

    if op in less_than_join_types:
        return _less_than_indices(
            left=left,
            right=right,
            strict=strict,
            multiple_conditions=multiple_conditions,
            keep=keep,
            return_ragged_arrays=return_ragged_arrays,
        )
    if op in greater_than_join_types:
        return _greater_than_indices(
            left=left,
            right=right,
            strict=strict,
            multiple_conditions=multiple_conditions,
            keep=keep,
            return_ragged_arrays=return_ragged_arrays,
        )
    if op == _JoinOperator.NOT_EQUAL.value:
        return _not_equal_indices(left=left, right=right, keep=keep)
    return _equal_indices(
        left=left, right=right, return_ragged_arrays=return_ragged_arrays
    )


def _keep_output(keep: str, left: np.ndarray, right: np.ndarray) -> tuple:
    """return indices for left and right index based on the value of `keep`."""
    if keep == "all":
        return left, right
    grouped = pd.Series(right).groupby(left)
    if keep == "first":
        grouped = grouped.min()
        return grouped.index, grouped._values
    grouped = grouped.max()
    return grouped.index, grouped._values
