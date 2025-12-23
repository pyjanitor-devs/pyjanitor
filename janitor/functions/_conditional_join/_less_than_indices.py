# helper functions for </<=
import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import (
    _null_checks_cond_join,
    _sort_if_not_monotonic,
)


def _le_lt_indices(
    left: pd.array,
    left_index: np.ndarray,
    right: pd.array,
    strict: bool,
) -> tuple | None:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.

    Returns the left index and the binary search positions for left in right.
    """
    search_indices = right.searchsorted(left, side="left")
    # if any of the positions in `search_indices`
    # is equal to the length of `right_keys`
    # that means the respective position in `left`
    # has no values from `right` that are less than
    # or equal, and should therefore be discarded
    len_right = right.size
    booleans = search_indices < len_right
    if not booleans.any():
        return None
    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]
    if not strict:
        return left_index, search_indices
    # the idea here is that if there are any equal values
    # shift to the right to the immediate next position
    # that is not equal
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
    if not booleans.any():
        return None
    if not booleans.all():
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]
    return left_index, search_indices


def _less_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    keep: str,
    return_matching_indices: bool,
) -> dict | None:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.
    """
    empty_array = np.array([], dtype=np.intp)
    outcome = _null_checks_cond_join(series=left)
    if not outcome:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left, _ = outcome
    outcome = _null_checks_cond_join(series=right)
    if not outcome:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    right, any_nulls = outcome
    right, right_is_sorted = _sort_if_not_monotonic(series=right)
    outcome = _le_lt_indices(
        left=left.array,
        right=right.array,
        left_index=left.index._values,
        strict=strict,
    )
    if not outcome:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left_index, search_indices = outcome
    len_right = right.size
    right_index = right.index._values
    if right_is_sorted & (keep == "last"):
        indexer = np.empty_like(search_indices)
        indexer[:] = len_right - 1
        return {"left_index": left_index, "right_index": right_index[indexer]}
    if right_is_sorted & (keep == "first") & any_nulls:
        return {
            "left_index": left_index,
            "right_index": right_index[search_indices],
        }
    if right_is_sorted & (keep == "first"):
        return {"left_index": left_index, "right_index": search_indices}
    if keep == "first":
        right = [right_index[ind:len_right] for ind in search_indices]
        right = [arr.min() for arr in right]
        return {"left_index": left_index, "right_index": right}
    if keep == "last":
        right = [right_index[ind:len_right] for ind in search_indices]
        right = [arr.max() for arr in right]
        return {"left_index": left_index, "right_index": right}
    if return_matching_indices:
        return dict(
            left_index=left_index,
            right_index=right_index,
            starts=search_indices,
            ends=np.repeat(len_right, search_indices.size),
        )
    right = [right_index[ind:len_right] for ind in search_indices]
    right = np.concatenate(right)
    counts = len_right - search_indices
    left = janitor_rs.repeat_index(
        index=left_index,
        counts=counts,
        length=counts.sum(),
    )
    return {"left_index": left, "right_index": right}
