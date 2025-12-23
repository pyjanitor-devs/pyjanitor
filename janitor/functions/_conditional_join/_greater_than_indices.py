# helper functions for >/>=
import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._helpers import (
    _null_checks_cond_join,
    _sort_if_not_monotonic,
)


def _ge_gt_indices(
    left: pd.array,
    left_index: np.ndarray,
    right: pd.array,
    strict: bool,
) -> tuple | None:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.
    """
    search_indices = right.searchsorted(left, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0 (less than 1), it implies that
    # left[position] is not greater than any value
    # in right
    booleans = search_indices > 0
    if not booleans.any():
        return None
    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]
    if not strict:
        return left_index, search_indices
    # the idea here is that if there are any equal values
    # shift downwards to the immediate next position
    # that is not equal
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
    if not booleans.any():
        return None
    if not booleans.all():
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]
    return left_index, search_indices


def _greater_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    keep: str,
    return_matching_indices: bool,
) -> dict | None:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.
    """
    empty_array = np.array([], dtype=np.intp)
    outcome = _null_checks_cond_join(series=left)
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left, _ = outcome
    outcome = _null_checks_cond_join(series=right)
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    right, any_nulls = outcome
    right, right_is_sorted = _sort_if_not_monotonic(series=right)
    outcome = _ge_gt_indices(
        left=left.array,
        right=right.array,
        left_index=left.index._values,
        strict=strict,
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left_index, search_indices = outcome
    right_index = right.index._values
    if right_is_sorted & (keep == "first"):
        indexer = np.zeros_like(search_indices)
        return {"left_index": left_index, "right_index": right_index[indexer]}
    if right_is_sorted & (keep == "last") & any_nulls:
        return {
            "left_index": left_index,
            "right_index": right_index[search_indices - 1],
        }
    if right_is_sorted & (keep == "last"):
        return {"left_index": left_index, "right_index": search_indices - 1}
    if keep == "first":
        right = [right_index[:ind] for ind in search_indices]
        right = [arr.min() for arr in right]
        return {"left_index": left_index, "right_index": right}
    if keep == "last":
        right = [right_index[:ind] for ind in search_indices]
        right = [arr.max() for arr in right]
        return {"left_index": left_index, "right_index": right}
    if return_matching_indices:
        return dict(
            left_index=left_index,
            right_index=right_index,
            starts=np.repeat(0, search_indices.size),
            ends=search_indices,
        )
    right = [right_index[:ind] for ind in search_indices]
    right = np.concatenate(right)
    left = janitor_rs.repeat_index(
        index=left_index,
        counts=search_indices,
        length=search_indices.sum(),
    )
    return {"left_index": left, "right_index": right}
