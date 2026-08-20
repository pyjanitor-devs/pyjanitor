# helper functions for >/>=
import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _binary_search
from janitor.functions._conditional_join._helpers import (
    _accumulate_keep_positions,
    _convert_array_to_numpy,
    _null_checks_cond_join,
    _sort_if_not_monotonic,
)


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
    left_array = _convert_array_to_numpy(array=left._values)
    right_array = _convert_array_to_numpy(array=right._values)
    if strict:
        outcome = _binary_search._binary_search_gt_first(
            left=left_array, right=right_array, left_index=left.index._values
        )
    else:
        outcome = _binary_search._binary_search_ge_first(
            left=left_array, right=right_array, left_index=left.index._values
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
    if keep in {"first", "last"}:
        # ELI5: each match is a prefix of this array. One forward pass remembers
        # the earliest/latest original row for every prefix.
        right = _accumulate_keep_positions(right_index, keep)
        return {
            "left_index": left_index,
            "right_index": right[search_indices - 1],
        }
    if return_matching_indices:
        return dict(
            left_index=left_index,
            right_index=right_index,
            starts=0,
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
