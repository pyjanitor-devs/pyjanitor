from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _greater_than_indices,
    _less_than_indices,
)


def _range_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    ge_gt: tuple,
    le_lt: tuple,
) -> dict | None:
    """
    Retrieve index positions for range/interval joins.

    Idea inspired by article:
    https://www.vertica.com/blog/what-is-a-range-join-and-why-is-it-so-fastba-p223413/

    Returns a tuple of (left_index, right_index)
    """
    # summary of code for range join:
    # get the positions where start_left is >/>= start_right
    # then within the positions,
    # get the positions where end_left is </<= end_right
    # this should reduce the search space
    left_on, right_on, op = ge_gt
    l_col = df[left_on]
    outcome = _greater_than_indices._ge_gt_indices(
        left=l_col._values,
        left_index=l_col.index._values,
        right=right[right_on]._values,
        strict=op == ">",
    )
    if outcome is None:
        return None
    l_index, ends = outcome
    left_on, right_on, op = le_lt
    l_col = df.loc[l_index, left_on]
    outcome = _less_than_indices._le_lt_indices(
        left=l_col._values,
        left_index=l_col.index._values,
        right=right[right_on]._values,
        strict=op == "<",
    )
    if outcome is None:
        return None
    left_index, starts = outcome
    if left_index.size < l_index.size:
        keep_rows = pd.Index(left_index).get_indexer(l_index) != -1
        ends = ends[keep_rows]
    # no point searching within (a, b)
    # if a == b
    # since range(a, b) yields none
    keep_rows = starts < ends

    if not keep_rows.any():
        return None

    if not keep_rows.all():
        left_index = left_index[keep_rows]
        starts = starts[keep_rows]
        ends = ends[keep_rows]

    return {"left_index": left_index, "starts": starts, "ends": ends}


def _build_indices(
    left_index: np.ndarray,
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    keep: str,
    right_is_sorted: bool,
    return_matching_indices: bool,
):
    """
    Build indices for a dual range join
    """
    counts = ends - starts
    if counts.max() == 1:
        # no point running a comparison op
        # if the width is all 1
        # this also implies that the intervals
        # do not overlap on the right side
        return {"left_index": left_index, "right_index": right_index[starts]}
    if (keep == "first") and right_is_sorted:
        return {"left_index": left_index, "right_index": right_index[starts]}
    if (keep == "last") and right_is_sorted:
        return {"left_index": left_index, "right_index": right_index[ends - 1]}
    if keep == "first":
        right = [right_index[start:end] for start, end in zip(starts, ends)]
        right = [arr.min() for arr in right]
        return {"left_index": left_index, "right_index": right}
    if keep == "last":
        right = [right_index[start:end] for start, end in zip(starts, ends)]
        right = [arr.max() for arr in right]
        return {"left_index": left_index, "right_index": right}
    if return_matching_indices:
        return dict(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
        )
    right = [right_index[start:end] for start, end in zip(starts, ends)]
    right = np.concatenate(right)
    left = janitor_rs.repeat_index(
        index=left_index,
        counts=counts,
        length=counts.sum(),
    )
    return {"left_index": left, "right_index": right}
