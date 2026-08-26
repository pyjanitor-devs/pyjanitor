from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _binary_search, _helpers

_RANGE_RMQ_WORK_FACTOR = 8.0


def _range_rmq(
    right_index: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    keep: str,
) -> np.ndarray | None:
    """
    Use the Rust range-query tree only when it is likely to amortize its build.

    ELI5: the current Python path reopens every interval. The Rust tree walks
    the right index once to prepare reusable block answers, so it is useful
    only when the total interval width is several times the right-table size.
    Returning ``None`` keeps all existing fast paths and dtype combinations
    unchanged when the new extension is unavailable or unsuitable.
    """
    if (
        right_index.dtype != np.dtype(np.int64)
        or right_index.size < 32
        or starts.size < 32
    ):
        return None
    total_width = float(np.asarray(ends - starts, dtype=np.int64).sum(dtype=np.float64))
    if total_width <= _RANGE_RMQ_WORK_FACTOR * right_index.size:
        return None
    function_name = (
        "index_starts_and_ends_keep_first_direct"
        if keep == "first"
        else "index_starts_and_ends_keep_last_direct"
    )
    function = getattr(janitor_rs, function_name, None)
    if function is None:
        return None
    return np.asarray(
        function(
            index=right_index,
            starts=np.asarray(starts, dtype=np.int64),
            ends=np.asarray(ends, dtype=np.int64),
        )
    )


def _range_indices(
    df: pd.DataFrame, right: pd.DataFrame, ge_gt: tuple, le_lt: tuple, is_sorted: bool
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
    r_col = right[right_on]
    left_array = _helpers._convert_array_to_numpy(array=l_col._values)
    right_array = _helpers._convert_array_to_numpy(array=r_col._values)
    if op == ">":
        outcome = _binary_search._binary_search_gt_first(
            left=left_array, right=right_array, left_index=l_col.index._values
        )
    elif op == ">=":
        outcome = _binary_search._binary_search_ge_first(
            left=left_array, right=right_array, left_index=l_col.index._values
        )
    if outcome is None:
        return None
    l_index, ends = outcome
    left_on, right_on, op = le_lt
    l_col = df.loc[l_index, left_on]
    r_col = right[right_on]
    if not is_sorted:
        r_col = r_col.cummax()
    left_array = _helpers._convert_array_to_numpy(array=l_col._values)
    right_array = _helpers._convert_array_to_numpy(array=r_col._values)
    if op == "<":
        outcome = _binary_search._binary_search_lt_first(
            left=left_array, right=right_array, left_index=l_index
        )
    elif op == "<=":
        outcome = _binary_search._binary_search_le_first(
            left=left_array, right=right_array, left_index=l_index
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
        right_rmq = _range_rmq(right_index, starts, ends, keep)
        if right_rmq is not None:
            return {"left_index": left_index, "right_index": right_rmq}
        right = [right_index[start:end] for start, end in zip(starts, ends)]
        right = [arr.min() for arr in right]
        return {"left_index": left_index, "right_index": right}
    if keep == "last":
        right_rmq = _range_rmq(right_index, starts, ends, keep)
        if right_rmq is not None:
            return {"left_index": left_index, "right_index": right_rmq}
        right = [right_index[start:end] for start, end in zip(starts, ends)]
        right = [arr.max() for arr in right]
        return {"left_index": left_index, "right_index": right}
    right = [right_index[start:end] for start, end in zip(starts, ends)]
    right = np.concatenate(right)
    left = janitor_rs.repeat_index(
        index=left_index,
        counts=counts,
        length=counts.sum(),
    )
    return {"left_index": left, "right_index": right}
