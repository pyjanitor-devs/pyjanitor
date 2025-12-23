# helper functions for !=
import numpy as np
import pandas as pd

from janitor.functions._conditional_join._greater_than_indices import (
    _ge_gt_indices,
)
from janitor.functions._conditional_join._helpers import (
    _keep_output,
    _null_checks_cond_join,
    _sort_if_not_monotonic,
)
from janitor.functions._conditional_join._less_than_indices import (
    _le_lt_indices,
)


def _not_equal_indices(left: pd.Series, right: pd.Series, keep: str) -> dict | None:
    """
    Use binary search to get indices where
    `left` is exactly  not equal to `right`.

    It is a combination of strictly less than
    and strictly greater than indices.
    """

    dummy = np.array([], dtype=np.intp)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    lt_left = [dummy]
    lt_right = [dummy]
    gt_left = [dummy]
    gt_right = [dummy]
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
        right = right[~any_right_nulls]
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)

    l1_nulls = [l1_nulls, l2_nulls]
    r1_nulls = [r1_nulls, r2_nulls]
    check1 = _null_checks_cond_join(series=left)
    check2 = _null_checks_cond_join(series=right)
    if (check1 is None) or (check2 is None):
        lt_left = [dummy]
        lt_right = [dummy]
    else:
        left, _ = check1
        right, _ = check2
        right, _ = _sort_if_not_monotonic(series=right)
        right_index = right.index._values
        outcome = _le_lt_indices(
            left=left.array,
            left_index=left.index._values,
            right=right.array,
            strict=True,
        )
        if outcome is not None:
            len_right = right.size
            lt_left, search_indices = outcome
            lt_right = [right_index[ind:len_right] for ind in search_indices]
            lt_left = [lt_left.repeat(len_right - search_indices)]
        outcome = _ge_gt_indices(
            left=left.array,
            right=right.array,
            left_index=left.index._values,
            strict=True,
        )
        if outcome is not None:
            gt_left, search_indices = outcome
            gt_right = [right_index[:ind] for ind in search_indices]
            gt_left = [gt_left.repeat(search_indices)]
    lt_left.extend(gt_left)
    lt_left.extend(l1_nulls)
    lt_right.extend(gt_right)
    lt_right.extend(r1_nulls)
    left = np.concatenate(lt_left)
    right = np.concatenate(lt_right)
    if (not left.size) & (not right.size):
        return {
            "left_index": dummy,
            "right_index": dummy,
        }
    outcome = _keep_output(keep, left, right)
    outcome = zip(["left_index", "right_index"], outcome)
    outcome = dict(outcome)
    return outcome
