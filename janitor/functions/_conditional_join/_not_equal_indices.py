# helper functions for !=
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _binary_search
from janitor.functions._conditional_join._helpers import (
    _convert_array_to_numpy,
    _keep_output,
    _null_checks_cond_join,
    _sort_if_not_monotonic,
)


def _not_equal_indices(
    left: pd.Series,
    right: pd.Series,
    keep: str,
) -> dict | None:
    """
    Use binary search to get indices where
    `left` is exactly  not equal to `right`.

    It is a combination of strictly less than
    and strictly greater than indices.
    """

    if keep in {"first", "last"}:
        return _not_equal_keep_one(left=left, right=right, keep=keep)

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
        left_array = _convert_array_to_numpy(array=left._values)
        right_array = _convert_array_to_numpy(array=right._values)
        outcome = _binary_search._binary_search_lt_first(
            left=left_array, right=right_array, left_index=left.index._values
        )
        if outcome is not None:
            len_right = right.size
            lt_left, search_indices = outcome
            lt_right = [right_index[ind:len_right] for ind in search_indices]
            lt_left = [lt_left.repeat(len_right - search_indices)]
        outcome = _binary_search._binary_search_gt_first(
            left=left_array, right=right_array, left_index=left.index._values
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
    left_index, right_index = outcome
    return {"left_index": left_index, "right_index": right_index}


def _not_equal_keep_one(
    left: pd.Series,
    right: pd.Series,
    keep: str,
) -> dict:
    """Return the first or last unequal right position for each left row."""
    dummy = np.array([], dtype=np.intp)
    if left.empty or right.empty:
        return {"left_index": dummy, "right_index": dummy}

    # The first unequal item is either the first item or, when those values
    # match, the first item with a different value. The same observation works
    # backwards for ``keep="last"``, so only two right-side candidates are
    # needed.
    reverse = keep == "last"
    first_offset = -1 if reverse else 0
    first_value = right.iloc[first_offset]
    first_is_null = pd.isna(first_value)

    if first_is_null:
        first_equals_right = right.isna()
        first_equals_left = pd.Series(False, index=left.index)
    else:
        first_equals_right = right.eq(first_value).fillna(False)
        first_equals_left = left.eq(first_value).fillna(False)

    first_equals_right = first_equals_right.to_numpy(dtype=bool, na_value=False)
    different_offsets = np.flatnonzero(~first_equals_right)
    second_position = None
    if different_offsets.size:
        second_offset = different_offsets[-1] if reverse else different_offsets[0]
        second_position = right.index[second_offset]

    left_index = left.index.to_numpy(copy=False)
    first_equals_left = first_equals_left.to_numpy(dtype=bool, na_value=False)
    if second_position is None:
        keep_rows = ~first_equals_left
        left_index = left_index[keep_rows]
        right_index = np.full(left_index.size, right.index[first_offset], dtype=np.intp)
    else:
        right_index = np.where(
            first_equals_left,
            second_position,
            right.index[first_offset],
        ).astype(np.intp, copy=False)

    left_order = _not_equal_left_order(left=left, right=right)
    selected = pd.Index(left_index).get_indexer(left_order)
    selected = selected[selected >= 0]
    left_index = left_index[selected]
    right_index = right_index[selected]

    return {"left_index": left_index, "right_index": right_index}


def _not_equal_left_order(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Return left positions in the order produced by the materialized join."""
    left_nulls = left.isna().to_numpy()
    right_nulls = right.isna().to_numpy()
    seen = np.zeros(left.size, dtype=bool)
    order = []
    nonnull_right = right[~right_nulls]

    if not nonnull_right.empty:
        less_than_max = left.lt(nonnull_right.max()).fillna(False)
        less_than_max = less_than_max.to_numpy(dtype=bool, na_value=False)
        less_than_max = less_than_max & ~left_nulls
        order.append(left.index[less_than_max].to_numpy(copy=False))
        seen |= less_than_max

        greater_than_min = left.gt(nonnull_right.min()).fillna(False)
        greater_than_min = greater_than_min.to_numpy(dtype=bool, na_value=False)
        greater_than_min = greater_than_min & ~left_nulls & ~seen
        order.append(left.index[greater_than_min].to_numpy(copy=False))
        seen |= greater_than_min

        unseen_nulls = left_nulls & ~seen
        order.append(left.index[unseen_nulls].to_numpy(copy=False))
        seen |= unseen_nulls

    if right_nulls.any():
        order.append(left.index[~seen].to_numpy(copy=False))

    return np.concatenate(order)
