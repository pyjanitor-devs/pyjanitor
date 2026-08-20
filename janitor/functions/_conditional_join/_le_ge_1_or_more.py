import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _binary_search, _helpers


def _evaluate_le_ge_candidate(candidate: tuple, df: pd.DataFrame, right: pd.DataFrame):
    """
    Run the binary search for one `(left_on, right_on, op)` candidate from
    `mapping["le_or_ge"]`.

    Returns `(left_index, starts, ends, right)`, where `right` is sorted by
    `right_on` if it wasn't already monotonic increasing, or `None` if this
    predicate alone has no matches anywhere.
    """
    left_on, right_on, op = candidate
    left_col = df[left_on]
    if not right[right_on].is_monotonic_increasing:
        right = right.sort_values(right_on, ignore_index=False, kind="stable")
    right_col = right[right_on]
    left_array = _helpers._convert_array_to_numpy(array=left_col._values)
    right_array = _helpers._convert_array_to_numpy(array=right_col._values)
    if op == "<":
        outcome = _binary_search._binary_search_lt_first(
            left=left_array, right=right_array, left_index=left_col.index._values
        )
    elif op == "<=":
        outcome = _binary_search._binary_search_le_first(
            left=left_array, right=right_array, left_index=left_col.index._values
        )
    elif op == ">":
        outcome = _binary_search._binary_search_gt_first(
            left=left_array, right=right_array, left_index=left_col.index._values
        )
    else:
        outcome = _binary_search._binary_search_ge_first(
            left=left_array, right=right_array, left_index=left_col.index._values
        )
    if outcome is None:
        return None
    if op in _helpers.less_than_join_types:
        left_index, starts = outcome
        ends = None
    else:
        left_index, ends = outcome
        starts = None
    return left_index, starts, ends, right


def _select_anchor(candidates: list, df: pd.DataFrame, right: pd.DataFrame):
    """
    Evaluate every `le_or_ge` candidate and pick the one whose binary-search
    window is cheapest to use as the anchor.

    Only called when there are 2+ candidates and `keep` is `'first'` or
    `'last'`: for those, the final output is provably invariant to which
    predicate is chosen as anchor (see issue #1641), so this only affects
    performance. Ties are broken toward the earliest candidate in the
    original order, so an already-optimal ordering is left untouched.

    Returns `(best_pos, left_index, starts, ends, right)` for the winning
    candidate, or `None` if any candidate has zero matches anywhere - since
    a match must satisfy every predicate, that makes the whole join empty
    regardless of anchor choice, so evaluation stops at the first such
    candidate.
    """
    best = None
    for pos, candidate in enumerate(candidates):
        result = _evaluate_le_ge_candidate(candidate, df, right)
        if result is None:
            return None
        left_index, starts, ends, sorted_right = result
        if starts is not None:
            cost = (len(sorted_right) - starts).sum()
        else:
            cost = ends.sum()
        if best is None or cost < best[0]:
            best = (cost, pos, left_index, starts, ends, sorted_right)
    _, best_pos, left_index, starts, ends, sorted_right = best
    return best_pos, left_index, starts, ends, sorted_right


def _get_indices(
    mapping: dict,
    df: pd.DataFrame,
    right: pd.DataFrame,
    return_matching_indices: bool,
    keep: str,
):
    empty_array = np.array([], dtype=np.intp)
    candidates = mapping["le_or_ge"]
    if keep == "all" or len(candidates) == 1:
        anchor, *rest = candidates
        result = _evaluate_le_ge_candidate(anchor, df, right)
        if result is None:
            return {"left_index": empty_array, "right_index": empty_array}
        left_index, starts, ends, right = result
    else:
        result = _select_anchor(candidates, df, right)
        if result is None:
            return {"left_index": empty_array, "right_index": empty_array}
        best_pos, left_index, starts, ends, right = result
        rest = [*candidates[:best_pos], *candidates[best_pos + 1 :]]
    rest.extend(mapping["equals"])
    rest.extend(mapping["not_equals"])
    rest = [entry for entry in rest if entry]
    outcome = _helpers._get_positive_matches_conditions(
        df=df,
        right=right,
        conditions=rest,
        left_index=left_index,
        starts=starts,
        ends=ends,
    )
    if outcome is None:
        return {"left_index": empty_array, "right_index": empty_array}
    if return_matching_indices:
        outcome["left_index"] = left_index
        outcome["right_index"] = right.index._values
        outcome["starts"] = starts
        outcome["ends"] = ends
        return outcome
    return _helpers.build_indices_matches(
        left_index=left_index,
        right_index=right.index._values,
        counts_array=outcome["counts_array"],
        starts=starts,
        ends=ends,
        matches=outcome["matches"],
        total=outcome["total"],
        keep=keep,
    )
