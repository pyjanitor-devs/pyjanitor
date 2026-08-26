import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _binary_search, _helpers

_SAMPLE_SIZE = 1024


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


def _sample_candidate_cost(candidate: tuple, df: pd.DataFrame, right: pd.DataFrame):
    """
    Cheaply estimate one `le_or_ge` candidate's binary-search window cost
    via a fixed-size random sample, so the estimate's cost is independent
    of `len(df)`/`len(right)` - unlike a real binary search, it doesn't
    grow with the size of the join.

    Used by `_select_anchor` to pick which single candidate to fully
    evaluate, instead of fully evaluating every candidate. Being wrong
    only costs performance, never correctness: for `keep='first'`/`'last'`,
    output is provably invariant to anchor choice (see issue #1641), so a
    suboptimal sample-based pick is never a correctness risk.

    Uses a freshly-seeded RNG on every call (not a shared/module-level
    one), so the same inputs always produce the same estimate - a shared
    RNG would make anchor choice, and therefore performance, silently vary
    across otherwise-identical calls.

    Known limitation: a fixed `_SAMPLE_SIZE`-row sample can miss a rare but
    decisive feature - for a feature present in only 0.1% of rows, the
    probability that a uniform sample of `_SAMPLE_SIZE` rows contains none
    of it is `0.999 ** _SAMPLE_SIZE` ~= 36%. Because the RNG is seeded
    deterministically rather than freshly randomized per call, whether a
    *specific* rare feature is captured is fixed by column length, not
    re-rolled on retry - it either always lands in the sample or never
    does, for a given input size. This can only lead to a suboptimal
    anchor choice, never incorrect output.
    """
    left_on, right_on, op = candidate
    left_col = df[left_on]
    right_col = right[right_on]
    rng = np.random.default_rng(0)
    left_count = min(len(left_col), _SAMPLE_SIZE)
    right_count = min(len(right_col), _SAMPLE_SIZE)
    left_positions = rng.choice(len(left_col), size=left_count, replace=False)
    right_positions = rng.choice(len(right_col), size=right_count, replace=False)
    left_array = _helpers._convert_array_to_numpy(
        array=left_col._values[left_positions]
    )
    right_array = np.sort(
        _helpers._convert_array_to_numpy(array=right_col._values[right_positions])
    )
    if op == "<":
        window = right_count - np.searchsorted(right_array, left_array, side="right")
    elif op == "<=":
        window = right_count - np.searchsorted(right_array, left_array, side="left")
    elif op == ">":
        window = np.searchsorted(right_array, left_array, side="left")
    else:
        window = np.searchsorted(right_array, left_array, side="right")
    return float(window.mean())


def _select_anchor(candidates: list, df: pd.DataFrame, right: pd.DataFrame):
    """
    Cheaply sample every `le_or_ge` candidate's window cost and fully
    evaluate only the one with the smallest estimate, instead of fully
    evaluating every candidate.

    Only called when there are 2+ candidates and `keep` is `'first'` or
    `'last'`: for those, the final output is provably invariant to which
    predicate is chosen as anchor (see issue #1641), so a suboptimal
    sample-based pick only affects performance, never correctness. Ties
    are broken toward the earliest candidate in the original order, so an
    already-optimal ordering is left untouched.

    Returns `(best_pos, left_index, starts, ends, right)` for the chosen
    candidate, or `None` if it has zero matches anywhere. (If some other,
    non-chosen candidate is the one with zero matches, the whole join is
    still correctly detected as empty downstream, once its predicate is
    applied as a post-filter in `_get_indices`.)
    """
    best_cost = None
    best_pos = None
    for pos, candidate in enumerate(candidates):
        cost = _sample_candidate_cost(candidate, df, right)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_pos = pos
    result = _evaluate_le_ge_candidate(candidates[best_pos], df, right)
    if result is None:
        return None
    left_index, starts, ends, right = result
    return best_pos, left_index, starts, ends, right


def _get_indices(
    mapping: dict,
    df: pd.DataFrame,
    right: pd.DataFrame,
    return_matching_indices: bool,
    keep: str,
):
    """
    Get indices for one or more `>/>=`/`</<=` conditions, no range join.
    """
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
