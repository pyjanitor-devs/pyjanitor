from __future__ import annotations

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _helpers,
    _le_ge_1_or_more,
    _not_range_join_regions,
    _range_join_default,
    _range_join_regions,
)


def _select_range_bound(candidates: list, df: pd.DataFrame, right: pd.DataFrame):
    """
    Pick the cheapest-looking candidate from a pool of same-direction
    range-join bound candidates (all `<`/`<=`-type, or all `>`/`>=`-type),
    via the same sampling approach as `_le_ge_1_or_more._select_anchor`.
    Ties are broken toward the earliest candidate, so an already-optimal
    choice is left untouched.

    Only called when `candidates` has 2+ entries - see the call site.
    """
    best_cost = None
    best = None
    for candidate in candidates:
        cost = _le_ge_1_or_more._sample_candidate_cost(candidate, df, right)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = candidate
    return best


def _maybe_select_better_range_bounds(
    mapping: dict, df: pd.DataFrame, right: pd.DataFrame
):
    """
    For a range join with 2+ eligible candidates for `le_lt` and/or
    `ge_gt`, replace `_separate_conditions_based_on_op`'s first-supplied
    pick with the most selective one, independently for each bound (see
    issue #1659 - `le_lt` and `ge_gt` are picked independently, since
    both range-join algorithms always process `ge_gt` first and `le_lt`
    second regardless of which candidate either one is).

    Mutates and returns `mapping`. The matched row set is unaffected
    either way (see #1641's invariance proof, which this reuses); only
    performance and, for `keep='all'`, row order can change - so this is
    only worth calling when there's an actual choice to make and
    `keep in ('first', 'last')`, mirroring #1658's own original scoping.
    """
    for bound_key, candidates_key in (
        ("le_lt", "le_lt_candidates"),
        ("ge_gt", "ge_gt_candidates"),
    ):
        candidates = mapping[candidates_key]
        if len(candidates) < 2:
            continue
        current = mapping[bound_key]
        best = _select_range_bound(candidates, df, right)
        if best == current:
            continue
        mapping["le_or_ge"] = [current, *[c for c in mapping["le_or_ge"] if c != best]]
        mapping[bound_key] = best
    return mapping


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    return_matching_indices: bool,
    join_algorithm: str,
) -> tuple:
    """
    Get indices, or aggregates, for multiple conditions,
    where `>/>=` or `</<=` is present
    """
    empty_array = np.array([], dtype=np.intp)
    mapping = _helpers._separate_conditions_based_on_op(conditions=conditions)
    columns = []
    columns.extend(mapping["le_or_ge"])
    columns.append(mapping["le_lt"])
    columns.append(mapping["ge_gt"])
    columns.extend(mapping["equals"])
    columns = filter(None, columns)
    left_columns = set()
    right_columns = set()
    for left_col, right_col, _ in columns:
        left_columns.add(left_col)
        right_columns.add(right_col)
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=left_columns)
    if df is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=right_columns)
    if right is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if mapping["is_range_join"] and keep in ("first", "last"):
        mapping = _maybe_select_better_range_bounds(mapping=mapping, df=df, right=right)
    if not mapping["is_range_join"]:
        if (len(mapping["le_or_ge"]) == 1) or (join_algorithm == "default"):
            return _le_ge_1_or_more._get_indices(
                mapping=mapping,
                df=df,
                right=right,
                return_matching_indices=return_matching_indices,
                keep=keep,
            )
        return _not_range_join_regions._get_indices(
            df=df,
            right=right,
            mapping=mapping,
            return_matching_indices=return_matching_indices,
            keep=keep,
        )
    # is range join
    if join_algorithm == "default":
        return _range_join_default._get_indices(
            mapping=mapping,
            df=df,
            right=right,
            return_matching_indices=return_matching_indices,
            keep=keep,
        )
    return _range_join_regions._get_indices(
        df=df,
        right=right,
        mapping=mapping,
        return_matching_indices=return_matching_indices,
        keep=keep,
    )
