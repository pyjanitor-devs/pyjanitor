from __future__ import annotations

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _equi_join_only,
    _equi_ne_only,
    _equi_not_range_join,
    _equi_range_join,
    _equi_uniq_join,
    _helpers,
    _le_ge_1_or_more,
)


def _select_best_candidate(candidates: list, df: pd.DataFrame, right: pd.DataFrame):
    """
    Pick the cheapest-looking candidate from a pool via the same sampling
    approach as `_le_ge_1_or_more._select_anchor` /
    `_get_indices_non_equi._select_range_bound`. Ties are broken toward
    the earliest candidate, so an already-optimal ordering is left
    untouched.

    Only called when `candidates` has 2+ entries - see the call sites.
    """
    best_cost = None
    best = None
    for candidate in candidates:
        cost = _le_ge_1_or_more._sample_candidate_cost(candidate, df, right)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = candidate
    return best


def _maybe_select_better_equi_predicates(
    mapping: dict, df: pd.DataFrame, right: pd.DataFrame
):
    """
    Selectivity-aware refinement of `mapping["le_or_ge"]`/`["le_lt"]`/
    `["ge_gt"]` for the equi + non-equi dispatch tree (`_get_indices_equi
    .py`), mirroring #1658's `_select_anchor` (for `_equi_not_range_join
    .py`'s single-anchor case) and #1663's `_maybe_select_better_range_
    bounds` (for `_equi_range_join.py`'s two-bound case) - see issue
    #1664. `_equi_uniq_join.py` needs no such refinement: it flattens
    every non-equi predicate into one unordered post-filter set
    regardless of which mapping key it came from, so which candidate
    holds which role never affects it - reordering here is harmless for
    that path, not just unneeded.

    Mutates and returns `mapping`. Matched row content is unaffected
    either way (same invariance as #1641/#1658/#1663); only performance
    - and, for `keep='all'`, row order - can change, so this is only
    worth calling when there's an actual choice and `keep in ('first',
    'last')`, mirroring #1658/#1663's own scoping.
    """
    if mapping["is_range_join"]:
        # _equi_range_join.py consumes le_lt/ge_gt directly and treats
        # any extra le_or_ge candidates as unordered post-filters
        # already (like _equi_uniq_join.py) - only le_lt/ge_gt matter.
        for bound_key, candidates_key in (
            ("le_lt", "le_lt_candidates"),
            ("ge_gt", "ge_gt_candidates"),
        ):
            candidates = mapping[candidates_key]
            if len(candidates) < 2:
                continue
            current = mapping[bound_key]
            best = _select_best_candidate(candidates, df, right)
            if best == current:
                continue
            # `best` is a le_or_ge member being promoted to `bound_key`;
            # `current` demotes into its slot. Order within le_or_ge
            # doesn't matter - it's always applied as an unordered set of
            # independent post-filters - so an in-place swap is enough.
            mapping["le_or_ge"][mapping["le_or_ge"].index(best)] = current
            mapping[bound_key] = best
        return mapping
    # _equi_not_range_join.py picks the first le_or_ge entry as anchor,
    # so unlike the range-join case above, position 0 matters here - swap
    # the winner into it (mutates mapping["le_or_ge"] in place).
    candidates = mapping["le_or_ge"]
    if len(candidates) < 2:
        return mapping
    best = _select_best_candidate(candidates, df, right)
    if best != candidates[0]:
        best_pos = candidates.index(best)
        candidates[0], candidates[best_pos] = candidates[best_pos], candidates[0]
    return mapping


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    return_matching_indices: bool,
) -> tuple:
    """
    Get indices, or aggregates, for multiple conditions,
    where `==` is present
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
    if keep in ("first", "last"):
        mapping = _maybe_select_better_equi_predicates(
            mapping=mapping, df=df, right=right
        )
    try:
        # this section assumes one-to-one or many-to-one
        # no need to capture indices for aggfunc here
        return _equi_uniq_join._get_indices(df=df, right=right, mapping=mapping)
    except pd.errors.InvalidIndexError:
        if mapping["is_range_join"]:
            return _equi_range_join._get_indices(
                df=df,
                right=right,
                mapping=mapping,
                keep=keep,
                return_matching_indices=return_matching_indices,
            )
        if mapping["not_equals"] and not mapping["le_or_ge"]:
            return _equi_ne_only._get_indices(
                df=df,
                right=right,
                mapping=mapping,
                keep=keep,
                return_matching_indices=return_matching_indices,
            )
        if mapping["le_or_ge"]:
            return _equi_not_range_join._get_indices(
                df=df,
                right=right,
                mapping=mapping,
                keep=keep,
                return_matching_indices=return_matching_indices,
            )
        return _equi_join_only._get_indices(
            df=df,
            right=right,
            mapping=mapping,
            return_matching_indices=return_matching_indices,
            keep=keep,
        )
