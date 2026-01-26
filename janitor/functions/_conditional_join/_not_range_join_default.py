import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _dual_non_equi,
    _helpers,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    return_matching_indices: bool,
    keep: str,
):
    first_condition, second_condition, *rest_ = mapping["le_or_ge"]
    others = (rest_, mapping["equals"], mapping["not_equals"])
    outcome = _dual_non_equi._get_indices(
        df=df,
        right=right,
        first_condition=first_condition,
        second_condition=second_condition,
    )
    rest = []
    for entry in others:
        if not entry:
            continue
        if isinstance(entry, tuple):
            rest.append(entry)
        else:
            rest.extend(entry)
    empty_array = np.array([], dtype=np.intp)
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if return_matching_indices and not rest:
        starts, ends = _dual_non_equi._build_starts_and_ends(
            counts=outcome["counts_array"]
        )
        outcome["starts"] = starts
        outcome["ends"] = ends
        return outcome
    if not rest:
        if keep == "all":
            starts = None
            ends = None
        else:
            starts, ends = _dual_non_equi._build_starts_and_ends(
                counts=outcome["counts_array"]
            )
        return _helpers._build_indices_positions(
            left_index=outcome["left_index"],
            right_index=outcome["right_index"],
            positions=outcome["positions"],
            starts=starts,
            ends=ends,
            counts_array=outcome["counts_array"],
            total=outcome["total"],
            keep=keep,
        )
    starts, ends = _dual_non_equi._build_starts_and_ends(counts=outcome["counts_array"])
    out = _helpers._get_positive_matches_conditions_posns(
        df=df,
        right=right,
        conditions=rest,
        left_index=outcome["left_index"],
        right_index=outcome["right_index"],
        positions=outcome["positions"],
        starts=starts,
        ends=ends,
    )
    if out is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if return_matching_indices:
        out["starts"] = starts
        out["ends"] = ends
        out = outcome | out
        return out
    return _helpers._build_indices_positions(
        left_index=outcome["left_index"],
        right_index=outcome["right_index"],
        positions=out["positions"],
        starts=starts,
        ends=ends,
        counts_array=out["counts_array"],
        total=out["total"],
        keep=keep,
    )
