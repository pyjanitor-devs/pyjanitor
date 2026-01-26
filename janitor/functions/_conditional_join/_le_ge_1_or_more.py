import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _greater_than_indices,
    _helpers,
    _less_than_indices,
)


def _get_indices(
    mapping: dict,
    df: pd.DataFrame,
    right: pd.DataFrame,
    return_matching_indices: bool,
    keep: str,
):
    empty_array = np.array([], dtype=np.intp)
    (left_on, right_on, op), *rest = mapping["le_or_ge"]
    left_col = df[left_on]
    if not right[right_on].is_monotonic_increasing:
        right = right.sort_values(right_on, ignore_index=False, kind="stable")
    right_col = right[right_on]
    lt_or_le_check = op in _helpers.less_than_join_types
    if lt_or_le_check:
        outcome = _less_than_indices._le_lt_indices(
            left=left_col._values,
            left_index=left_col.index._values,
            right=right_col._values,
            strict=op == "<",
        )
    else:
        outcome = _greater_than_indices._ge_gt_indices(
            left=left_col._values,
            left_index=left_col.index._values,
            right=right_col._values,
            strict=op == ">",
        )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if lt_or_le_check:
        left_index, starts = outcome
        ends = None
    else:
        left_index, ends = outcome
        starts = None
    rest.extend(mapping["equals"])
    rest.extend(mapping["not_equals"])
    outcome = _helpers._get_positive_matches_conditions(
        df=df,
        right=right,
        conditions=rest,
        left_index=left_index,
        starts=starts,
        ends=ends,
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
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
