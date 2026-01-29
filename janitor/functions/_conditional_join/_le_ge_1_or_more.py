import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _binary_search, _helpers


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
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if op in _helpers.less_than_join_types:
        left_index, starts = outcome
        ends = None
    else:
        left_index, ends = outcome
        starts = None
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
