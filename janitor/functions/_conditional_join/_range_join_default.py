import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers, _range_indices


def _get_indices(
    mapping: dict,
    df: pd.DataFrame,
    right: pd.DataFrame,
    return_matching_indices: bool,
    keep: str,
):
    empty_array = np.array([], dtype=np.intp)
    (_, r1_col, _) = mapping["ge_gt"]
    (_, r2_col, _) = mapping["le_lt"]
    rest = []
    rest.extend(mapping["le_or_ge"])
    rest.extend(mapping["equals"])
    rest.extend(mapping["not_equals"])
    rest = [entry for entry in rest if entry]
    right_is_sorted = right[r1_col].is_monotonic_increasing
    if not right_is_sorted:
        # defensive approach,
        # in case of duplicates
        sorter = dict.fromkeys([r1_col, r2_col])
        sorter = [*sorter]
        right = right.sort_values(sorter, kind="stable", ignore_index=False)
    is_sorted = right[r2_col].is_monotonic_increasing
    if not is_sorted:
        rest = [mapping["le_lt"]] + rest
    outcome = _range_indices._range_indices(
        df=df,
        right=right,
        ge_gt=mapping["ge_gt"],
        le_lt=mapping["le_lt"],
        is_sorted=is_sorted,
    )
    if outcome is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if return_matching_indices and not rest:
        outcome["right_index"] = right.index._values
        return outcome
    if not rest:
        return _range_indices._build_indices(
            left_index=outcome["left_index"],
            right_index=right.index._values,
            starts=outcome["starts"],
            ends=outcome["ends"],
            keep=keep,
            right_is_sorted=right_is_sorted,
        )
    direct = _helpers._get_direct_indices_conditions(
        df=df,
        right=right,
        conditions=rest,
        left_index=outcome["left_index"],
        starts=outcome["starts"],
        ends=outcome["ends"],
        keep=keep,
    )
    if direct is not None:
        return direct
    out = _helpers._get_positive_matches_conditions(
        df=df,
        right=right,
        conditions=rest,
        left_index=outcome["left_index"],
        starts=outcome["starts"],
        ends=outcome["ends"],
    )
    if out is None:
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    if return_matching_indices:
        out = outcome | out
        out["right_index"] = right.index._values
        return out
    return _helpers.build_indices_matches(
        left_index=outcome["left_index"],
        right_index=right.index._values,
        counts_array=out["counts_array"],
        starts=outcome["starts"],
        ends=outcome["ends"],
        matches=out["matches"],
        total=out["total"],
        keep=keep,
    )
