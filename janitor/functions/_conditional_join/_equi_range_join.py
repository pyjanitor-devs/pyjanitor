from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _equi_helpers,
    _helpers,
    _range_indices,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    keep: str,
    return_matching_indices: bool,
) -> dict:
    """Get indices if '>/>=' and '</<=' present"""
    empty_array = np.array([], dtype=np.intp)
    (_, r1_col, _) = mapping["ge_gt"]
    (_, r2_col, _) = mapping["le_lt"]
    is_sorted = _equi_helpers._check_sorted_within_groups(
        equi_cols=mapping["equals"],
        right=right,
        r_col=r1_col,
    )
    right = _equi_helpers._maybe_sort_right(
        right=right, r1_col=r1_col, r2_col=r2_col, is_sorted=is_sorted
    )
    l_cols, r_cols = _equi_helpers._build_equi_indices(
        df=df, right=right, mapping=mapping
    )
    positions, uniques, starts, ends = _equi_helpers._get_positions_right(
        right_columns=r_cols
    )
    check = r_cols.is_monotonic_increasing
    if not check:
        reordered_positions = janitor_rs.reorder_index(
            positions=positions, starts=starts
        )
        right = right.iloc[reordered_positions]
    outcome = _equi_helpers._get_indexers(
        l_cols=l_cols, uniques=uniques, starts=starts, ends=ends
    )
    if outcome is None:
        return {"left_index": empty_array, "right_index": empty_array}
    _, starts, ends = outcome
    (l1_col, r1_col, op) = mapping["ge_gt"]
    r_column = right[r1_col]._values
    r_column = _helpers._convert_array_to_numpy(array=r_column)
    l_column = df[l1_col]._values
    l_column = _helpers._convert_array_to_numpy(array=l_column)
    ends = _equi_helpers._update_positions_ge_gt(
        op=op,
        l_column=l_column,
        r_column=r_column,
        starts=starts,
        ends=ends,
    )
    (l2_col, r2_col, op) = mapping["le_lt"]
    is_sorted = _equi_helpers._check_sorted_within_groups(
        equi_cols=mapping["equals"],
        right=right,
        r_col=r2_col,
    )
    # if possible, run binary search on both sides
    #  - ge_gt and le_lt
    if is_sorted:
        r_column = right[r2_col]._values
        r_column = _helpers._convert_array_to_numpy(array=r_column)
        l_column = df[l2_col]._values
        l_column = _helpers._convert_array_to_numpy(array=l_column)
        starts = _equi_helpers._update_positions_le_lt(
            op=op,
            l_column=l_column,
            r_column=r_column,
            starts=starts,
            ends=ends,
        )
        rest = []
    else:
        rest = [mapping["le_lt"]]
    rest.extend(mapping["le_or_ge"])
    rest.extend(mapping["not_equals"])
    rest = [entry for entry in rest if entry]
    booleans = (starts == -1) | (ends == -1) | (starts >= ends)
    if booleans.all():
        return {
            "left_index": empty_array,
            "right_index": empty_array,
        }
    left_index = df.index._values
    if booleans.any():
        booleans = ~booleans
        starts = starts[booleans]
        ends = ends[booleans]
        left_index = left_index[booleans]
    if return_matching_indices and not rest:
        return {
            "left_index": left_index,
            "right_index": right.index._values,
            "starts": starts,
            "ends": ends,
        }
    if not rest:
        right_index = right.index._values
        return _range_indices._build_indices(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            ends=ends,
            keep=keep,
            right_is_sorted=check and is_sorted,
        )
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
        return {
            "left_index": left_index,
            "right_index": right.index._values,
            "starts": starts,
            "ends": ends,
            "counts_array": outcome["counts_array"],
            "matches": outcome["matches"],
        }
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
