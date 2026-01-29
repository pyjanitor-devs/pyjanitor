from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _equi_helpers, _helpers


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    keep: str,
    return_matching_indices: bool,
) -> dict:
    """
    Get indices for != only
    """
    empty_array = np.array([], dtype=np.intp)
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
    indexers, starts, ends = outcome
    booleans = indexers == -1
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
    outcome = _helpers._get_positive_matches_conditions(
        df=df,
        right=right,
        conditions=mapping["not_equals"],
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
