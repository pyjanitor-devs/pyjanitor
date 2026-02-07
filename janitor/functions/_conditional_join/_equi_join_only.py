from __future__ import annotations

import janitor_rs
import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _equi_helpers,
    _range_indices,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    mapping: dict,
    return_matching_indices: bool,
    keep: str,
) -> dict:
    """Get indices for an equi join only"""
    # purely equi join, and return_matching_indices is True
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
        return {
            "left_index": np.array([], dtype=np.intp),
            "right_index": np.array([], dtype=np.intp),
        }
    indexers, starts, ends = outcome
    booleans = indexers == -1
    if booleans.all():
        return {
            "left_index": np.array([], dtype=np.intp),
            "right_index": np.array([], dtype=np.intp),
        }
    left_index = df.index._values
    if booleans.any():
        booleans = ~booleans
        starts = starts[booleans]
        ends = ends[booleans]
        left_index = left_index[booleans]
    right_index = right.index._values
    if return_matching_indices:
        return {
            "left_index": left_index,
            "right_index": right_index,
            "starts": starts,
            "ends": ends,
        }
    return _range_indices._build_indices(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        ends=ends,
        keep=keep,
        right_is_sorted=check,
    )
