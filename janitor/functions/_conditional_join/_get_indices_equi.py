from __future__ import annotations

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _dual_non_equi,
    _greater_than_indices,
    _helpers,
    _less_than_indices,
    _range_indices,
)


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
    _columns = (
        mapping["le_or_ge"],
        mapping["le_lt"],
        mapping["ge_gt"],
        mapping["equals"],
    )
    columns = []
    for entry in _columns:
        if not entry:
            continue
        if isinstance(entry, tuple):
            columns.append(entry)
        else:
            columns.extend(entry)
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
    return mapping