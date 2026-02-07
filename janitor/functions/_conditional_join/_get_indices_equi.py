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
