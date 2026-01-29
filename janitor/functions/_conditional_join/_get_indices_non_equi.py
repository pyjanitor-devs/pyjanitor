from __future__ import annotations

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import (
    _helpers,
    _le_ge_1_or_more,
    _not_range_join_regions,
    _range_join_default,
    _range_join_regions,
)


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list,
    keep: str,
    return_matching_indices: bool,
    join_algorithm: str,
) -> tuple:
    """
    Get indices, or aggregates, for multiple conditions,
    where `>/>=` or `</<=` is present
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
    if not mapping["is_range_join"]:
        if (len(mapping["le_or_ge"]) == 1) or (join_algorithm == "default"):
            return _le_ge_1_or_more._get_indices(
                mapping=mapping,
                df=df,
                right=right,
                return_matching_indices=return_matching_indices,
                keep=keep,
            )
        return _not_range_join_regions._get_indices(
            df=df,
            right=right,
            mapping=mapping,
            return_matching_indices=return_matching_indices,
            keep=keep,
        )
    # is range join
    if join_algorithm == "default":
        return _range_join_default._get_indices(
            mapping=mapping,
            df=df,
            right=right,
            return_matching_indices=return_matching_indices,
            keep=keep,
        )
    return _range_join_regions._get_indices(
        df=df,
        right=right,
        mapping=mapping,
        return_matching_indices=return_matching_indices,
        keep=keep,
    )
