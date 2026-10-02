from __future__ import annotations

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers


def _empty_indices() -> dict[str, np.ndarray]:
    """Return the standard empty conditional-join result.

    Returns:
        A dictionary containing empty left_index and right_index arrays,
        matching the other conditional-join paths.
    """
    empty = np.array([], dtype=np.intp)
    return {"left_index": empty, "right_index": empty}


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
    keep: str,
    return_building_blocks: bool = False,
) -> dict[str, np.ndarray] | None:
    left_columns = {
        left_column
        for left_column, _, operator in conditions
        if operator != _helpers._JoinOperator.NOT_EQUAL.value
    }
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=left_columns)
    if df is None:
        return None
    right_columns = {
        right_column
        for _, right_column, operator in conditions
        if operator != _helpers._JoinOperator.NOT_EQUAL.value
    }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=right_columns)
    if right is None:
        return None
    anchor_condition = None
    for condition in conditions:
        if condition[-1] in _helpers.less_than_join_types.union(
            _helpers.greater_than_join_types
        ) and (anchor_condition is None):
            anchor_condition = condition
        break
    if anchor_condition:
        left_column, right_column, op = anchor_condition
        right_, _ = _helpers._sort_if_not_monotonic(series=right[right_column])
        left_array = _helpers._convert_array_to_numpy(array=df[left_column]._values)
        right_array = _helpers._convert_array_to_numpy(array=right_._values)
        range_predicate = (
            left_array,
            right_array,
            op,
        )
        right_index = right_.index
    else:  # all !=
        range_predicate = None
        right_index = right.index

    rest = [condition for condition in conditions if condition != anchor_condition]
