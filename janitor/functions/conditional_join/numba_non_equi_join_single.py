import numpy as np
import pandas as pd

from janitor.functions.conditional_join import _numba

from . import helpers


def _numba_single_non_equi_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    keep: str,
    row_count: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching indices for single non-equi join."""
    outcome = helpers._null_checks_cond_join(series=left)
    if not outcome:
        return None
    left, _ = outcome
    left_index = left.index._values
    outcome = helpers._null_checks_cond_join(series=right)
    if not outcome:
        return None
    right, _ = outcome
    right, _ = helpers._sort_if_not_monotonic(series=right)
    right_index = right.index._values
    left, right = helpers._convert_to_numpy(
        left=left._values, right=right._values
    )
    keep_mapping = {"all": 0, "first": 1, "last": 2}
    result = _numba._get_indices_or_row_count_single_join(
        left=left,
        right=right,
        left_index=left_index,
        right_index=right_index,
        op=helpers.operator_mapping[op],
        keep=keep_mapping[keep],
        row_count=row_count,
    )
    if result[0] is None:
        return None
    if row_count:
        return pd.Series(data=result[0], index=left_index, name=row_count)
    return result
