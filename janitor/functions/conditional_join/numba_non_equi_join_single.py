import numpy as np
import pandas as pd

from . import helpers


def _numba_single_non_equi_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    keep: str,
    row_count: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching indices for single non-equi join."""
    if row_count:
        return helpers._generic_func_cond_join(
            left=left,
            right=right,
            op=op,
            keep=keep,
            row_count=row_count,
        )
    if op == "!=":
        return helpers._generic_func_cond_join(
            left=left, right=right, op=op, keep=keep
        )
    from janitor.functions.conditional_join import _numba

    outcome = helpers._generic_func_cond_join(
        left=left, right=right, op=op, keep="all"
    )
    if outcome is None:
        return None
    left_index, right_index, starts = outcome
    if op in helpers.greater_than_join_types:
        right_index = right_index[::-1]
        starts = right_index.size - starts
    if keep in {"first", "last"}:
        left_indices = np.empty(left_index.size, dtype=np.intp)
        right_indices = np.empty(left_index.size, dtype=np.intp)
        return _numba._numba_non_equi_join_monotonic_increasing_keep_first_or_last_dual(
            left_index=left_index,
            right_index=right_index,
            starts=starts,
            left_indices=left_indices,
            right_indices=right_indices,
            position=keep == "first",
        )

    start_indices = np.empty(left_index.size, dtype=np.intp)
    start_indices[0] = 0
    indices = (right_index.size - starts).cumsum()
    start_indices[1:] = indices[:-1]
    indices = indices[-1]
    left_indices = np.empty(indices, dtype=np.intp)
    right_indices = np.empty(indices, dtype=np.intp)
    return _numba._numba_non_equi_join_monotonic_increasing_keep_all_dual(
        left_index=left_index,
        right_index=right_index,
        starts=starts,
        left_indices=left_indices,
        right_indices=right_indices,
        start_indices=start_indices,
    )
