from __future__ import annotations

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _binary_search


def _check_sorted_within_groups(
    equi_cols: list[tuple],
    right: pd.DataFrame,
    r_col: str | tuple,
) -> bool:
    """Check if r_col is sorted within each group of equi_cols."""
    r_cols = []
    for _, right_col, _ in equi_cols:
        r_cols.append(right_col)
    # handle duplicate keys in equi_cols
    # while still maintaining order
    grouper = dict.fromkeys(r_cols)
    grouper = [*grouper]
    grouped = right.groupby(grouper, sort=False, observed=True)
    check = grouped[r_col].is_monotonic_increasing.all()
    return check


def _maybe_sort_right(
    right: pd.DataFrame,
    r1_col: str | tuple,
    r2_col: str | tuple | None = None,
    is_sorted: bool = False,
) -> pd.DataFrame:
    """Sort the right DataFrame if needed."""
    if is_sorted:
        return right
    if r2_col is None:
        sorter = r1_col
    # possibly a range join
    else:
        # handle duplicates
        # while still maintaining order
        sorter = dict.fromkeys([r1_col, r2_col])
        sorter = [*sorter]
    right = right.sort_values(sorter, kind="stable", ignore_index=False)
    return right


def _build_equi_indices(df, right, mapping):
    l_cols = []
    r_cols = []
    for left_col, right_col, _ in mapping["equals"]:
        l_cols.append(df[left_col]._values)
        r_cols.append(right[right_col]._values)
    if len(l_cols) > 1:
        l_cols = pd.MultiIndex.from_arrays(l_cols)
        r_cols = pd.MultiIndex.from_arrays(r_cols)
    else:
        l_cols = pd.Index(l_cols[0])
        r_cols = pd.Index(r_cols[0])
    return l_cols, r_cols


def _get_positions_right(right_columns: pd.Index):
    positions, uniques = right_columns.factorize()
    counts = np.bincount(positions, minlength=len(uniques))
    starts = np.empty(counts.size, dtype=np.int64)
    starts[0] = 0
    starts[1:] = counts.cumsum()[:-1]
    ends = starts + counts
    return positions, uniques, starts, ends


def _get_indexers(l_cols, uniques, starts, ends):
    indexers = uniques.get_indexer(l_cols)
    booleans = indexers == -1
    if booleans.all():
        return None
    # align df to right
    starts = starts[indexers]
    ends = ends[indexers]
    if booleans.any():
        starts = np.where(booleans, -1, starts)
        ends = np.where(booleans, -1, ends)
    return indexers, starts, ends


def _update_positions_ge_gt(
    op: str,
    l_column: np.ndarray,
    r_column: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    if op == ">":
        return _binary_search._binary_search_gt(
            left=l_column, right=r_column, starts=starts, ends=ends
        )
    return _binary_search._binary_search_ge(
        left=l_column, right=r_column, starts=starts, ends=ends
    )


def _update_positions_le_lt(
    op: str,
    l_column: np.ndarray,
    r_column: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    if op == "<":
        return _binary_search._binary_search_lt(
            left=l_column, right=r_column, starts=starts, ends=ends
        )
    return _binary_search._binary_search_le(
        left=l_column, right=r_column, starts=starts, ends=ends
    )
