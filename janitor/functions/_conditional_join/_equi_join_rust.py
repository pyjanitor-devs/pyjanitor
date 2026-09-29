"""Representation passed from PyJanitor to the fresh equi Rust path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers


@dataclass(frozen=True)
class EquiPredicate:
    """The global equi candidate representation consumed by Rust."""

    left_indexer: np.ndarray
    original_right_positions: np.ndarray | None


@dataclass(frozen=True)
class RangePredicate:
    """A range predicate whose arrays use the payload's global indexes."""

    left_values: np.ndarray
    right_values: np.ndarray
    operator: str


@dataclass(frozen=True)
class ResidualPredicate:
    """A predicate evaluated after equi/range candidate generation."""

    left_values: np.ndarray
    right_values: np.ndarray
    operator: str


def _build_equi_predicate(
    left_keys: pd.Index,
    right_keys: pd.Index,
) -> EquiPredicate | None:
    try:
        left_indexer = right_keys.get_indexer(left_keys)
        if np.all(left_indexer == -1):
            return None
        return EquiPredicate(
            left_indexer=left_indexer,
            original_right_positions=None,
        )
    except pd.errors.InvalidIndexError:
        original_right_positions, uniques = right_keys.factorize(sort=False)
        left_indexer = uniques.get_indexer(left_keys)
        if np.all(left_indexer == -1):
            return None
        return EquiPredicate(
            left_indexer=left_indexer,
            original_right_positions=original_right_positions,
        )


def _build_equi_keys(df, right, right_index, equi_conditions):
    l_cols = []
    r_cols = []
    if right_index is None:
        for left_col, right_col, _ in equi_conditions:
            l_cols.append(df[left_col]._values)
            r_cols.append(right[right_col]._values)
    else:
        for left_col, right_col, _ in equi_conditions:
            l_cols.append(df[left_col]._values)
            r_cols.append(right.loc[right_index, right_col]._values)
    if len(l_cols) > 1:
        l_cols = pd.MultiIndex.from_arrays(l_cols)
        r_cols = pd.MultiIndex.from_arrays(r_cols)
    else:
        l_cols = pd.Index(l_cols[0])
        r_cols = pd.Index(r_cols[0])
    return l_cols, r_cols


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple | None:
    """Apply the existing null policy before building the representation."""
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
    mapped_conditions = _helpers._separate_conditions_based_on_join_op(
        conditions=conditions
    )
    le_lt = mapped_conditions["le_lt"]
    ge_gt = mapped_conditions["ge_gt"]
    rest = []
    rest.extend(le_lt)
    rest.extend(ge_gt)
    rest.extend(mapped_conditions["not_equals"])
    range_maybe = []
    if le_lt:
        range_maybe.append(le_lt[0])
    if ge_gt:
        range_maybe.append(ge_gt[0])
    if len(range_maybe) == 1 and len(le_lt) > 1:
        range_maybe = le_lt[:2]
    elif len(range_maybe) == 1 and len(ge_gt) > 1:
        range_maybe = ge_gt[:2]
    # check if range exists and meets criteria
    range_predicates = []
    if range_maybe:
        left_column, right_column, op = range_maybe[0]
        right_, _ = _helpers._sort_if_not_monotonic(series=right[right_column])
        range_predicate = RangePredicate(
            left_values=_helpers._convert_array_to_numpy(array=df[left_column]._values),
            right_values=_helpers._convert_array_to_numpy(array=right_._values),
            operator=op,
        )
        range_predicates.append(range_predicate)
        if len(range_maybe) > 1:
            left_column, right_column, op = range_maybe[1]
            right_ = right.loc[right_.index, right_column]
            if right_.is_monotonic_increasing:
                range_predicate = RangePredicate(
                    left_values=_helpers._convert_array_to_numpy(
                        array=df[left_column]._values
                    ),
                    right_values=_helpers._convert_array_to_numpy(array=right_._values),
                    operator=op,
                )
                range_predicates.append(range_predicate)
            else:
                range_maybe = [range_maybe[0]]
        right_index = right_.index
        rest = [condition for condition in rest if condition not in range_maybe]
    else:
        right_index = None
    left_keys, right_keys = _build_equi_keys(
        df=df,
        right=right,
        right_index=right_index,
        equi_conditions=mapped_conditions["equals"],
    )
    equi_predicates = _build_equi_predicate(
        left_keys=left_keys,
        right_keys=right_keys,
    )

    residual_predicates = []
    if rest:
        for left_column, right_column, operator in rest:
            residual_predicate = _helpers._build_residual_predicate(
                left=df[left_column],
                right=right[right_column],
                operation=operator,
                right_index=right_index,
            )
            residual_predicates.append(residual_predicate)
    left_index = _helpers._convert_array_to_numpy(array=df.index._values)
    if right_index is None:
        right_index = right.index
    right_index = _helpers._convert_array_to_numpy(array=right_index._values)
    return (
        equi_predicates,
        range_predicates,
        residual_predicates,
        left_index,
        right_index,
    )
