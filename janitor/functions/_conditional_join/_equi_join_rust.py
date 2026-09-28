"""Representation passed from PyJanitor to the fresh equi Rust path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janitor.functions._conditional_join import _helpers
from janitor.functions._conditional_join._helpers import _prepare_range_anchor

RANGE_OPERATORS = frozenset({"<", "<=", ">", ">="})
EQUI_OPERATOR = "=="
NOT_EQUAL_OPERATOR = "!="

_RANGE_PAIR_PRIORITY = (
    (">", "<"),
    (">", "<="),
    (">=", "<"),
    (">=", "<="),
    (">", ">"),
    (">", ">="),
    (">=", ">"),
    (">=", ">="),
    ("<", "<"),
    ("<", "<="),
    ("<=", "<"),
    ("<=", "<="),
)


@dataclass(frozen=True)
class EquiPredicate:
    """The global equi candidate representation consumed by Rust."""

    left_index: np.ndarray
    right_index: np.ndarray
    left_indexer: np.ndarray
    original_right_positions: np.ndarray | None
    right_codes: np.ndarray | None = None


@dataclass(frozen=True)
class RangePredicate:
    """A range predicate whose arrays use the payload's global indexes."""

    left_index: np.ndarray
    right_index: np.ndarray
    left_values: np.ndarray
    right_values: np.ndarray
    operator: str


@dataclass(frozen=True)
class ResidualPredicate:
    """A predicate evaluated after equi/range candidate generation."""

    left_index: np.ndarray
    right_index: np.ndarray
    left_values: np.ndarray
    right_values: np.ndarray
    operator: str


@dataclass(frozen=True)
class PreparedEquiJoin:
    """Ordered predicate list passed to the future Rust entry point."""

    predicates: tuple[EquiPredicate | RangePredicate | ResidualPredicate, ...]

    @property
    def equi(self) -> EquiPredicate:
        return self.predicates[0]


def prepare_equi_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> PreparedEquiJoin | None:
    """Build the clear Python-to-Rust representation for an equi-led join."""
    left, right = _get_indices(left, right, conditions)
    if left is None or right is None:
        return None

    range_conditions = [
        (index, condition)
        for index, condition in enumerate(conditions)
        if condition[2] in RANGE_OPERATORS
    ]
    first_range, second_range, selected_ranges = _select_range_pair_maybe(
        left, right, range_conditions
    )
    right_index = (
        first_range.right_index
        if first_range is not None
        else np.asarray(right.index._values, dtype=np.int64)
    )
    right_for_join = right.loc[right_index]

    equi_conditions = [
        condition for condition in conditions if condition[2] == EQUI_OPERATOR
    ]
    equi = _build_equi_predicate(left, right_for_join, equi_conditions)
    if equi is None:
        return None
    equi = EquiPredicate(
        left_index=equi.left_index,
        right_index=right_index,
        left_indexer=equi.left_indexer,
        original_right_positions=equi.original_right_positions,
        right_codes=equi.right_codes,
    )

    predicates: list[EquiPredicate | RangePredicate | ResidualPredicate] = [equi]
    if first_range is not None:
        predicates.append(first_range)
    if second_range is not None:
        predicates.append(second_range)

    for index, (left_column, right_column, operator) in enumerate(conditions):
        if operator == EQUI_OPERATOR or index in selected_ranges:
            continue
        predicates.append(
            ResidualPredicate(
                left_index=np.asarray(left.index._values, dtype=np.int64),
                right_index=right_index,
                left_values=np.asarray(left[left_column]._values),
                right_values=np.asarray(right_for_join[right_column]._values),
                operator=operator,
            )
        )
    return PreparedEquiJoin(predicates=tuple(predicates))


def _build_equi_predicate(
    left: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> EquiPredicate | None:
    left_keys, right_keys = _build_equi_keys(left, right, conditions)
    left_index = np.asarray(left.index._values, dtype=np.int64)
    right_index = np.asarray(right.index._values, dtype=np.int64)
    try:
        left_indexer = np.asarray(right_keys.get_indexer(left_keys), dtype=np.int64)
        if np.all(left_indexer == -1):
            return None
        return EquiPredicate(left_index, right_index, left_indexer, None)
    except pd.errors.InvalidIndexError:
        right_codes, uniques = right_keys.factorize(sort=False)
        left_indexer = np.asarray(uniques.get_indexer(left_keys), dtype=np.int64)
        if np.all(left_indexer == -1):
            return None
        return EquiPredicate(
            left_index=left_index,
            right_index=right_index,
            left_indexer=left_indexer,
            original_right_positions=right_index.copy(),
            right_codes=np.asarray(right_codes, dtype=np.int64),
        )


def _build_equi_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple[pd.Index, pd.Index]:
    left_arrays = [left[left_column]._values for left_column, _, _ in conditions]
    right_arrays = [right[right_column]._values for _, right_column, _ in conditions]
    if len(left_arrays) == 1:
        return pd.Index(left_arrays[0]), pd.Index(right_arrays[0])
    return pd.MultiIndex.from_arrays(left_arrays), pd.MultiIndex.from_arrays(
        right_arrays
    )


def _select_range_pair_maybe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    range_conditions: list[tuple[int, tuple[str, str, str]]],
) -> tuple[RangePredicate | None, RangePredicate | None, set[int]]:
    """Select one or two range predicates in one shared right layout."""
    if not range_conditions:
        return None, None, set()

    anchors = {}
    for position, (left_column, right_column, operator) in range_conditions:
        anchor = _prepare_range_anchor(left[left_column], right[right_column])
        if anchor is not None:
            anchors[position] = ((left_column, right_column, operator), anchor)
    if not anchors:
        return None, None, set()

    for first_operator, second_operator in _RANGE_PAIR_PRIORITY:
        for first_position, (first_condition, first_anchor) in anchors.items():
            if first_condition[2] != first_operator:
                continue
            for second_position, (second_condition, _) in anchors.items():
                if (
                    second_position == first_position
                    or second_condition[2] != second_operator
                ):
                    continue
                second_values = right.loc[first_anchor.right_index, second_condition[1]]
                if second_values.is_monotonic_increasing:
                    return (
                        _range_predicate(first_anchor, first_condition[2]),
                        RangePredicate(
                            left_index=first_anchor.left_index,
                            right_index=first_anchor.right_index,
                            left_values=np.asarray(left[second_condition[0]]._values),
                            right_values=np.asarray(second_values._values),
                            operator=second_condition[2],
                        ),
                        {first_position, second_position},
                    )

    first_position, (first_condition, first_anchor) = next(iter(anchors.items()))
    return _range_predicate(first_anchor, first_condition[2]), None, {first_position}


def _range_predicate(anchor, operator: str) -> RangePredicate:
    return RangePredicate(
        left_index=anchor.left_index,
        right_index=anchor.right_index,
        left_values=anchor.left_array,
        right_values=anchor.right_array,
        operator=operator,
    )


def _get_indices(
    df: pd.DataFrame,
    right: pd.DataFrame,
    conditions: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Apply the existing null policy before building the representation."""
    left_columns = {
        left_column
        for left_column, _, operator in conditions
        if operator != NOT_EQUAL_OPERATOR
    }
    df = _helpers._maybe_remove_nulls_from_dataframe(df=df, columns=left_columns)
    if df is None:
        return None, None
    right_columns = {
        right_column
        for _, right_column, operator in conditions
        if operator != NOT_EQUAL_OPERATOR
    }
    right = _helpers._maybe_remove_nulls_from_dataframe(df=right, columns=right_columns)
    if right is None:
        return None, None
    return df, right
