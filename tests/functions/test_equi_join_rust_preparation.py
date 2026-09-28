import numpy as np
import pandas as pd

from janitor.functions._conditional_join._equi_join_rust import (
    EquiPredicate,
    RangePredicate,
    ResidualPredicate,
    prepare_equi_join,
)


def test_unique_equi_representation_uses_get_indexer_positions():
    left = pd.DataFrame({"key": ["b", "missing", "a"]})
    right = pd.DataFrame({"key": ["a", "b"]})

    prepared = prepare_equi_join(left, right, [("key", "key", "==")])

    assert prepared is not None
    assert len(prepared.predicates) == 1
    equi = prepared.equi
    assert isinstance(equi, EquiPredicate)
    np.testing.assert_array_equal(equi.left_index, [0, 1, 2])
    np.testing.assert_array_equal(equi.right_index, [0, 1])
    np.testing.assert_array_equal(equi.left_indexer, [1, -1, 0])
    assert equi.original_right_positions is None


def test_duplicate_equi_representation_keeps_original_positions():
    left = pd.DataFrame({"key": ["b", "a", "missing"]})
    right = pd.DataFrame({"key": ["a", "b", "a"]})

    prepared = prepare_equi_join(left, right, [("key", "key", "==")])

    assert prepared is not None
    equi = prepared.equi
    np.testing.assert_array_equal(equi.left_indexer, [1, 0, -1])
    np.testing.assert_array_equal(equi.right_codes, [0, 1, 0])
    np.testing.assert_array_equal(equi.original_right_positions, [0, 1, 2])


def test_shared_range_index_is_used_by_equi_and_both_ranges():
    left = pd.DataFrame({"key": [1], "upper": [8], "lower": [2]})
    right = pd.DataFrame(
        {"key": [1, 1], "upper": [9, 7], "lower": [7, 3]},
        index=[10, 11],
    )

    prepared = prepare_equi_join(
        left,
        right,
        [
            ("key", "key", "=="),
            ("upper", "upper", ">="),
            ("lower", "lower", "<="),
        ],
    )

    assert prepared is not None
    assert [type(predicate) for predicate in prepared.predicates] == [
        EquiPredicate,
        RangePredicate,
        RangePredicate,
    ]
    equi, first_range, second_range = prepared.predicates
    np.testing.assert_array_equal(equi.right_index, [11, 10])
    np.testing.assert_array_equal(first_range.right_index, equi.right_index)
    np.testing.assert_array_equal(second_range.right_index, equi.right_index)


def test_non_shared_second_range_is_residual():
    left = pd.DataFrame({"key": [1], "first": [5], "second": [5]})
    right = pd.DataFrame(
        {"key": [1, 1], "first": [8, 3], "second": [1, 9]},
        index=[10, 11],
    )

    prepared = prepare_equi_join(
        left,
        right,
        [
            ("key", "key", "=="),
            ("first", "first", "<="),
            ("second", "second", "<="),
        ],
    )

    assert prepared is not None
    assert isinstance(prepared.predicates[1], RangePredicate)
    assert isinstance(prepared.predicates[2], ResidualPredicate)
    residual = prepared.predicates[2]
    np.testing.assert_array_equal(residual.right_index, prepared.equi.right_index)
