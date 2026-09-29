import numpy as np
import pandas as pd

from janitor.functions._conditional_join._equi_join_rust import (
    EquiPredicate,
    RangePredicate,
    _get_indices,
)


def test_unique_equi_representation_uses_get_indexer_positions():
    left = pd.DataFrame({"key": ["b", "missing", "a"]})
    right = pd.DataFrame({"key": ["a", "b"]})

    result = _get_indices(left, right, [("key", "key", "==")])

    assert result is not None
    equi, ranges, residuals, left_index, right_index = result
    assert isinstance(equi, EquiPredicate)
    assert ranges == []
    assert residuals == []
    np.testing.assert_array_equal(left_index, [0, 1, 2])
    np.testing.assert_array_equal(right_index, [0, 1])
    np.testing.assert_array_equal(equi.left_indexer, [1, -1, 0])
    assert equi.original_right_positions is None


def test_duplicate_equi_representation_keeps_factorized_positions():
    left = pd.DataFrame({"key": ["b", "a", "missing"]})
    right = pd.DataFrame({"key": ["a", "b", "a"]})

    result = _get_indices(left, right, [("key", "key", "==")])

    assert result is not None
    equi, _, _, _, _ = result
    np.testing.assert_array_equal(equi.left_indexer, [1, 0, -1])
    np.testing.assert_array_equal(equi.original_right_positions, [0, 1, 0])


def test_shared_range_index_is_global_for_equi_and_both_ranges():
    left = pd.DataFrame({"key": [1], "upper": [8], "lower": [2]})
    right = pd.DataFrame(
        {"key": [1, 1], "upper": [9, 7], "lower": [7, 3]},
        index=[10, 11],
    )

    result = _get_indices(
        left,
        right,
        [
            ("key", "key", "=="),
            ("upper", "upper", ">="),
            ("lower", "lower", "<="),
        ],
    )

    assert result is not None
    equi, ranges, residuals, _, right_index = result
    assert len(ranges) == 2
    assert residuals == []
    assert all(isinstance(item, RangePredicate) for item in ranges)
    np.testing.assert_array_equal(right_index, [11, 10])


def test_non_shared_second_range_is_residual():
    left = pd.DataFrame({"key": [1], "first": [5], "second": [5]})
    right = pd.DataFrame(
        {"key": [1, 1], "first": [8, 3], "second": [1, 9]},
        index=[10, 11],
    )

    result = _get_indices(
        left,
        right,
        [
            ("key", "key", "=="),
            ("first", "first", "<="),
            ("second", "second", "<="),
        ],
    )

    assert result is not None
    _, ranges, residuals, _, right_index = result
    assert len(ranges) == 1
    assert len(residuals) == 1
    np.testing.assert_array_equal(right_index, [11, 10])
    np.testing.assert_array_equal(residuals[0][1], [9, 1])
    assert residuals[0][2] == "<="
