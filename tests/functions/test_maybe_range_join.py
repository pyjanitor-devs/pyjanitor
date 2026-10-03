"""Focused tests for the dual-range Python dispatch boundary."""

import numpy as np
import pandas as pd
import pytest

from janitor.functions._conditional_join._maybe_range_join import (
    _compute_multi_range_join,
)


def _conditions():
    return [
        ("left_int", "right_int", "<"),
        ("left_float", "right_float", ">"),
    ]


@pytest.fixture
def dual_frames():
    left = pd.DataFrame({"left_int": [2, 4], "left_float": [6.0, 5.0]})
    right = pd.DataFrame(
        {
            "right_int": [1, 3, 5, 7],
            "right_float": pd.Series([0.0, 2.0, 4.0, 6.0], dtype="float64"),
        }
    )
    return left, right


def test_dual_range_indices_support_independent_anchor_dtypes(dual_frames):
    left, right = dual_frames
    result = _compute_multi_range_join(left, right, _conditions(), "all", False)

    np.testing.assert_array_equal(result["left_index"], [0, 0, 1])
    np.testing.assert_array_equal(result["right_index"], [1, 2, 2])


@pytest.mark.parametrize(
    ("keep", "expected_right"),
    [("first", [1, 2]), ("last", [2, 2]), ("any", [1, 2])],
)
def test_dual_range_keep_modes_use_shared_right_layout(
    dual_frames, keep, expected_right
):
    left, right = dual_frames
    result = _compute_multi_range_join(left, right, _conditions(), keep, False)

    np.testing.assert_array_equal(result["left_index"], [0, 1])
    np.testing.assert_array_equal(result["right_index"], expected_right)


def test_dual_range_unordered_right_index_uses_window_extrema():
    left = pd.DataFrame({"left": [3], "other_left": [7]})
    right = pd.DataFrame(
        {"right": [3, 1, 5, 7], "other_right": [2, 0, 4, 6]},
    )
    conditions = [("left", "right", "<"), ("other_left", "other_right", ">")]

    first = _compute_multi_range_join(left, right, conditions, "first", False)
    last = _compute_multi_range_join(left, right, conditions, "last", False)

    np.testing.assert_array_equal(first["right_index"], [2])
    np.testing.assert_array_equal(last["right_index"], [3])


def test_dual_range_building_blocks_are_returned_when_requested(dual_frames):
    left, right = dual_frames
    result = _compute_multi_range_join(left, right, _conditions(), "first", True)

    np.testing.assert_array_equal(result["left_index"], [0, 1])
    np.testing.assert_array_equal(result["right_index"], [0, 1, 2, 3])
    np.testing.assert_array_equal(result["starts"], [1, 2])
    np.testing.assert_array_equal(result["ends"], [3, 3])


def test_dual_range_null_rows_are_removed_before_dispatch():
    left = pd.DataFrame({"left": [2.0, np.nan], "other_left": [5.0, 5.0]})
    right = pd.DataFrame({"right": [1.0, 3.0, np.nan], "other_right": [0.0, 2.0, 1.0]})
    conditions = [("left", "right", "<"), ("other_left", "other_right", ">")]

    result = _compute_multi_range_join(left, right, conditions, "all", False)

    np.testing.assert_array_equal(result["left_index"], [0])
    np.testing.assert_array_equal(result["right_index"], [1])


def test_dual_range_no_match_returns_standard_empty_result():
    left = pd.DataFrame({"left": [1], "other_left": [1]})
    right = pd.DataFrame({"right": [1, 2], "other_right": [2, 3]})
    conditions = [("left", "right", ">"), ("other_left", "other_right", ">")]

    result = _compute_multi_range_join(left, right, conditions, "all", False)

    assert result.keys() == {"left_index", "right_index"}
    assert result["left_index"].size == 0
    assert result["right_index"].size == 0


def test_dual_range_extended_dispatch_applies_residual_predicates():
    left = pd.DataFrame({"left_a": [2], "left_b": [6], "left_x": [0]})
    right = pd.DataFrame(
        {
            "right_a": [1, 3, 5, 7],
            "right_b": [0, 2, 4, 6],
            "right_x": [1, 0, 2, 3],
        }
    )
    conditions = [
        ("left_a", "right_a", "<"),
        ("left_b", "right_b", ">"),
        ("left_x", "right_x", "!="),
    ]

    result = _compute_multi_range_join(left, right, conditions, "all", False)

    np.testing.assert_array_equal(result["left_index"], [0])
    np.testing.assert_array_equal(result["right_index"], [2])
