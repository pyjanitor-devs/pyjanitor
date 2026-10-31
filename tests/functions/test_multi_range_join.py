"""Focused tests for the dual-range Python dispatch boundary."""

import operator

import numpy as np
import pandas as pd
import pytest

import janitor
from janitor.functions._conditional_join._helpers import _normalize_conditions
from janitor.functions._conditional_join._range_join import (
    _compute_multi_range_join,
)


def _conditions():
    return _normalize_conditions(
        [
            ("left_int", "right_int", "<"),
            ("left_float", "right_float", ">"),
        ]
    )


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
    conditions = _normalize_conditions(
        [("left", "right", "<"), ("other_left", "other_right", ">")]
    )

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
    conditions = _normalize_conditions(
        [("left", "right", "<"), ("other_left", "other_right", ">")]
    )

    result = _compute_multi_range_join(left, right, conditions, "all", False)

    np.testing.assert_array_equal(result["left_index"], [0])
    np.testing.assert_array_equal(result["right_index"], [1])


def test_dual_range_no_match_returns_standard_empty_result():
    left = pd.DataFrame({"left": [1], "other_left": [1]})
    right = pd.DataFrame({"right": [1, 2], "other_right": [2, 3]})
    conditions = _normalize_conditions(
        [("left", "right", ">"), ("other_left", "other_right", ">")]
    )

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
    conditions = _normalize_conditions(
        [
            ("left_a", "right_a", "<"),
            ("left_b", "right_b", ">"),
            ("left_x", "right_x", "!="),
        ]
    )

    result = _compute_multi_range_join(left, right, conditions, "all", False)

    np.testing.assert_array_equal(result["left_index"], [0])
    np.testing.assert_array_equal(result["right_index"], [2])


def test_dual_range_uses_cumulative_envelope_for_unsorted_second_bound():
    """Keep interval joins on the bounded-window path after right sorting."""
    left = pd.DataFrame(
        {
            "start": [1, 4, 7],
            "end": [2, 6, 9],
        }
    )
    right = pd.DataFrame(
        {
            "start": [2, 0, 4, 1],
            "end": [3, 8, 5, 2],
        }
    )
    # This is the usual interval shape, deliberately supplied in upper-bound
    # first order. Sorting by right.start makes right.end non-monotonic.
    conditions = _normalize_conditions(
        [
            ("start", "end", "<="),
            ("end", "start", ">="),
        ]
    )

    result = _compute_multi_range_join(left, right, conditions, "all", False)

    expected = [
        (left_row, right_row)
        for left_row in range(len(left))
        for right_row in range(len(right))
        if left.iloc[left_row]["start"] <= right.iloc[right_row]["end"]
        and left.iloc[left_row]["end"] >= right.iloc[right_row]["start"]
    ]
    actual = list(zip(result["left_index"], result["right_index"]))
    assert sorted(actual) == sorted(expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_dual_range_aggregation_rechecks_integer_sum_and_prod(reverse):
    left = pd.DataFrame({"start": [1, 4, 7], "end": [2, 6, 9], "value": [2, 3, 4]})
    right = pd.DataFrame(
        {
            "start": [2, 0, 4, 1],
            "end": [3, 8, 5, 2],
            "value": [5, 6, 7, 8],
        }
    )
    conditions = [("start", "end", "<="), ("end", "start", ">=")]
    aggregation = [("value", "sum"), ("value", "prod")]

    actual = left.join_agg(
        right,
        *conditions,
        aggfunc=aggregation,
        reverse=reverse,
        return_matched=True,
    )

    if reverse:
        expected = pd.DataFrame(
            [[9, 24], [2, 2], [2, 2], [3, 3]],
            index=pd.MultiIndex.from_tuples(
                [(1, True), (3, True), (0, True), (2, True)],
                names=[None, "matched"],
            ),
            columns=pd.MultiIndex.from_tuples([("value", "sum"), ("value", "prod")]),
        )
    else:
        expected = pd.DataFrame(
            [[19, 240], [13, 42], [6, 6]],
            index=pd.MultiIndex.from_tuples(
                [(0, True), (1, True), (2, True)],
                names=[None, "matched"],
            ),
            columns=pd.MultiIndex.from_tuples([("value", "sum"), ("value", "prod")]),
        )

    pd.testing.assert_frame_equal(actual.sort_index(), expected.sort_index())


_ANY_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


@pytest.mark.parametrize("first_op", list(_ANY_OPS))
@pytest.mark.parametrize("second_op", list(_ANY_OPS))
@pytest.mark.parametrize("kind", ["int", "float", "datetime", "int_min", "uint_big"])
def test_keep_any_two_ranges_every_direction(first_op, second_op, kind):
    """keep="any" with two range predicates, including same-direction pairs
    whose second right array is not monotonic (negated envelope),
    must report exactly the left rows that have a match, each with a real one.
    """
    seed = sum(map(ord, first_op + second_op + kind))
    rng = np.random.default_rng(seed)

    def make(size):
        base = rng.integers(0, 12, size)
        if kind == "float":
            return base.astype(float) / 2
        if kind == "datetime":
            return pd.Timestamp("2021-01-01") + pd.to_timedelta(base, "D")
        if kind == "int_min":  # no positive twin: must fall back, not overflow
            return np.where(base == 0, np.iinfo(np.int64).min, base)
        if kind == "uint_big":  # too large to widen to int64
            return base.astype(np.uint64) + np.uint64(2**63)
        return base

    left = pd.DataFrame({"x": make(80), "y": make(80)})
    right = pd.DataFrame({"x": make(120), "y": make(120)})
    # a second right array that is not monotonic in the first one's order
    right = right.sort_values("x").reset_index(drop=True)
    right["y"] = right["y"].to_numpy()[::-1]

    actual = janitor.get_join_indices(
        left,
        right,
        ("x", "x", first_op),
        ("y", "y", second_op),
        keep="any",
    )
    left_index, right_index = (np.asarray(v) for v in list(actual.values())[:2])

    hits = _ANY_OPS[first_op](
        left["x"].to_numpy()[:, None], right["x"].to_numpy()[None, :]
    ) & _ANY_OPS[second_op](
        left["y"].to_numpy()[:, None], right["y"].to_numpy()[None, :]
    )
    assert len(left_index) == len(set(left_index.tolist()))
    assert set(left_index.tolist()) == set(np.flatnonzero(hits.any(axis=1)).tolist())
    assert hits[left_index, right_index].all()
