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
        """Random integer frame with ties."""
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


def _spy_staircase(monkeypatch):
    """Count calls to the staircase kernel made by the dual-range module."""
    from janitor.functions._conditional_join import _range_join

    calls = []
    original = _range_join.janitor_rs.range_exists_indices

    def spy(*args, **kwargs):
        """Count the call, then run the real kernel."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_range_join.janitor_rs, "range_exists_indices", spy)
    return calls


def test_keep_any_staircase_only_for_two_range_predicates(monkeypatch):
    """Exactly two range predicates and keep="any" use the staircase; an extra
    predicate, another keep mode, or building blocks keep the window kernels.
    """
    calls = _spy_staircase(monkeypatch)
    left = pd.DataFrame({"x": [1, 5, 9], "y": [3, 3, 3], "z": [0, 0, 1]})
    right = pd.DataFrame({"x": [2, 6], "y": [1, 9], "z": [0, 1]})
    two = (("x", "x", "<="), ("y", "y", ">"))

    result = janitor.get_join_indices(left, right, *two, keep="any")
    assert list(result["left_index"]) == [0]
    assert len(calls) == 1

    janitor.get_join_indices(left, right, *two, ("z", "z", "!="), keep="any")
    janitor.get_join_indices(left, right, *two, ("z", "z", "=="), keep="any")
    janitor.get_join_indices(left, right, *two, keep="first")
    janitor.get_join_indices(left, right, *two, keep="all")
    janitor.get_join_indices(left, right, *two, keep="any", return_building_blocks=True)
    assert len(calls) == 1


@pytest.mark.parametrize("first_op", list(_ANY_OPS))
@pytest.mark.parametrize("second_op", list(_ANY_OPS))
def test_keep_any_two_ranges_with_residual_every_direction(
    first_op, second_op, monkeypatch
):
    """A third predicate sends keep="any" to the window kernels, so the
    same-direction negated envelope keeps its coverage: matched left rows are
    exact and every pair satisfies all three predicates.
    """
    calls = _spy_staircase(monkeypatch)
    rng = np.random.default_rng(sum(map(ord, first_op + second_op)))
    left = pd.DataFrame(
        {
            "x": rng.integers(0, 12, 80),
            "y": rng.integers(0, 12, 80),
            "z": rng.integers(0, 3, 80),
        }
    )
    right = pd.DataFrame(
        {
            "x": rng.integers(0, 12, 120),
            "y": rng.integers(0, 12, 120),
            "z": rng.integers(0, 3, 120),
        }
    )
    right = right.sort_values("x").reset_index(drop=True)
    right["y"] = right["y"].to_numpy()[::-1]

    actual = janitor.get_join_indices(
        left,
        right,
        ("x", "x", first_op),
        ("y", "y", second_op),
        ("z", "z", "!="),
        keep="any",
    )
    left_index, right_index = (np.asarray(v) for v in list(actual.values())[:2])
    hits = (
        _ANY_OPS[first_op](
            left["x"].to_numpy()[:, None], right["x"].to_numpy()[None, :]
        )
        & _ANY_OPS[second_op](
            left["y"].to_numpy()[:, None], right["y"].to_numpy()[None, :]
        )
        & (left["z"].to_numpy()[:, None] != right["z"].to_numpy()[None, :])
    )
    assert not calls
    assert len(left_index) == len(set(left_index.tolist()))
    assert set(left_index.tolist()) == set(np.flatnonzero(hits.any(axis=1)).tolist())
    assert hits[left_index, right_index].all()


@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "int32", "uint32", "float32", "Int64", "Float64"]
)
@pytest.mark.parametrize("first_op", ["<=", ">"])
@pytest.mark.parametrize("second_op", ["<", ">="])
def test_keep_any_staircase_dtypes_and_nulls(dtype, first_op, second_op):
    """Narrow, unsigned, float32 and nullable columns widen inside the kernel;
    null rows on either side never match and never shift the reported rows.
    """
    rng = np.random.default_rng(7)

    def build(size):
        """Random frame with nulls, typed for the case under test."""
        frame = pd.DataFrame(
            {
                name: pd.array(rng.integers(0, 30, size), dtype="Int64")
                for name in ("x", "y")
            }
        )
        for name in frame:
            frame.loc[rng.random(size) < 0.15, name] = pd.NA
        if dtype in ("Int64", "Float64"):
            return frame.astype(dtype)
        # NumPy dtypes cannot hold NA, so only they lose the null rows.
        return frame.dropna().astype(dtype)

    left, right = build(90), build(110)
    actual = janitor.get_join_indices(
        left, right, ("x", "x", first_op), ("y", "y", second_op), keep="any"
    )
    left_index, right_index = (np.asarray(v) for v in list(actual.values())[:2])
    lx = left["x"].to_numpy(dtype="float64", na_value=np.nan)
    ly = left["y"].to_numpy(dtype="float64", na_value=np.nan)
    rx = right["x"].to_numpy(dtype="float64", na_value=np.nan)
    ry = right["y"].to_numpy(dtype="float64", na_value=np.nan)
    hits = _ANY_OPS[first_op](lx[:, None], rx[None, :]) & _ANY_OPS[second_op](
        ly[:, None], ry[None, :]
    )
    assert len(left_index) == len(set(left_index.tolist()))
    assert set(left_index.tolist()) == set(np.flatnonzero(hits.any(axis=1)).tolist())
    assert hits[left_index, right_index].all()


def _staircase_pools():
    """Value pools that stress the kernel's dtype families, with plenty of ties."""
    i64 = np.iinfo(np.int64)
    return {
        "int64_extremes": (np.array([i64.min, i64.min + 1, 0, 7, i64.max]), None),
        "uint64_above_int64": (
            np.array([2**64 - 1, 2**64 - 2, 2**63 + 1, 2**63, 5], dtype=np.uint64),
            None,
        ),
        "datetime": (np.arange(6) * 3, lambda a: pd.to_datetime(a, unit="s")),
        "timedelta": (np.arange(6) * 3, lambda a: pd.to_timedelta(a, unit="s")),
    }


@pytest.mark.parametrize("pool", list(_staircase_pools()))
@pytest.mark.parametrize("how", ["inner", "left", "right"])
@pytest.mark.parametrize("ops", [("<=", ">"), (">=", "<")])
def test_keep_any_staircase_labels_dtypes_and_how(monkeypatch, pool, how, ops):
    """The label mapping into the join result is right for datetimes,
    timedeltas, the int64 minimum, uint64 above the int64 maximum, and a
    shuffled non-default index, for every ``how``.
    """
    values, wrap = _staircase_pools()[pool]
    wrap = wrap or (lambda a: a)
    rng = np.random.default_rng(11)

    def build(size, prefix, id_name):
        """A frame with ties, an id column, and a shuffled index."""
        frame = pd.DataFrame(
            {
                prefix + "x": wrap(rng.choice(values, size)),
                prefix + "y": wrap(rng.choice(values, size)),
                id_name: np.arange(size),
            }
        )
        frame.index = rng.permutation(np.arange(size) + 1000)
        return frame

    left, right = build(40, "", "lid"), build(50, "r", "rid")
    calls = _spy_staircase(monkeypatch)
    actual = left.conditional_join(
        right, ("x", "rx", ops[0]), ("y", "ry", ops[1]), how=how, keep="any"
    )
    assert calls, "the staircase kernel did not run"

    ops_table = {
        "<": np.less,
        "<=": np.less_equal,
        ">": np.greater,
        ">=": np.greater_equal,
    }
    hits = ops_table[ops[0]](
        left["x"].to_numpy()[:, None], right["rx"].to_numpy()[None, :]
    ) & ops_table[ops[1]](
        left["y"].to_numpy()[:, None], right["ry"].to_numpy()[None, :]
    )
    want_left = set(np.flatnonzero(hits.any(axis=1)).tolist())

    paired = actual.dropna(subset=["lid", "rid"])
    assert len(paired) == len(want_left)
    assert set(paired["lid"].astype(int)) == want_left
    assert hits[paired["lid"].astype(int), paired["rid"].astype(int)].all()
    # Carried-along columns must belong to the reported rows.
    assert (
        paired["x"].to_numpy()
        == left.set_index("lid").loc[paired["lid"], "x"].to_numpy()
    ).all()
    assert (
        paired["ry"].to_numpy()
        == right.set_index("rid").loc[paired["rid"], "ry"].to_numpy()
    ).all()
    if how == "left":
        assert set(actual["lid"].astype(int)) == set(range(len(left)))
    if how == "right":
        assert set(actual["rid"].astype(int)) == set(range(len(right)))
