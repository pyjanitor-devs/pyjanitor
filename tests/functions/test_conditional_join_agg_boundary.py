"""Focused compatibility tests for the conditional_join <-> janitor_rs
reverse-aggregation boundary.

These call the private `_agg_functions` wrappers directly with small,
hand-computed fixtures instead of going through the full `conditional_join`
condition-selection machinery (already covered by the broader `_agg_rev`
tests in `test_conditional_join.py`). The point here is narrower: confirm
that `_agg_functions`/`_get_join_aggs` and the currently pinned janitor-rs
build agree on the call signature -- i.e. that a caller-supplied `length`
that janitor-rs no longer accepts hasn't been left behind on the Python
side (or vice versa). Each test covers one reverse-aggregation shape.
"""

import numpy as np
import pytest

from janitor.functions._conditional_join import _agg_functions


def test_sum_rev_no_ranges():
    """Equi-join (no_range) shape."""
    arr = np.array([5, 3], dtype=np.int64)
    left_index = np.array([0, 1], dtype=np.int64)
    right_index = np.array([10, 10], dtype=np.int64)
    booleans = np.array([False, False])

    index, out = _agg_functions._sum_rev_no_ranges(
        arr=arr,
        left_index=left_index,
        right_index=right_index,
        booleans=booleans,
    )

    assert list(index) == [10]
    assert list(out) == [8]


def test_prod_rev_starts():
    """Suffix (`starts`) shape."""
    arr = np.array([2, 3], dtype=np.int64)
    starts = np.array([0, 1], dtype=np.int64)
    index = np.array([20, 10, 90], dtype=np.int64)
    booleans = np.array([False, False])

    labels, out = _agg_functions._prod_rev_starts(
        arr=arr, starts=starts, index=index, booleans=booleans
    )

    assert list(labels) == [20, 10, 90]
    assert list(out) == [2, 6, 6]


def test_min_rev_ends():
    """Prefix (`ends`) shape."""
    arr = np.array([5, 2, 4], dtype=np.int64)
    ends = np.array([2, 3, 1], dtype=np.int64)
    index = np.array([50, 10, 90], dtype=np.int64)
    booleans = np.array([False, False, False])

    labels, positions = _agg_functions._min_rev_ends(
        arr=arr, ends=ends, index=index, booleans=booleans
    )

    assert list(labels) == [50, 10, 90]
    assert list(positions) == [1, 1, 1]


def test_max_rev_starts_matches():
    """`starts` + candidate-tape (`matches`) shape."""
    arr = np.array([5, 2], dtype=np.int64)
    starts = np.array([0, 0], dtype=np.int64)
    counts = np.array([1, 1], dtype=np.int64)
    index = np.array([10, 20], dtype=np.int64)
    matches = np.array([1, 1, 1, 1], dtype=np.int8)
    booleans = np.array([False, False])

    labels, positions = _agg_functions._max_rev_starts_matches(
        arr=arr,
        starts=starts,
        counts=counts,
        index=index,
        matches=matches,
        booleans=booleans,
    )

    assert list(labels) == [10, 20]
    assert list(positions) == [0, 0]


def test_sum_rev_ends_matches():
    """`ends` + candidate-tape (`matches`) shape."""
    arr = np.array([5, 7], dtype=np.int64)
    index = np.array([10, 30], dtype=np.int64)
    ends = np.array([2, 1], dtype=np.int64)
    counts = np.array([1, 1], dtype=np.int64)
    matches = np.array([1, 1, 1], dtype=np.int8)
    booleans = np.array([False, False])

    labels, out = _agg_functions._sum_rev_ends_matches(
        arr=arr,
        index=index,
        ends=ends,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )

    assert list(labels) == [10, 30]
    assert list(out) == [12, 5]


def test_prod_rev_starts_ends():
    """Dual-bound (`starts_ends`) shape."""
    arr = np.array([2, 3, 4], dtype=np.int64)
    starts = np.array([0, 1, 0], dtype=np.int64)
    ends = np.array([2, 3, 1], dtype=np.int64)
    index = np.array([10, 20, 10], dtype=np.int64)
    booleans = np.array([False, False, False])

    labels, out = _agg_functions._prod_rev_starts_ends(
        arr=arr, starts=starts, ends=ends, index=index, booleans=booleans
    )

    assert list(labels) == [10, 20]
    assert list(out) == [24, 6]


def test_sum_rev_starts_ends_matches():
    """Dual-bound + candidate-tape (`starts_ends_matches`) shape."""
    arr = np.array([1, 2, 3], dtype=np.int64)
    starts = np.array([0, 1, 0], dtype=np.int64)
    ends = np.array([2, 3, 1], dtype=np.int64)
    index = np.array([10, 20, 10], dtype=np.int64)
    counts = np.array([1, 1, 1], dtype=np.int64)
    matches = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    booleans = np.array([False, False, False])

    labels, out = _agg_functions._sum_rev_starts_ends_matches(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        counts=counts,
        matches=matches,
        booleans=booleans,
    )

    assert list(labels) == [10, 20]
    assert list(out) == [6, 3]


def test_min_rev_positions():
    """Indirect-range (`positions`) shape."""
    arr = np.array([5], dtype=np.int64)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([1], dtype=np.int64)
    index = np.array([10], dtype=np.int64)
    positions = np.array([0], dtype=np.int64)
    booleans = np.array([False])

    labels, out = _agg_functions._min_rev_positions(
        arr=arr,
        starts=starts,
        ends=ends,
        index=index,
        positions=positions,
        booleans=booleans,
    )

    assert list(labels) == [10]
    assert list(out) == [0]


def test_size_rev_starts():
    """`size` agg, `starts` shape (no `arr`/`booleans` at all)."""
    starts = np.array([1, 0, 2], dtype=np.int64)
    index = np.array([50, 10, 90], dtype=np.int64)

    labels, out = _agg_functions._size_rev_starts(starts=starts, index=index)

    assert list(labels) == [50, 10, 90]
    assert list(out) == [1, 2, 3]


def test_size_rev_positions():
    """`size` agg, `positions` shape."""
    starts = np.array([0], dtype=np.int64)
    ends = np.array([1], dtype=np.int64)
    index = np.array([10], dtype=np.int64)
    positions = np.array([0], dtype=np.int64)

    labels, out = _agg_functions._size_rev_positions(
        starts=starts, ends=ends, index=index, positions=positions
    )

    assert list(labels) == [10]
    assert list(out) == [1]


@pytest.mark.parametrize(
    "fn_name",
    [
        "_sum_rev_no_ranges",
        "_prod_rev_no_ranges",
        "_min_rev_no_ranges",
        "_max_rev_no_ranges",
        "_sum_rev_starts",
        "_sum_rev_ends",
        "_min_rev_starts",
        "_min_rev_ends",
        "_max_rev_starts",
        "_max_rev_ends",
        "_prod_rev_starts",
        "_prod_rev_ends",
        "_sum_rev_starts_matches",
        "_sum_rev_ends_matches",
        "_min_rev_starts_matches",
        "_min_rev_ends_matches",
        "_max_rev_starts_matches",
        "_max_rev_ends_matches",
        "_prod_rev_starts_matches",
        "_prod_rev_ends_matches",
        "_sum_rev_starts_ends",
        "_min_rev_starts_ends",
        "_max_rev_starts_ends",
        "_prod_rev_starts_ends",
        "_sum_rev_starts_ends_matches",
        "_min_rev_starts_ends_matches",
        "_max_rev_starts_ends_matches",
        "_prod_rev_starts_ends_matches",
        "_sum_rev_positions",
        "_min_rev_positions",
        "_max_rev_positions",
        "_prod_rev_positions",
        "_size_rev_starts",
        "_size_rev_ends",
        "_size_rev_starts_matches",
        "_size_rev_ends_matches",
        "_size_rev_starts_ends",
        "_size_rev_starts_ends_matches",
        "_size_rev_positions",
    ],
)
def test_no_reverse_agg_function_still_declares_length(fn_name):
    """None of these wrappers should carry a `length` parameter any more.

    A regression here means either the Python wrapper or the pinned
    janitor-rs build has drifted back to requiring/accepting `length`
    without the other side following -- exactly the class of bug this
    module exists to catch.
    """
    fn = getattr(_agg_functions, fn_name)
    params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    assert "length" not in params, f"{fn_name} still declares a length param"
