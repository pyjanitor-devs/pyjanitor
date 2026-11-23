"""Hard interval shapes for equality-plus-overlap conditional joins.

Each shape is checked against a brute-force cross join, for ``conditional_join``
(matched pairs) and ``join_agg`` (per-left-row size and sum), in closed
(``<=`` / ``>=``) and half-open (``<`` / ``>``) form, and for both the
``default`` and ``regions`` range algorithms.

The shapes target boundary behaviour that random data rarely reaches:

- ``heavy_duplicates``: many tied starts and tied ends.
- ``nesting``: every interval contains the next smaller one.
- ``dominant_key``: one equality group holds most of the rows.
- ``point_intervals``: ``start == end`` (closed) or unit width (half-open),
  so intervals touch exactly at the ``<`` vs ``<=`` boundary.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401

N = 120
SHAPES = ["heavy_duplicates", "nesting", "dominant_key", "point_intervals"]
FORMS = {
    # overlap of left [a_start, a_end] and right [b_start, b_end]
    "closed": (">=", "<="),
    "half_open": (">", "<"),
}


def _intervals(shape, rng, n, closed):
    """Return ``(key, start, end)`` arrays with ``start <= end`` (``<`` if
    half-open)."""
    key = rng.integers(0, 4, n)
    if shape == "heavy_duplicates":
        start = rng.choice([0, 5, 10], n)
        end = start + rng.choice([1, 5, 10], n)
    elif shape == "nesting":
        radius = rng.integers(1, n + 1, n)
        start, end = 50 - radius, 50 + radius
    elif shape == "dominant_key":
        key = np.where(rng.random(n) < 0.9, 0, rng.integers(1, 5, n))
        start = rng.integers(0, 40, n)
        end = start + rng.integers(1, 15, n)
    else:  # point_intervals
        start = rng.integers(0, 20, n)
        end = start if closed else start + 1
    return key, start, end


def _frames(shape, seed, closed):
    """Build left and right frames for one shape and seed."""
    rng = np.random.default_rng(seed)
    key, start, end = _intervals(shape, rng, N, closed)
    left = pd.DataFrame(
        {"lid": np.arange(N), "lkey": key, "a_start": start, "a_end": end}
    )
    key, start, end = _intervals(shape, rng, N, closed)
    right = pd.DataFrame(
        {
            "rid": np.arange(N),
            "rkey": key,
            "b_start": start,
            "b_end": end,
            "val": rng.integers(-5, 50, N),
        }
    )
    assert (left.a_start <= left.a_end).all()
    assert (right.b_start <= right.b_end).all()
    if not closed:
        assert (left.a_start < left.a_end).all()
        assert (right.b_start < right.b_end).all()
    return left, right


def _conditions(form):
    """Return the equality-plus-overlap predicates for a form."""
    ge, le = FORMS[form]
    return [
        ("lkey", "rkey", "=="),
        ("a_end", "b_start", ge),
        ("a_start", "b_end", le),
    ]


def _brute_force(left, right, form):
    """Return matching row pairs from a full cross join."""
    pairs = left.merge(right, how="cross")
    start_ok = pairs.a_end >= pairs.b_start
    end_ok = pairs.a_start <= pairs.b_end
    if form == "half_open":
        start_ok = pairs.a_end > pairs.b_start
        end_ok = pairs.a_start < pairs.b_end
    return pairs[(pairs.lkey == pairs.rkey) & start_ok & end_ok]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("algorithm", ["default", "regions"])
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("shape", SHAPES)
def test_hard_shapes_conditional_join_matches_brute_force(shape, form, algorithm, seed):
    """conditional_join pairs equal the brute-force pairs."""
    left, right = _frames(shape, seed, closed=form == "closed")
    expected = _brute_force(left, right, form)
    assert len(expected) > 0  # the shape must actually produce matches
    actual = left.conditional_join(right, *_conditions(form), join_algorithm=algorithm)
    cols = ["lid", "rid"]
    assert_frame_equal(
        actual[cols].sort_values(cols).reset_index(drop=True),
        expected[cols].sort_values(cols).reset_index(drop=True),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("algorithm", ["default", "regions"])
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("shape", SHAPES)
def test_hard_shapes_join_agg_matches_brute_force(shape, form, algorithm, seed):
    """join_agg size and sum equal the brute-force aggregates."""
    left, right = _frames(shape, seed, closed=form == "closed")
    expected_pairs = _brute_force(left, right, form)
    grouped = expected_pairs.groupby("lid")["val"]
    expected = pd.DataFrame({"size": grouped.size(), "sum": grouped.sum()}).reindex(
        left["lid"], fill_value=0
    )
    actual = left.join_agg(
        right,
        *_conditions(form),
        aggfunc=[("val", "size"), ("val", "sum")],
        join_algorithm=algorithm,
    )
    # rows without a match may be absent; they aggregate to the identity
    actual = actual.droplevel(0, axis=1).reindex(range(len(left)), fill_value=0)
    actual.index = left["lid"]
    assert_frame_equal(
        actual[["size", "sum"]].astype("int64"),
        expected[["size", "sum"]].astype("int64"),
        check_names=False,
    )
