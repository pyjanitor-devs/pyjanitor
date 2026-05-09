import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
from scipy.stats import norm


@pytest.mark.functions
def test_normal_cdf():
    s = pd.Series([0, 1, 2, 3, -1])
    out = s.normal_cdf()
    assert (out == norm.cdf(s)).all()
    assert (s.index == out.index).all()


@pytest.mark.functions
def test_normal_cdf_matches_ndtr_exactly():
    """``normal_cdf`` should yield exactly ``scipy.special.ndtr(s)``.

    ``norm.cdf`` is itself a thin wrapper around ``ndtr``, so going direct
    must produce bit-identical output (no float drift). This pin exists so
    a future refactor can't silently swap to a different (e.g. ``erf``-based)
    kernel and change the numerics. Issue #1468.
    """
    s = pd.Series(np.linspace(-5, 5, 101), name="x")
    out = s.normal_cdf()
    expected = ndtr(s.to_numpy())
    np.testing.assert_array_equal(out.to_numpy(), expected)
