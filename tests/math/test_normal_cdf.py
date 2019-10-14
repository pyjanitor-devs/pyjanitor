import pandas as pd
import pytest
from scipy.stats import norm


@pytest.mark.functions
def test_normal_cdf():
    s = pd.Series([0, 1, 2, 3, -1])
    out = s.normal_cdf()
    assert (out == norm.cdf(s)).all()
    assert (s.index == out.index).all()
