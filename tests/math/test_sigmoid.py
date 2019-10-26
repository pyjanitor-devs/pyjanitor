import pandas as pd
import pytest
from scipy.special import expit


@pytest.mark.functions
def test_sigmoid():
    s = pd.Series([0, 1, 2, 3, -1])
    out = s.sigmoid()
    assert (out == expit(s)).all()
    assert (s.index == out.index).all()
