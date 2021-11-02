import pandas as pd
import pytest
from scipy.special import softmax as scipy_softmax


@pytest.mark.functions
def test_softmax():
    s = pd.Series([0, 1, 2, 3, -1])
    out = s.softmax()
    assert (out == scipy_softmax(s)).all()
    assert (s.index == out.index).all()
    assert out.sum() == 1.0
