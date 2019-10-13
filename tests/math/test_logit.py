import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.functions
def test_logit():
    s = pd.Series([0, 0.1, 0.2, 0.3, 1, 2])
    inside = (0 < s) & (s < 1)
    valid = np.array([0.1, 0.2, 0.3])
    ans = np.log(valid / (1 - valid))

    with pytest.raises(RuntimeError):
        s.logit(error="raise")

    with pytest.warns(RuntimeWarning):
        out = s.logit(error="warn")

    assert out[inside].notnull().all()
    assert (out[inside] == ans).all()
    assert (out.index == s.index).all()
    assert out[~inside].isnull().all()

    out = s.logit(error="ignore")

    assert out[inside].notnull().all()
    assert (out[inside] == ans).all()
    assert (out.index == s.index).all()
    assert out[~inside].isnull().all()
