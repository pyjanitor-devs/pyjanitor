import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


@pytest.mark.functions
def test_probit():
    s = pd.Series([-1, 0, 0.1, 0.2, 0.3, 1, 2])
    inside = (0 < s) & (s < 1)
    valid = np.array([0.1, 0.2, 0.3])
    ans = norm.ppf(valid)

    with pytest.raises(RuntimeError):
        s.probit(error="raise")

    with pytest.warns(RuntimeWarning):
        out = s.probit(error="warn")

    assert out[inside].notna().all()
    assert (out[inside] == ans).all()
    assert (out.index == s.index).all()
    assert out[~inside].isna().all()

    out = s.probit(error="ignore")

    assert out[inside].notna().all()
    assert (out[inside] == ans).all()
    assert (out.index == s.index).all()
    assert out[~inside].isna().all()
