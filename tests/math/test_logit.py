import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_logit():
    s = pd.Series([0, 0.1, 0.2, 0.3, 0.5, 0.9, 1, 2])
    inside = (0 < s) & (s < 1)
    valid = np.array([0.1, 0.2, 0.3, 0.5, 0.9])
    ans = np.log(valid / (1 - valid))

    with pytest.raises(RuntimeError):
        s.logit(error="raise")

    with pytest.warns(RuntimeWarning):
        out = s.logit(error="warn")

    assert out[inside].notna().all()
    assert out[inside].to_numpy() == pytest.approx(ans)
    assert (out.index == s.index).all()
    assert out[~inside].isna().all()

    out = s.logit(error="ignore")

    assert out[inside].notna().all()
    assert out[inside].to_numpy() == pytest.approx(ans)
    assert (out.index == s.index).all()
    assert out[~inside].isna().all()
