import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_log():
    s = pd.Series([0, 1, 2, 3, -1])

    with pytest.raises(RuntimeError):
        s.log(error="raise")

    with pytest.warns(RuntimeWarning):
        out = s.log(error="warn")

    assert out[s <= 0].isna().all()
    assert (out.index == s.index).all()
    assert (out[s > 0] == np.log(np.array([1, 2, 3]))).all()

    out = s.log(error="ignore")

    assert out[s <= 0].isna().all()
    assert (out.index == s.index).all()
    assert (out[s > 0] == np.log(np.array([1, 2, 3]))).all()
