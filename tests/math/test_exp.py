import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_exp():
    s = pd.Series([0, 1, 2, 3, -1])
    out = s.exp()
    assert (out == np.exp(s)).all()
    assert (s.index == out.index).all()
