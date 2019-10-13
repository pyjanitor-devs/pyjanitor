import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import expit


@pytest.mark.functions
def test_sigmoid():
    s = pd.Series([0, 1, 2, 3, -1])
    out = s.sigmoid()
    assert (out == expit(s)).all()
    assert (s.index == out.index).all()
