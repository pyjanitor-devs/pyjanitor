import pandas as pd
import pytest


@pytest.mark.functions
def test_z_score():
    s = pd.Series([0, 1, 2, 3, -1])

    m = s.mean()
    st = s.std()

    ans = (s - m) / st

    d = {}

    assert (s.z_score(moments_dict=d) == ans).all()
    assert (s.z_score().index == s.index).all()

    assert d["mean"] == m
    assert d["std"] == st
