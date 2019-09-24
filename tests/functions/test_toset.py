import pandas as pd
import pytest


@pytest.mark.functions
def test_coalesce_with_title():
    s = pd.Series([1, 2, 3, 5, 5], index=["a", "b", "c", "d", "e"]).toset()

    assert isinstance(s, set)
    assert len(s) == 4
    assert s == set([1, 2, 3, 5])
