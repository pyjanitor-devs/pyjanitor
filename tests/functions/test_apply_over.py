import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_apply_over():
    df = pd.DataFrame({"a": ["dog", "dog", "cat", "cat"], "b": [1, 2, 3, 4]})

    res = df.apply_over(func=np.max, col="b", by="a", name="max")
    exp = pd.concat((df, pd.Series([2, 2, 4, 4], name="max")), axis=1)

    assert res == exp
