import pandas as pd
import numpy as np


def test_coalesce():
    df = pd.DataFrame(
        {"a": [1, np.nan, 3], "b": [2, 3, 1], "c": [2, np.nan, 9]}
    ).coalesce(["a", "b", "c"], "a")
    assert df.shape == (3, 1)
    assert pd.isnull(df).sum().sum() == 0
