import pandas as pd
from janitor.utils import skipna
import numpy as np

import pytest


@pytest.mark.functions
def test_skipna():
    df = pd.DataFrame({"x": ["a", "b", "c", np.nan], "y": [1, 2, 3, np.nan]})
    func = lambda s: s + "1"

    # Verify that applying function causes error
    try:
        df["x"].apply(func)
        assert False
    except:
        pass

    result = df["x"].apply(skipna(func))
    assert (
        result.values[:-1] == np.array(["a1", "b1", "c1"])
    ).all() and np.isnan(result.values[-1])
