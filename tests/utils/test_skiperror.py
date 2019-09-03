import pandas as pd
from janitor.utils import skiperror
import numpy as np

import pytest


@pytest.mark.functions
def test_skiperror():
    df = pd.DataFrame({"x": [1, 2, 3, "a"], "y": [1, 2, 3, "b"]})

    def func(s):
        return s + 1

    # Verify that applying function causes error
    with pytest.raises(Exception):
        df["x"].apply(func)

    result = df["x"].apply(skiperror(func))
    assert (result.values[:-1] == np.array([2, 3, 4])).all() and np.isnan(
        result.values[-1]
    )

    result = df["x"].apply(skiperror(func, return_x=True))
    assert (result.values == np.array([2, 3, 4, "a"], dtype=object)).all()

    result = df["x"].apply(skiperror(func, return_x=False, return_val=5))
    assert (result.values == np.array([2, 3, 4, 5])).all()
