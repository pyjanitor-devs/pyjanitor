"""Tests for skiperror."""
import numpy as np
import pandas as pd
import pytest

from janitor.utils import skiperror


@pytest.mark.functions
def test_skiperror():
    """
    Overall test for skiperror.

    TODO: I believe this test should be refactored into smaller "unit" tests.
    """
    df = pd.DataFrame({"x": [1, 2, 3, "a"], "y": [1, 2, 3, "b"]})

    def func(s):
        """Dummy helper function."""
        return s + 1

    # Verify that applying function causes error
    with pytest.raises(Exception):
        df["x"].apply(func)

    result = df["x"].apply(skiperror(func))
    assert (result.to_numpy()[:-1] == np.array([2, 3, 4])).all() and np.isnan(
        result.to_numpy()[-1]
    )

    result = df["x"].apply(skiperror(func, return_x=True))
    assert (result.to_numpy() == np.array([2, 3, 4, "a"], dtype=object)).all()

    result = df["x"].apply(skiperror(func, return_x=False, return_val=5))
    assert (result.to_numpy() == np.array([2, 3, 4, 5])).all()
