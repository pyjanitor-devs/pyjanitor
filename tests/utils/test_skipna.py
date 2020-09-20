"""Tests for skipna."""
import numpy as np
import pandas as pd
import pytest

from janitor.utils import skipna


@pytest.mark.functions
def test_skipna():
    """
    Overall test for skipna.

    TODO: Should be refactored into separate tests.
    """
    df = pd.DataFrame({"x": ["a", "b", "c", np.nan], "y": [1, 2, 3, np.nan]})

    def func(s):
        """Dummy helper func."""
        return s + "1"

    # Verify that applying function causes error
    with pytest.raises(Exception):
        df["x"].apply(func)

    result = df["x"].apply(skipna(func))
    assert (
        result.to_numpy()[:-1] == np.array(["a1", "b1", "c1"])
    ).all() and np.isnan(result.to_numpy()[-1])
