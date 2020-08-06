import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_take_first():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": [0, 1, 2, 3]})

    res = df.take_first(subset="a", by="b")
    exp = df.iloc[[0, 2], :]

    assert_frame_equal(res, exp)
