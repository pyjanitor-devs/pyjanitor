import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401


@pytest.mark.functions
def test_join_apply():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}).join_apply(
        lambda x: 2 * x["a"] + x["b"], new_column_name="2a+b"
    )

    expected = df.copy()
    expected["2a+b"] = [4, 7, 10]

    assert_frame_equal(df, expected)
