import pandas as pd
from pandas.testing import assert_frame_equal

from janitor.functions import case_when


def test_update_where_query():
    """Test that function works with pandas query-style string expression."""
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, "hi"],
            "b": [0, 3, 4, 5, "bye"],
            "c": [6, 7, 8, 9, "wait"],
        }
    )
    expected = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, "hi"],
            "b": [0, 3, 4, 5, "bye"],
            "c": [6, 7, 8, 9, "wait"],
            "value": ["x", 0, 8, 9, "hi"],
        }
    )
    result = case_when(
        df,
        "value",
        [((df.a == 0) & (df.b != 0)) | (df.c == "wait"), df.a],
        [(df.b == 0) & (df.a == 0), "x"],
        [True, df.c],
    )

    assert_frame_equal(result, expected)
