import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def test_str_split():
    "Test Pandas string split method"

    df = pd.DataFrame(
        {"text": ["a_b_c", "c_d_e", np.nan, "f_g_h"], "numbers": range(1, 5)}
    )

    result = df.string.split(column="text", pat="_")

    expected = pd.DataFrame(
        {
            "text": [
                ["a", "b", "c"],
                ["c", "d", "e"],
                np.nan,
                ["f", "g", "h"],
            ],
            "numbers": [1, 2, 3, 4],
        }
    )

    assert_frame_equal(result, expected)


def test_str_concat():
    "Test Pandas string cat method"

    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "numbers": range(1, 5)})

    result = df.string.concat(column="text", others=["A", "B", "C", "D"])

    expected = pd.DataFrame(
        {"text": ["aA", "bB", "cC", "dD"], "numbers": [1, 2, 3, 4]}
    )

    assert_frame_equal(result, expected)


def test_str_get():
    """Test Pandas string get method"""

    df = pd.DataFrame(
        {"text": ["aA", "bB", "cC", "dD"], "numbers": range(1, 5)}
    )

    result = df.string.get("text", -1)

    expected = pd.DataFrame(
        {"text": ["A", "B", "C", "D"], "numbers": [1, 2, 3, 4]}
    )

    assert_frame_equal(result, expected)


box = pd.DataFrame({"text": ["aA", "bB", "cC", "dD"], "numbers": range(1, 5)})
