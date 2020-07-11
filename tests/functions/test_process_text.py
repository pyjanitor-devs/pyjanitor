import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions import process_text


def test_str_split():
    "Test wrapper for Pandas ``.str.split()`` method"

    df = pd.DataFrame(
        {"text": ["a_b_c", "c_d_e", np.nan, "f_g_h"], "numbers": range(1, 5)}
    )

    result = df.process_text(
        column="text", string_function="split", pat="_"
    )

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


def test_str_cat():
    "Test wrapper for Pandas ``.str.cat()`` method"

    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "numbers": range(1, 5)})

    result = df.process_text(
        column="text",
        string_function="cat",
        others=["A", "B", "C", "D"],
    )

    expected = pd.DataFrame(
        {"text": ["aA", "bB", "cC", "dD"], "numbers": [1, 2, 3, 4]}
    )

    assert_frame_equal(result, expected)


def test_str_get():
    """Test wrapper for Pandas ``.str.get()`` method"""

    df = pd.DataFrame(
        {"text": ["aA", "bB", "cC", "dD"], "numbers": range(1, 5)}
    )

    result = process_text(df=df, column="text", string_function="get", i=-1)

    expected = pd.DataFrame(
        {"text": ["A", "B", "C", "D"], "numbers": [1, 2, 3, 4]}
    )

    assert_frame_equal(result, expected)


def test_str_lower():
    """Test string conversion to lowercase using ``.str.lower()``"""

    df = pd.DataFrame(
        {
            "codes": range(1, 7),
            "names": [
                "Graham Chapman",
                "John Cleese",
                "Terry Gilliam",
                "Eric Idle",
                "Terry Jones",
                "Michael Palin",
            ],
        }
    )

    expected = pd.DataFrame(
        {
            "codes": range(1, 7),
            "names": [
                "graham chapman",
                "john cleese",
                "terry gilliam",
                "eric idle",
                "terry jones",
                "michael palin",
            ],
        }
    )

    result = process_text(df, column="names", string_function="lower")

    assert_frame_equal(result, expected)


def test_str_wrong():
    """Test that string_function that is not a Pandas string method actually errors out"""
    df = pd.DataFrame(
        {"text": ["ragnar", "sammywemmy", "ginger"], "code": [1, 2, 3]}
    )
    with pytest.raises(KeyError):
        process_text(df, column="text", string_function="ragnar")
