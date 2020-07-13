import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def test_str_split():
    "Test wrapper for Pandas ``.str.split()`` method."

    df = pd.DataFrame(
        {"text": ["a_b_c", "c_d_e", np.nan, "f_g_h"], "numbers": range(1, 5)}
    )

    expected = df.copy()
    expected["text"] = expected["text"].str.split("_")

    result = df.process_text(column="text", string_function="split", pat="_")

    assert_frame_equal(result, expected)


def test_str_cat():
    "Test wrapper for Pandas ``.str.cat()`` method."

    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "numbers": range(1, 5)})

    expected = df.copy()
    expected["text"] = expected["text"].str.cat(others=["A", "B", "C", "D"])

    result = df.process_text(
        column="text", string_function="cat", others=["A", "B", "C", "D"],
    )

    assert_frame_equal(result, expected)


def test_str_get():
    """Test wrapper for Pandas ``.str.get()`` method."""

    df = pd.DataFrame(
        {"text": ["aA", "bB", "cC", "dD"], "numbers": range(1, 5)}
    )

    expected = df.copy()
    expected["text"] = expected["text"].str.get(1)
    result = df.process_text(column="text", string_function="get", i=-1)

    assert_frame_equal(result, expected)


def test_str_lower():
    """Test string conversion to lowercase using ``.str.lower()``."""

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

    expected = df.copy()
    expected["names"] = expected["names"].str.lower()

    result = df.process_text(column="names", string_function="lower")

    assert_frame_equal(result, expected)


def test_str_wrong():
    """Test that an invalid Pandas string method raises an exception."""
    df = pd.DataFrame(
        {"text": ["ragnar", "sammywemmy", "ginger"], "code": [1, 2, 3]}
    )
    with pytest.raises(KeyError):
        df.process_text(column="text", string_function="invalid_function")


def test_str_wrong_parameters():
    """Test that invalid argument for Pandas string method raises an error."""

    df = pd.DataFrame(
        {"text": ["a_b_c", "c_d_e", np.nan, "f_g_h"], "numbers": range(1, 5)}
    )

    with pytest.raises(TypeError):
        df.process_text(column="text", string_function="split", pattern="_")
