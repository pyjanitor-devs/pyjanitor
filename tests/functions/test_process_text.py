import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {"text": ["a_b_c", "c_d_e", np.nan, "f_g_h"], "numbers": range(1, 5)}
    )


def test_column_name_type(test_df):
    """Raise TypeError if `column_name` type is not `str`."""
    with pytest.raises(TypeError):
        test_df.process_text(["test"])


def test_new_column_name_type(test_df):
    """Raise TypeError if `new_column_name` type is not `str`."""
    with pytest.raises(TypeError):
        test_df.process_text(column_name="text", new_column_name=["new_text"])


def test_column_name_presence(test_df):
    """Raise ValueError if `column_name` is not in dataframe."""
    with pytest.raises(ValueError):
        test_df.process_text(column_name="Test")


def test_str_split(test_df):
    """Test wrapper for Pandas ``str.split()`` method."""

    expected = test_df.assign(text=test_df["text"].str.split("_"))

    result = test_df.process_text(
        column_name="text", string_function="split", pat="_"
    )

    assert_frame_equal(result, expected)


def test_new_column_names(test_df):
    """
    Test that a new column name is created when
    `new_column_name` is not None.
    """
    result = test_df.process_text(
        column_name="text",
        new_column_names="new_text",
        string_function="slice",
        start=2,
    )
    expected = test_df.assign(new_text=test_df["text"].str.slice(start=2))
    assert_frame_equal(result, expected)


def test_str_cat():
    "Test wrapper for Pandas ``.str.cat()`` method."

    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "numbers": range(1, 5)})

    result = df.process_text(
        column_name="text", string_function="cat", others=["A", "B", "C", "D"],
    )

    expected = df.assign(text=df["text"].str.cat(others=["A", "B", "C", "D"]))

    assert_frame_equal(result, expected)


def test_str_get():
    """Test wrapper for Pandas ``.str.get()`` method."""

    df = pd.DataFrame(
        {"text": ["aA", "bB", "cC", "dD"], "numbers": range(1, 5)}
    )

    expected = df.assign(text=df["text"].str.get(1))

    result = df.process_text(column_name="text", string_function="get", i=-1)

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

    expected = df.assign(names=df["names"].str.lower())

    result = df.process_text(column_name="names", string_function="lower")

    assert_frame_equal(result, expected)


def test_str_wrong():
    """Test that an invalid Pandas string method raises an exception."""
    df = pd.DataFrame(
        {"text": ["ragnar", "sammywemmy", "ginger"], "code": [1, 2, 3]}
    )
    with pytest.raises(KeyError):
        df.process_text(column_name="text", string_function="invalid_function")


def test_str_wrong_parameters(test_df):
    """Test that invalid argument for Pandas string method raises an error."""
    with pytest.raises(TypeError):
        test_df.process_text(
            column_name="text", string_function="split", pattern="_"
        )

import janitor
s = pd.Series(['a', 'b', np.nan, 'd'], index=["A", "B", "C","A"])
df = pd.DataFrame()
df = df.assign(text = s, number = 1)
t = pd.Series(['d', 'a', 'e', 'c'], index=[3, 0, 4, 2])

result =df.process_text("text",  merge_frame=True, new_column_names = "newtext_", string_function="cat", na_rep='-', others=['A', 'B', 'C', 'D'])
res = df.text.str.cat(t, join='outer', na_rep='-')

print(df, end="\n\n")
print(res)