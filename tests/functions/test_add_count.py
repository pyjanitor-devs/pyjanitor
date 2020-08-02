import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "name": ("black", "black", "black", "red", "red"),
            "type": ("chair", "chair", "sofa", "sofa", "plate"),
            "num": (4, 5, 12, 4, 3),
            "nulls": (1, 1, np.nan, np.nan, 3),
        }
    )


def test_add_count_one_column(df):
    """Group by one column and add count per row """
    expected = df.copy()
    expected["n"] = expected.groupby("name")["name"].transform("size")
    assert_frame_equal(df.add_count(by="name", name="n"), expected)


def test_add_count_multiple_columns(df):
    """Group by multiple columns and add count per row """
    columns = ["name", "type"]
    expected = df.copy()
    expected["n"] = expected.groupby(["name", "type"])["type"].transform(
        "size"
    )
    assert_frame_equal(df.add_count(by=columns, name="n"), expected)


def test_add_count_null_columns(df):
    """Group on null column """
    expected = df.copy()
    expected["n"] = (
        expected.fillna("rar").groupby(["nulls"])["type"].transform("size")
    )
    assert_frame_equal(df.add_count(by="nulls", name="n"), expected)
