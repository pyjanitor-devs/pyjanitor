import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df():
    return pd.DataFrame(
        [
            {"age": 22, "major": "science", "ID": 145, "result": "pass"},
            {"age": 20, "major": "history", "ID": 322, "result": "fail"},
            {"age": 23, "major": "history", "ID": 321, "result": "pass"},
            {"age": 21, "major": "law", "ID": 49, "result": "fail"},
            {"age": 19, "major": "mathematics", "ID": 224, "result": "pass"},
            {"age": 20, "major": "science", "ID": 132, "result": "pass"},
        ]
    )


def test_ascending_groupby_k_2(df):
    """Test ascending group by, k=2"""
    expected = df.groupby("result").apply(
        lambda d: d.sort_values("age").head(2)
    )
    assert_frame_equal(df.groupby_topk("result", "age", 2), expected)


def test_descending_groupby_k_3(df):
    """Test descending group by, k=3"""
    expected = df.groupby("result").apply(
        lambda d: d.sort_values("age", ascending=False).head(3)
    )
    assert_frame_equal(
        df.groupby_topk("result", "age", 3, {"ascending": False}), expected
    )


def test_wrong_groupby_column_name(df):
    """Raise Value Error if wrong groupby column name is provided."""
    with pytest.raises(KeyError):
        df.groupby_topk("RESULT", "age", 3)


def test_wrong_sort_column_name(df):
    """Raise Value Error if wrong sort column name is provided."""
    with pytest.raises(KeyError):
        df.groupby_topk("result", "Age", 3)


def test_negative_k(df):
    """Raises Value Error if k is less than 1 (negative or 0)."""
    with pytest.raises(ValueError):
        df.groupby_topk("result", "age", -2)
    with pytest.raises(ValueError):
        df.groupby_topk("result", "age", 0)


def test_same_sort_groupby_columns(df):
    """Raises Value Error if columns to group by and sort along are same."""
    with pytest.raises(ValueError):
        df.groupby_topk("result", "result", 2)


def test_inplace(df):
    """Raise Key Error if inplace is True in sort_values_kwargs"""
    with pytest.raises(KeyError):
        df.groupby_topk("result", "age", 1, {"inplace": True})
