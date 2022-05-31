import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df():
    """fixture for groupby_topk"""
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


def test_dtype_by(df):
    """Check dtype for by."""
    with pytest.raises(TypeError):
        df.groupby_topk(by={"result"}, column="age", k=2)


def test_ascending_groupby_k_2(df):
    """Test ascending group by, k=2"""
    expected = (
        df.groupby("result", sort=False)
        .apply(lambda d: d.sort_values("age").head(2))
        .droplevel(0)
    )
    assert_frame_equal(
        df.groupby_topk("result", "age", 2, ignore_index=False), expected
    )


def test_ascending_groupby_non_numeric(df):
    """Test output for non-numeric column"""
    expected = (
        df.groupby("result", sort=False)
        .apply(lambda d: d.sort_values("major").head(2))
        .droplevel(0)
    )
    assert_frame_equal(
        df.groupby_topk("result", "major", 2, ignore_index=False), expected
    )


def test_descending_groupby_k_3(df):
    """Test descending group by, k=3"""
    expected = (
        df.groupby("result", sort=False)
        .apply(lambda d: d.sort_values("age", ascending=False).head(3))
        .droplevel(0)
        .reset_index(drop=True)
    )
    assert_frame_equal(
        df.groupby_topk("result", "age", 3, ascending=False), expected
    )


def test_wrong_groupby_column_name(df):
    """Raise Value Error if wrong groupby column name is provided."""
    with pytest.raises(
        ValueError, match="RESULT not present in dataframe columns!"
    ):
        df.groupby_topk("RESULT", "age", 3)


def test_wrong_sort_column_name(df):
    """Raise Value Error if wrong sort column name is provided."""
    with pytest.raises(
        ValueError, match="Age not present in dataframe columns!"
    ):
        df.groupby_topk("result", "Age", 3)


def test_negative_k(df):
    """Raises Value Error if k is less than 1 (negative or 0)."""
    with pytest.raises(
        ValueError,
        match="Numbers of rows per group.+",
    ):
        df.groupby_topk("result", "age", -2)


@pytest.mark.xfail(reason="sort_value_kwargs parameter deprecated.")
def test_inplace(df):
    """Raise Key Error if inplace is True in sort_values_kwargs"""
    with pytest.raises(KeyError):
        df.groupby_topk("result", "age", 1, {"inplace": True})
