"""
Author: Vamsi Krishna
Date: 23 July 2021

The intent of these tests is to test factorize_columns function works.
Because underneath the hood we are using `pd.factorize`,
we intentionally do not test the values of the resultant dataframe.
That would be duplicating the tests from what the `pandas` library provides.
"""
import pandas as pd
import pytest


@pytest.mark.functions
def test_single_column_factorize_columns():
    """Tests if Single Column Factorize is working correctly"""
    df = pd.DataFrame(
        {"a": ["hello", "hello", "sup"], "b": [1, 2, 3]}
    ).factorize_columns(column_names="a")
    assert "a_enc" in df.columns


@pytest.mark.functions
def test_single_column_fail_factorize_columns():
    """Tests if Single Column Factorize fails"""
    with pytest.raises(ValueError):
        pd.DataFrame(
            {"a": ["hello", "hello", "sup"], "b": [1, 2, 3]}
        ).factorize_columns(
            column_names="c"
        )  # noqa: 841


@pytest.mark.functions
def test_multicolumn_factorize_columns():
    """Tests if Multi Column Factorize is working correctly"""
    df = pd.DataFrame(
        {
            "a": ["hello", "hello", "sup"],
            "b": [1, 2, 3],
            "c": ["aloha", "nihao", "nihao"],
        }
    ).factorize_columns(column_names=["a", "c"])
    assert "a_enc" in df.columns
    assert "c_enc" in df.columns


@pytest.mark.functions
def test_multicolumn_factorize_columns_suffix_change():
    """Tests if Multi Column Factorize works with suffix change"""
    df = pd.DataFrame(
        {
            "a": ["hello", "hello", "sup"],
            "b": [1, 2, 3],
            "c": ["aloha", "nihao", "nihao"],
        }
    ).factorize_columns(column_names=["a", "c"], suffix="_col")
    assert "a_col" in df.columns
    assert "c_col" in df.columns
    assert "a_enc" not in df.columns
    assert "c_enc" not in df.columns


@pytest.mark.functions
def test_multicolumn_factorize_columns_empty_suffix():
    """Tests if Multi Column Factorize works with empty suffix"""
    df = pd.DataFrame(
        {
            "a": ["hello", "hello", "sup"],
            "b": [1, 2, 3],
            "c": ["aloha", "nihao", "nihao"],
        }
    ).factorize_columns(column_names=["a", "c"], suffix="")
    assert "a_enc" not in df.columns
    assert "c_enc" not in df.columns
    assert 3 == len(df.columns)


@pytest.mark.functions
def test_factorize_columns_invalid_input(dataframe):
    """Tests if Multi Column Factorize throws error"""
    with pytest.raises(NotImplementedError):
        dataframe.factorize_columns(1)
