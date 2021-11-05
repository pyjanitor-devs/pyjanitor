import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df():
    "Base DataFrame fixture"
    return pd.DataFrame(
        {"a": [1, np.nan, 3], "b": [2, 3, 1], "c": [2, np.nan, 9]}
    )


@pytest.mark.xfail(reason="column_names is a variable args")
def test_wrong_type_column_names(df):
    """Raise Error if wrong type is provided for `column_names`."""
    with pytest.raises(TypeError):
        df.coalesce("a", "b")


def test_wrong_type_target_column_name(df):
    """Raise TypeError if wrong type is provided for `target_column_name`."""
    with pytest.raises(TypeError):
        df.coalesce("a", "b", target_column_name=["new_name"])


def test_wrong_type_default_value(df):
    """Raise TypeError if wrong type is provided for `default_value`."""
    with pytest.raises(TypeError):
        df.coalesce(
            "a", "b", target_column_name="new_name", default_value=[1, 2, 3]
        )


def test_len_column_names_less_than_2(df):
    """Raise Error if column_names length is less than 2."""
    with pytest.raises(ValueError):
        df.coalesce("a")


def test_empty_column_names(df):
    """Return dataframe if `column_names` is empty."""
    assert_frame_equal(df.coalesce(), df)


@pytest.mark.functions
def test_coalesce_without_target(df):
    """Test output if `target_column_name` is not provided."""
    result = df.coalesce("a", "b", "c")
    expected_output = df.assign(
        a=df["a"].combine_first(df["b"].combine_first(df["c"]))
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.functions
def test_coalesce_without_delete():
    """Test ouptut if nulls remain and `default_value` is provided."""
    df = pd.DataFrame(
        {"s1": [np.nan, np.nan, 6, 9, 9], "s2": [np.nan, 8, 7, 9, 9]}
    )
    expected = df.assign(s3=df.s1.combine_first(df.s2).fillna(0))
    result = df.coalesce("s1", "s2", target_column_name="s3", default_value=0)
    assert_frame_equal(result, expected)
