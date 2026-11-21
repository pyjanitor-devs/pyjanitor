"""Tests for `flag_outliers` function."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from janitor.functions import flag_outliers


@pytest.mark.functions
def test_iqr_flags_outlier():
    """Checks that IQR method correctly flags a clear outlier."""
    df = pd.DataFrame({"values": [10, 12, 11, 13, 100, 9, 11]})
    result = df.flag_outliers(column_name="values")
    assert "values_outlier_flag" in result.columns
    assert result["values_outlier_flag"].iloc[4]
    assert not result["values_outlier_flag"].iloc[0]


@pytest.mark.functions
def test_zscore_flags_outlier():
    """Checks that Z-score method correctly flags a clear outlier."""
    df = pd.DataFrame({"values": [10, 12, 11, 13, 100, 9, 11]})
    result = df.flag_outliers(column_name="values", method="zscore", threshold=2.0)
    assert "values_outlier_flag" in result.columns
    assert result["values_outlier_flag"].iloc[4]


@pytest.mark.functions
def test_no_outliers_all_false():
    """Checks that no rows are flagged when data has no outliers."""
    df = pd.DataFrame({"values": [10, 11, 12, 11, 10, 12, 11]})
    result = df.flag_outliers(column_name="values")
    assert result["values_outlier_flag"].sum() == 0


@pytest.mark.functions
def test_custom_flag_column_name():
    """Checks that custom flag column name is used correctly."""
    df = pd.DataFrame({"values": [10, 12, 11, 100]})
    result = df.flag_outliers(column_name="values", flag_column_name="is_outlier")
    assert "is_outlier" in result.columns
    assert "values_outlier_flag" not in result.columns


@pytest.mark.functions
def test_does_not_mutate_original():
    """Checks that the original DataFrame is not mutated."""
    df = pd.DataFrame({"values": [10, 12, 11, 100]})
    original = df.copy()
    df.flag_outliers(column_name="values")
    assert_frame_equal(df, original)


@pytest.mark.functions
def test_non_method_functional():
    """Checks behaviour when flag_outliers is used as a function."""
    df = pd.DataFrame({"values": [10, 12, 11, 100]})
    result = flag_outliers(df, column_name="values")
    assert "values_outlier_flag" in result.columns


@pytest.mark.functions
def test_fail_invalid_method():
    """Checks that ValueError is raised for an invalid method."""
    df = pd.DataFrame({"values": [10, 12, 11, 100]})
    with pytest.raises(ValueError):
        df.flag_outliers(column_name="values", method="invalid")


@pytest.mark.functions
def test_fail_negative_threshold():
    """Checks that ValueError is raised for a non-positive threshold."""
    df = pd.DataFrame({"values": [10, 12, 11, 100]})
    with pytest.raises(ValueError):
        df.flag_outliers(column_name="values", threshold=-1.0)


@pytest.mark.functions
def test_fail_non_numeric_column():
    """Checks that TypeError is raised for a non-numeric column."""
    df = pd.DataFrame({"names": ["alice", "bob", "charlie"]})
    with pytest.raises(TypeError):
        df.flag_outliers(column_name="names")


@pytest.mark.functions
def test_fail_column_not_in_df():
    """Checks that ValueError is raised when column is not in DataFrame."""
    df = pd.DataFrame({"values": [10, 12, 11, 100]})
    with pytest.raises(ValueError):
        df.flag_outliers(column_name="nonexistent")


@pytest.mark.functions
def test_fail_flag_column_already_exists():
    """Checks that ValueError is raised when flag column already exists."""
    df = pd.DataFrame({"values": [10, 12, 11, 100], "values_outlier_flag": [0, 0, 0, 0]})
    with pytest.raises(ValueError):
        df.flag_outliers(column_name="values")


@pytest.mark.functions
def test_zscore_constant_column():
    """Checks that a constant column (std=0) produces no outliers."""
    df = pd.DataFrame({"values": [5, 5, 5, 5, 5]})
    result = df.flag_outliers(column_name="values", method="zscore")
    assert result["values_outlier_flag"].sum() == 0
