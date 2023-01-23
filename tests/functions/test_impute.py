"""Tests for the `impute` functions"""
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_impute_single_value(missingdata_df):
    """Check if constant value is imputed correctly."""
    df = missingdata_df.impute("a", 5)
    assert set(df["a"]) == set([1, 2, 5])


@pytest.mark.functions
def test_impute_single_value_multiple_columns(missingdata_df):
    """Check if constant value is imputed correctly."""
    df = missingdata_df.impute(["a", "Bell__Chart"], 5)
    assert_frame_equal(
        missingdata_df.assign(**df.loc[:, ["a", "Bell__Chart"]].fillna(5)), df
    )


@pytest.mark.functions
@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("mean", set([1, 2, 1.5])),
        ("average", set([1, 2, 1.5])),
        ("median", set([1, 2, 1.5])),
        ("mode", set([1, 2])),
        ("min", set([1, 2])),
        ("minimum", set([1, 2])),
        ("max", set([1, 2])),
        ("maximum", set([1, 2])),
    ],
)
def test_impute_statistical(missingdata_df, statistic, expected):
    """Check if imputing via statistic_column_name works correctly."""
    df = missingdata_df.impute("a", statistic_column_name=statistic)
    assert set(df["a"]) == expected


@pytest.mark.functions
def test_impute_error_with_invalid_inputs(missingdata_df):
    """Check errors are properly raised with invalid inputs."""
    with pytest.raises(
        ValueError,
        match="Only one of `value` or "
        "`statistic_column_name` "
        "should be provided.",
    ):
        missingdata_df.impute(
            "a",
            value=0,
            statistic_column_name="mean",
        )

    with pytest.raises(
        KeyError, match="`statistic_column_name` must be one of.+"
    ):
        missingdata_df.impute("a", statistic_column_name="foobar")

    with pytest.raises(
        ValueError, match="Kindly specify a value or a statistic_column_name"
    ):
        missingdata_df.impute("a")
