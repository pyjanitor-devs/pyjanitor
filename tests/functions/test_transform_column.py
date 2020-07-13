"""
Tests referring to the method transform_column of the module functions.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal


@pytest.mark.functions
@pytest.mark.parametrize("elementwise", [True, False])
def test_transform_column(dataframe, elementwise):
    """
    Test replacing data of the original column.

    The function that is used for testing here
    must be able to operate elementwise
    and on a single pandas Series.
    We use np.log10 as an example
    """

    df = dataframe.transform_column("a", np.log10, elementwise=elementwise)
    expected = pd.Series(np.log10([1, 2, 3] * 3))
    expected.name = "a"
    assert_series_equal(df["a"], expected)


@pytest.mark.functions
def test_transform_column_with_dest(dataframe):
    """Test creating a new destination column."""

    expected_df = dataframe.assign(a_log10=np.log10(dataframe["a"]))

    df = dataframe.copy().transform_column(
        "a", np.log10, dest_column_name="a_log10"
    )

    assert_frame_equal(df, expected_df)


@pytest.mark.functions
def test_transform_column_no_mutation(dataframe):
    """Test checking that transform_column doesn't mutate the dataframe."""
    df = dataframe.transform_column("a", np.log10)

    with pytest.raises(AssertionError):
        assert_frame_equal(dataframe, df)
