"""
Tests referring to the method transform_column of the module functions.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal


@pytest.mark.functions
def test_transform_nonexisting_column(dataframe):
    """Checks an error is raised when the column being transformed
    is non-existent.
    """
    with pytest.raises(
        ValueError,
        match="_foobar_ not present",
    ):
        dataframe.transform_column("_foobar_", np.log10)


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
        "a",
        np.log10,
        dest_column_name="a_log10",
    )

    assert_frame_equal(df, expected_df)


@pytest.mark.functions
def test_transform_column_dest_column_already_present(dataframe):
    """Test behaviour when new destination column is provided and already
    exists."""
    # If dest_column_name already exists, throw an error
    with pytest.raises(
        ValueError,
        match="pyjanitor already present in dataframe",
    ):
        _ = dataframe.assign(pyjanitor=1).transform_column(
            "a",
            np.log10,
            dest_column_name="pyjanitor",
        )

    # unless dest_column_name is the same as column_name (the column being
    # transformed); assume user wants an in-place transformation in this
    # case.
    expected_df = dataframe.copy().assign(a=np.log10(dataframe["a"]))
    result_df = dataframe.transform_column(
        "a",
        np.log10,
        dest_column_name="a",
    )
    assert_frame_equal(result_df, expected_df)


@pytest.mark.functions
def test_transform_column_no_mutation(dataframe):
    """Test checking that transform_column doesn't mutate the dataframe."""
    df = dataframe.transform_column("a", np.log10)

    with pytest.raises(AssertionError):
        assert_frame_equal(dataframe, df)
