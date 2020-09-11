"""
Unit tests for `.flag_jumps()`.
"""

import pandas as pd
import pytest

from janitor.timeseries import _flag_jumps_single_col, flag_jumps
from janitor.errors import JanitorError


@pytest.fixture
def timeseries_dataframe() -> pd.DataFrame:
    """Returns a time series dataframe."""
    ts_index = pd.date_range("1/1/2019", periods=10, freq="1H")
    c1 = [*range(10)]
    c2 = [*range(100, 110)]
    c3 = c1[::-1]
    c4 = c2[::-1]
    c5 = [-0.5, -0.25, 0, 0.25, 0.5, 0.25, 0, -0.25, -0.5, -3.5]
    test_df = pd.DataFrame(
        {"col1": c1, "col2": c2, "col3": c3, "col4": c4, "col5": c5},
        index=ts_index,
    )
    return test_df


@pytest.mark.timeseries
def test__flag_jumps_single_col_raises_error_for_bad_scale_type(
    timeseries_dataframe
):
    """Test that invalid scale argument raises a JanitorError."""
    # Setup
    df = timeseries_dataframe
    expected_error_msg = "Unrecognized scale='bad_scale'. Must be one of: ['absolute', 'percentage']"

    # Exercise
    with pytest.raises(JanitorError) as error_info:
        _flag_jumps_single_col(
            df, col="col1", scale="bad_scale", direction="any", threshold=1
        )

    # Verify
    assert str(error_info.value) == expected_error_msg

    # Cleanup - none necessary


@pytest.mark.timeseries
def test__flag_jumps_single_col_raises_error_for_bad_direction_type(
    timeseries_dataframe
):
    """Test that invalid direction argument raises a JanitorError."""
    # Setup
    df = timeseries_dataframe
    expected_error_msg = "Unrecognized direction='bad_direction'. Must be one of: ['increasing', 'decreasing', 'any']"

    # Exercise
    with pytest.raises(JanitorError) as error_info:
        _flag_jumps_single_col(
            df,
            col="col1",
            scale="absolute",
            direction="bad_direction",
            threshold=1,
        )

    # Verify
    assert str(error_info.value) == expected_error_msg

    # Cleanup - none necessary


@pytest.mark.timeseries
def test__flag_jumps_single_col_absolute_scale():
    """Test utility function for flagging jumps with absolute scale."""
    # Setup
    df = timeserties_dataframe

    # Exercise
    # col1_result = _flag_jumps_single_col(df, scale='absolute', direction='increasing', threshold=)
    # col2_result = _flag_jumps_single_col(df, scale='absolute', direction='increasing', threshold=)
    # col3_result = _flag_jumps_single_col(df, scale='absolute', direction='increasing', threshold=)
    # col4_result = _flag_jumps_single_col(df, scale='absolute', direction='increasing', threshold=)
    # col5_result = _flag_jumps_single_col(df, scale='absolute', direction='increasing', threshold=)

    # Verify

    # Cleanup - none necessary
