from random import randint

import pandas as pd
import pytest

from janitor.timeseries import _get_missing_timestamps, fill_missing_timestamps


# Random data for testing
@pytest.fixture
def timeseries_dataframe() -> pd.DataFrame:
    """
    Returns a time series dataframe
    """
    ts_index = pd.date_range("1/1/2019", periods=1000, freq="1H")
    v1 = [randint(1, 2000) for i in range(1000)]
    test_df = pd.DataFrame({"v1": v1}, index=ts_index)
    return test_df


@pytest.mark.timeseries
def test_fill_missing_timestamps(timeseries_dataframe):
    """Test that filling missing timestamps works as expected."""
    # Remove random row from the data frame
    random_number = randint(1, len(timeseries_dataframe))
    df1 = timeseries_dataframe.drop(timeseries_dataframe.index[random_number])

    # Fill missing timestamps
    # fix for GH#1184 is to use the start and end from
    # timeseries_dataframe
    # imagine that the last row of df1 is removed, or the first entry
    # the length check in the assert line will fail
    result = fill_missing_timestamps(
        df1,
        frequency="1H",
        first_time_stamp=timeseries_dataframe.index.min(),
        last_time_stamp=timeseries_dataframe.index.max(),
    )

    # Testing if the missing timestamp has been filled
    assert len(result) == len(timeseries_dataframe)

    # Testing if indices are exactly the same after filling
    original_index = timeseries_dataframe.index
    new_index = result.index
    delta = original_index.difference(new_index)

    assert delta.empty is True


@pytest.mark.timeseries
def test__get_missing_timestamps(timeseries_dataframe):
    """Test utility function for identifying the missing timestamps."""
    from random import sample

    timestamps_to_drop = sample(timeseries_dataframe.index.tolist(), 3)
    df = timeseries_dataframe.drop(index=timestamps_to_drop)
    missing_timestamps = _get_missing_timestamps(df, "1H")
    assert set(missing_timestamps.index) == set(timestamps_to_drop)
