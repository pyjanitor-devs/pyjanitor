import pytest
import pandas as pd
from random import randint
from janitor.timeseries import sort_timestamps_monotonically
from janitor.functions import shuffle


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
def test_sort_timestamps_monotonically(timeseries_dataframe):
    """Test sort_timestamps_monotonically for ascending order"""
    df = shuffle(timeseries_dataframe, reset_index=False)
    df1 = sort_timestamps_monotonically(df)
    pd.testing.assert_frame_equal(df1, timeseries_dataframe)


@pytest.mark.timeseries
def test_sort_timestamps_monotonically_decreasing(timeseries_dataframe):
    """Test sort_timestamps_monotonically for descending order"""
    df2 = timeseries_dataframe.sort_index(ascending=False)
    df3 = sort_timestamps_monotonically(df2, "decreasing")
    pd.testing.assert_frame_equal(df3, df2)


@pytest.mark.timeseries
def test_sort_timestamps_monotonically_strict(timeseries_dataframe):
    """Test sort_timestamps_monotonically for index duplication handling"""
    df = shuffle(timeseries_dataframe, reset_index=False)
    random_number = randint(1, len(timeseries_dataframe))
    df4 = df.append(df.loc[df.index[random_number], :])
    df5 = sort_timestamps_monotonically(df4, "increasing", True)
    pd.testing.assert_frame_equal(df5, timeseries_dataframe)
