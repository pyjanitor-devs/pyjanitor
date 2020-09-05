from random import randint

import pandas as pd
import pytest

import janitor  # noqa: F401
import janitor.timeseries  # noqa: F401


@pytest.fixture
def timeseries_dataframe() -> pd.DataFrame:
    """
    Returns a time series dataframe
    """
    ts_index = pd.date_range("1/1/2019", periods=1000, freq="1H")
    v1 = [randint(1, 2000) for i in range(1000)]
    test_df = pd.DataFrame({"v1": v1}, index=ts_index)
    return test_df


# NOTE: The tests possibly can be merged back together later
# if they are parametrized properly.
# NOTE: These tests use `df.equals(other_df)`,
# because the desired `pd.assert_frame_equal(df, other_df)`
# constantly failed on the CI systems.
# It's a task for later to fix.


@pytest.mark.timeseries
def test_sort_timestamps_monotonically(timeseries_dataframe):
    """Test sort_timestamps_monotonically for ascending order"""
    df = timeseries_dataframe.shuffle(
        reset_index=False
    ).sort_timestamps_monotonically()
    assert df.equals(timeseries_dataframe)


@pytest.mark.timeseries
def test_sort_timestamps_monotonically_decreasing(timeseries_dataframe):
    """Test sort_timestamps_monotonically for descending order"""
    df2 = timeseries_dataframe.sort_index(ascending=False)
    df3 = df2.sort_timestamps_monotonically("decreasing")
    assert df3.equals(df2)


@pytest.mark.timeseries
def test_sort_timestamps_monotonically_strict(timeseries_dataframe):
    """Test sort_timestamps_monotonically for index duplication handling"""
    df = timeseries_dataframe.shuffle(reset_index=False)
    random_number = randint(1, len(timeseries_dataframe))
    df = df.append(
        df.loc[df.index[random_number], :]
    ).sort_timestamps_monotonically(direction="increasing", strict=True)
    assert df.equals(timeseries_dataframe)
