import pandas as pd
import pytest
from random import randint
from janitor.timeseries import range_flag


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
def test_range_flag(timeseries_dataframe):
    """ Test that range flag works as expected """
    my_range_flag = range_flag(
        df=timeseries_dataframe,
        bound=[100, 750],
        inclusive=False
    )

    # Subset using the booleans created
    v1_subset_index = my_range_flag[
        my_range_flag['v1_range_flag'] == 1
    ].index
    v1_subset = timeseries_dataframe.loc[v1_subset_index, 'v1']

    # Test if bounds have been honored
    v1_subset_minimum = v1_subset.min()
    v1_subset_maximum = v1_subset.max()
    assert(100 < v1_subset_minimum)
    assert(750 > v1_subset_maximum)


@pytest.mark.timeseries
def test_range_flag_inclusive(timeseries_dataframe):
    """ Test range flag with inclusive argument """
    # Set some values equal to boundary values
    random_low = randint(1, len(timeseries_dataframe))
    random_high = randint(1, len(timeseries_dataframe))
    timeseries_dataframe.loc[timeseries_dataframe.index[random_low]] = 100
    timeseries_dataframe.loc[timeseries_dataframe.index[random_high]] = 750

    my_range_flag = range_flag(
        df=timeseries_dataframe,
        bound=[100, 750],
        inclusive=True
    )

    # Subset using the booleans created
    v1_subset_index = my_range_flag[
        my_range_flag['v1_range_flag'] == 1
    ].index
    v1_subset = timeseries_dataframe.loc[v1_subset_index, 'v1']

    # Test if bounds have been honored
    v1_subset_minimum = v1_subset.min()
    v1_subset_maximum = v1_subset.max()
    assert(100 == v1_subset_minimum)
    assert(750 == v1_subset_maximum)


if __name__ == '__main__':
    pytest.main()
