import pytest
import pandas as pd
from random import randint
from janitor.timeseries import sort_timestamps_monotonically


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
    """Test that sort_timestamps_monotonically works as expected."""
    from sklearn.utils import shuffle

    # Increasing direction
    df = shuffle(timeseries_dataframe)
    df1 = sort_timestamps_monotonically(df)
    assert df1.equals(timeseries_dataframe)

    # Decreasing direction
    df2 = timeseries_dataframe.sort_index(ascending=False)
    df3 = sort_timestamps_monotonically(df2, "decreasing")
    assert df3.equals(df2)

    # Test for strictness
    random_number = randint(1, len(timeseries_dataframe))
    df4 = df.append(df.loc[df.index[random_number], :])
    df5 = sort_timestamps_monotonically(df4, "increasing", True)
    assert df5.equals(timeseries_dataframe)
