import pytest
import pandas as pd
from random import randint
from janitor.timeseries import fill_missing_timestamps

# Random data for testing 
ts_index = pd.date_range('1/1/2019', periods=1000, freq='1H')
v1 = [randint(1, 2000) for i in range(1000)]
test_df = pd.DataFrame({'v1': v1}, index=ts_index)


@pytest.mark.parametrize(
    "df,frequency,column_name,first_time_stamp,last_time_stamp",
    [
        (4, '1T', None, None, None),
        (pd.DataFrame([]), 2, None, None, None),
        (pd.DataFrame([]), '1B', 1, None, None),
        (pd.DataFrame([]), '1B', 'DateTime', '2020-01-01', None),
        (pd.DataFrame([]), '1B', 'DateTime', pd.Timestamp(2020, 1, 1), '2020-02-01'),
    ]
)
def test_datatypes_check(df, frequency, column_name, first_time_stamp, last_time_stamp):
    with pytest.raises(TypeError):
        fill_missing_timestamps(df, frequency, column_name, first_time_stamp, last_time_stamp)


@pytest.mark.timeseries
def test_fill_missing_timestamps():
    # Remove random row from the data frame
    df1 = test_df.drop(test_df.sample())

    result = fill_missing_timestamps(
        df1,
        frequency='1H'
    )

    assert result == test_df




