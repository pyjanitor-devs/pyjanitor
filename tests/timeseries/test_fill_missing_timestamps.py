import pytest
import pandas as pd
from janitor.timeseries import fill_missing_timestamps


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
    assert fill_missing_timestamps(
        pd.DataFrame([]), '1B', 'DateTime', pd.Timestamp(2020, 1, 1), pd.Timestamp(2020, 2, 1)
    )
