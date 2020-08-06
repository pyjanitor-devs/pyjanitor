import pytest
import pandas as pd
from random import randint
from janitor.timeseries import fill_missing_timestamps


# Random data for testing
@pytest.fixture
def my_dataframe() -> pd.DataFrame:
    """
    Returns a time series dataframe
    """
    ts_index = pd.date_range("1/1/2019", periods=1000, freq="1H")
    v1 = [randint(1, 2000) for i in range(1000)]
    test_df = pd.DataFrame({"v1": v1}, index=ts_index)
    return test_df


@pytest.mark.parametrize(
    "df,frequency,column_name,first_time_stamp,last_time_stamp",
    [
        (4, "1T", None, None, None),
        (pd.DataFrame([]), 2, None, None, None),
        (pd.DataFrame([]), "1B", 1, None, None),
        (pd.DataFrame([]), "1B", "DateTime", "2020-01-01", None),
        (
            pd.DataFrame([]),
            "1B",
            "DateTime",
            pd.Timestamp(2020, 1, 1),
            "2020-02-01",
        ),
    ],
)
def test_datatypes_check(
    df, frequency, column_name, first_time_stamp, last_time_stamp
):
    with pytest.raises(TypeError):
        fill_missing_timestamps(
            df, frequency, column_name, first_time_stamp, last_time_stamp
        )


@pytest.mark.timeseries
def test_fill_missing_timestamps(my_dataframe):
    # Remove random row from the data frame
    random_number = randint(1, len(my_dataframe))
    df1 = my_dataframe.drop(my_dataframe.index[random_number])

    # Fill missing timestamps
    result = fill_missing_timestamps(df1, frequency="1H")

    # Testing if the missing timestamp has been filled
    assert len(result) == len(my_dataframe)

    # Testing if indices are exactly the same after filling
    original_index = my_dataframe.index
    new_index = result.index
    delta = original_index.difference(new_index)

    assert delta.empty is True
