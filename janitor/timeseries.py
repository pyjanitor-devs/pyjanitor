"""
Time series-specific data testing and cleaning functions.
"""

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from collections import namedtuple
from janitor import check


def _flag_missing_timestamps(
    df: pd.DataFrame,
    frequency: str,
    column_name: str,
    first_time_stamp: pd.Timestamp,
    last_time_stamp: pd.Timestamp,
) -> namedtuple:
    """
    Test if timestamps are missing

    Utility function to test if input data frame
    is missing timestamps relative to expected timestamps.
    They are generated based on the first_time_stamp,
    last_time_stamp
    and frequency.

    :param df: data frame to test for missing timestamps
    :param frequency: frequency i.e. sampling frequency
        Acceptable frequency strings are available
        [Reference](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases/):
    :param column_name: name of the column which has time series if not the index
    :param first_time_stamp: timestamp the time_series is expected to start from
    :param last_time_stamp: timestamp the time_series is expected to end with
    :return: namedtuple with 3 attributes namely flag, raw_data and new_index
        1. flag - boolean set to True if there are missing timestamps,
            else set to False
        2. raw_data - input data frame as is without any modifications
        3. new_index - pd.DateTimeIndex that can be used to set the new index.
            Defaults to None.
            Assigned a value only when flag is set to True
    """
    # Declare a named tuple to hold results
    MissingTimeStampFlag = namedtuple(
        "MissingTimeStampFlag", ["flag", "raw_data", "new_index"]
    )
    result = {"flag": None, "raw_data": df.copy(deep=True), "new_index": None}

    # Generate expected timestamps
    expected_timestamps = pd.date_range(
        start=first_time_stamp, end=last_time_stamp, frequency=frequency
    )

    # Get actual timestamps
    if column_name:
        df = df.set_index(column_name)

    df = df.sort_index(inplace=True)
    actual_timestamps = df.index.array

    # Check if they are the same
    comparison_index = expected_timestamps.difference(actual_timestamps)
    if comparison_index.empty:
        result["flag"] = False
    result["flag"] = True
    result["new_index"] = expected_timestamps

    # Return the result as a Named Tuple
    return MissingTimeStampFlag._make(result)


@pf.register_dataframe_method
def fill_missing_timestamps(
    df: pd.DataFrame,
    frequency: str,
    column_name: str = None,
    first_time_stamp: pd.Timestamp = None,
    last_time_stamp: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Fills data frame with missing timestamps if missing

    Test the data frame for missing timestamps
    If timestamps are missing, Re-indexes the data frame
    If timestamps are not missing, returns original data frame

    :param df: data frame which needs to be tested for missing timestamps
    :param frequency: frequency i.e. sampling frequency of the data.
        Acceptable frequency strings are available
        [Reference](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases/):
    :param column_name: name of the column which has time series if not the index.
        Defaults to None.
        By default the index is used for checking for the timestamps,
        unless a value other than None is assigned to column_name
    :param first_time_stamp: timestamp at which the time_series is expected to start from.
        Defaults to None.
        If no input is provided assumes the minimum value in time_series
    :param last_time_stamp: timestamp at which the time_series is expected to end with.
        Defaults to None.
        If no input is provided, assumes the maximum value in time_series
    :return: reindexed data frame if it turns out to have missing timestamps,
        else returns the input data frame as is
    """
    # Check all the inputs are the correct data type
    check("df", df, [pd.DataFrame])
    check("frequency", frequency, [str])
    check("column_name", column_name, [str, None])
    check("first_time_stamp", first_time_stamp, [pd.Timestamp, None])
    check("last_time_stamp", last_time_stamp, [pd.Timestamp, None])

    # Assign inputs if not provided
    if column_name:
        test_new_index_data_type = is_datetime(df[column_name])
        if not test_new_index_data_type:
            raise TypeError(
                "\n column_name should refer to a column whose data type is datetime"
            )
    if not first_time_stamp:
        first_time_stamp = df.index.min()
    if not last_time_stamp:
        last_time_stamp = df.index.max()

    # Test if there are any timestamps missing
    timestamps_missing_flag = _flag_missing_timestamps(
        df, frequency, column_name, first_time_stamp, last_time_stamp
    )

    # Return result based on whether timestamps are missing or not
    if timestamps_missing_flag["flag"]:
        df = df.set_index(timestamps_missing_flag["new_index"])
        return df
    return timestamps_missing_flag["raw_data"]
