from pandas.errors import OutOfBoundsDatetime
import datetime as dt
from typing import Hashable
import pandas_flavor as pf
import pandas as pd
from pandas.api.types import is_numeric_dtype

from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_excel_date(
    df: pd.DataFrame, column_name: Hashable
) -> pd.DataFrame:
    """
    Convert Excel's serial date format into Python datetime format.

    This method mutates the original DataFrame.

    Implementation is also from
    [Stack Overflow](https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas)

    Method chaining syntax:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"date": [39690, 39690, 37118]})
        >>> df
            date
        0  39690
        1  39690
        2  37118
        >>> df.convert_excel_date('date')
                date
        0 2008-08-30
        1 2008-08-30
        2 2001-08-15

    :param df: A pandas DataFrame.
    :param column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    :raises ValueError: if there are non numeric values in the column.
    """  # noqa: E501

    if not is_numeric_dtype(df[column_name]):
        raise ValueError(
            "There are non-numeric values in the column. \
    All values must be numeric"
        )

    df[column_name] = pd.TimedeltaIndex(
        df[column_name], unit="d"
    ) + dt.datetime(
        1899, 12, 30
    )  # noqa: W503
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_matlab_date(
    df: pd.DataFrame, column_name: Hashable
) -> pd.DataFrame:
    """
    Convert Matlab's serial date number into Python datetime format.

    Implementation is also from
    [Stack Overflow](https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python)

    This method mutates the original DataFrame.

    Method chaining syntax:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"date": [737125.0, 737124.815863, 737124.4985, 737124]})
        >>> df
                    date
        0  737125.000000
        1  737124.815863
        2  737124.498500
        3  737124.000000
        >>> df.convert_matlab_date('date')
                                date
        0 2018-03-06 00:00:00.000000
        1 2018-03-05 19:34:50.563200
        2 2018-03-05 11:57:50.399999
        3 2018-03-05 00:00:00.000000

    :param df: A pandas DataFrame.
    :param column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """  # noqa: E501
    days = pd.Series([dt.timedelta(v % 1) for v in df[column_name]])
    df[column_name] = (
        df[column_name].astype(int).apply(dt.datetime.fromordinal)
        + days
        - dt.timedelta(days=366)
    )
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_unix_date(df: pd.DataFrame, column_name: Hashable) -> pd.DataFrame:
    """
    Convert unix epoch time into Python datetime format.

    Note that this ignores local tz and convert all timestamps to naive
    datetime based on UTC!

    This method mutates the original DataFrame.

    Method chaining syntax:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"date": [1651510462, 53394822, 1126233195]})
        >>> df
                 date
        0  1651510462
        1    53394822
        2  1126233195
        >>> df.convert_unix_date('date')
                         date
        0 2022-05-02 16:54:22
        1 1971-09-10 23:53:42
        2 2005-09-09 02:33:15

    :param df: A pandas DataFrame.
    :param column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """

    try:
        df[column_name] = pd.to_datetime(df[column_name], unit="s")
    except OutOfBoundsDatetime:  # Indicates time is in milliseconds.
        df[column_name] = pd.to_datetime(df[column_name], unit="ms")
    return df
