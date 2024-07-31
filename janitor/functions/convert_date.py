from typing import Hashable, Union

import pandas as pd
import pandas_flavor as pf
from pandas.errors import OutOfBoundsDatetime

from janitor.utils import deprecated_alias, refactored_function


@pf.register_dataframe_method
@deprecated_alias(column="column_names")
def convert_excel_date(
    df: pd.DataFrame, column_names: Union[Hashable, list]
) -> pd.DataFrame:
    """Convert Excel's serial date format into Python datetime format.

    This method does not mutate the original DataFrame.

    Implementation is based on
    [Stack Overflow](https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas).

    Examples:
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

    Args:
        df: A pandas DataFrame.
        column_names: A column name, or a list of column names.

    Returns:
        A pandas DataFrame with corrected dates.
    """  # noqa: E501

    if not isinstance(column_names, list):
        column_names = [column_names]
    # https://stackoverflow.com/a/65460255/7175713
    dictionary = {
        column_name: pd.to_datetime(
            df[column_name], unit="D", origin="1899-12-30"
        )
        for column_name in column_names
    }

    return df.assign(**dictionary)


@pf.register_dataframe_method
@deprecated_alias(column="column_names")
def convert_matlab_date(
    df: pd.DataFrame, column_names: Union[Hashable, list]
) -> pd.DataFrame:
    """Convert Matlab's serial date number into Python datetime format.

    Implementation is based on
    [Stack Overflow](https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python).

    This method does not mutate the original DataFrame.

    Examples:
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
        0 2018-03-06 00:00:00.000000000
        1 2018-03-05 19:34:50.563199671
        2 2018-03-05 11:57:50.399998876
        3 2018-03-05 00:00:00.000000000

    Args:
        df: A pandas DataFrame.
        column_names: A column name, or a list of column names.

    Returns:
        A pandas DataFrame with corrected dates.
    """  # noqa: E501
    # https://stackoverflow.com/a/49135037/7175713
    if not isinstance(column_names, list):
        column_names = [column_names]
    dictionary = {
        column_name: pd.to_datetime(df[column_name] - 719529, unit="D")
        for column_name in column_names
    }

    return df.assign(**dictionary)


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.to_datetime` instead."
    )
)
@deprecated_alias(column="column_name")
def convert_unix_date(df: pd.DataFrame, column_name: Hashable) -> pd.DataFrame:
    """Convert unix epoch time into Python datetime format.

    Note that this ignores local tz and convert all timestamps to naive
    datetime based on UTC!

    This method mutates the original DataFrame.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.to_datetime` instead.

    Examples:
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

    Args:
        df: A pandas DataFrame.
        column_name: A column name.

    Returns:
        A pandas DataFrame with corrected dates.
    """

    try:
        df[column_name] = pd.to_datetime(df[column_name], unit="s")
    except OutOfBoundsDatetime:  # Indicates time is in milliseconds.
        df[column_name] = pd.to_datetime(df[column_name], unit="ms")
    return df
