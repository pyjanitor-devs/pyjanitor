"""Implementation of the `truncate_datetime` family of functions."""
import datetime as dt

import pandas_flavor as pf
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


@pf.register_dataframe_method
def truncate_datetime_dataframe(
    df: pd.DataFrame,
    datepart: str,
) -> pd.DataFrame:
    """Truncate times down to a user-specified precision of
    year, month, day, hour, minute, or second.

    This method does not mutate the original DataFrame.

    Examples:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "foo": ["xxxx", "yyyy", "zzzz"],
        ...     "dt": pd.date_range("2020-03-11", periods=3, freq="15H"),
        ... })
        >>> df
            foo                  dt
        0  xxxx 2020-03-11 00:00:00
        1  yyyy 2020-03-11 15:00:00
        2  zzzz 2020-03-12 06:00:00
        >>> df.truncate_datetime_dataframe("day")
            foo         dt
        0  xxxx 2020-03-11
        1  yyyy 2020-03-11
        2  zzzz 2020-03-12

    :param df: The pandas DataFrame on which to truncate datetime.
    :param datepart: Truncation precision, YEAR, MONTH, DAY,
        HOUR, MINUTE, SECOND. (String is automagically
        capitalized)

    :raises ValueError: If an invalid `datepart` precision is passed in.
    :returns: A pandas DataFrame with all valid datetimes truncated down
        to the specified precision.
    """
    ACCEPTABLE_DATEPARTS = ("YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND")
    datepart = datepart.upper()
    if datepart not in ACCEPTABLE_DATEPARTS:
        raise ValueError(
            "Received an invalid `datepart` precision. "
            f"Please enter any one of {ACCEPTABLE_DATEPARTS}."
        )

    dt_cols = [
        column
        for column, coltype in df.dtypes.items()
        if is_datetime64_any_dtype(coltype)
    ]
    if not dt_cols:
        # avoid copying df if no-op is expected
        return df

    df = df.copy()
    # NOTE: use **kwargs of `applymap` instead of lambda when we upgrade to
    #   pandas >= 1.3.0
    df[dt_cols] = df[dt_cols].applymap(
        lambda x: _truncate_datetime(x, datepart=datepart),
    )

    return df


def _truncate_datetime(timestamp: dt.datetime, datepart: str) -> dt.datetime:
    """Truncate a given timestamp to the given datepart.

    Truncation will only occur on valid timestamps (datetime-like objects).

    :param timestamp: Expecting a datetime from python `datetime` class (dt).
    :param datepart: Truncation precision, YEAR, MONTH, DAY,
        HOUR, MINUTE, SECOND.
    :returns: A truncated datetime object to the precision specified by
        datepart.
    """
    if pd.isna(timestamp):
        return timestamp

    recurrence = [0, 1, 1, 0, 0, 0]  # [YEAR, MONTH, DAY, HOUR, MINUTE, SECOND]
    ENUM = {
        "YEAR": 0,
        "MONTH": 1,
        "DAY": 2,
        "HOUR": 3,
        "MINUTE": 4,
        "SECOND": 5,
        0: timestamp.year,
        1: timestamp.month,
        2: timestamp.day,
        3: timestamp.hour,
        4: timestamp.minute,
        5: timestamp.second,
    }

    for i in range(ENUM[datepart] + 1):
        recurrence[i] = ENUM[i]

    return dt.datetime(*recurrence)
