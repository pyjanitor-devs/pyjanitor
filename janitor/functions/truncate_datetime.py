"""Implementation of the `truncate_datetime` family of functions."""
import datetime as dt
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_method
def truncate_datetime_dataframe(
    df: pd.DataFrame,
    datepart: str,
) -> pd.DataFrame:
    """
    Truncate times down to a user-specified precision of
    year, month, day, hour, minute, or second.

    Call on datetime object to truncate it.
    Calling on existing df will not alter the contents
    of said df.

    This method does not mutate the original DataFrame.

    Examples:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "foo": ["xxxx", "yyyy", "zzzz"],
        ...     "dt": pd.date_range("2020-03-11", periods=3),
        ... })
        >>> df
            foo         dt
        0  xxxx 2020-03-11
        1  yyyy 2020-03-12
        2  zzzz 2020-03-13
        >>> df.truncate_datetime_dataframe("month")
            foo         dt
        0  xxxx 2020-03-01
        1  yyyy 2020-03-01
        2  zzzz 2020-03-01

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

    df = df.copy()

    for col in df.columns:
        for row in df.index:
            try:
                df.loc[row, col] = _truncate_datetime(
                    datepart, df.loc[row, col]
                )
            except AttributeError:
                pass

    return df


def _truncate_datetime(datepart: str, timestamp: dt.datetime) -> dt.datetime:
    """Truncate a given timestamp to the given datepart.

    Data checks are assumed to have already been done; No further checks will
    be performed within this internal function.

    :param datepart: Truncation precision, YEAR, MONTH, DAY,
        HOUR, MINUTE, SECOND.
    :param timestamp: Expecting a datetime from python `datetime` class (dt).
    :raises AttributeError: If the input timestamp is not a datetime-like
        object.
    :returns: A truncated datetime object to the precision specified by
        datepart.
    """
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
