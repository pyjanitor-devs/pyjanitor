import pandas_flavor as pf
import pandas as pd
import datetime as dt


@pf.register_dataframe_method
def truncate_datetime_dataframe(
    df: pd.DataFrame, datepart: str
) -> pd.DataFrame:
    """
    Truncate times down to a user-specified precision of
    year, month, day, hour, minute, or second.

    Call on datetime object to truncate it.
    Calling on existing df will not alter the contents
    of said df.

    Note: Truncating down to a Month or Day will yields 0s,
    as there is no 0 month or 0 day in most datetime systems.

    :param df: The dataframe on which to truncate datetime.
    :param datepart: Truncation precision, YEAR, MONTH, DAY,
        HOUR, MINUTE, SECOND. (String is automagically
        capitalized)

    :returns: a truncated datetime object to
        the precision specified by datepart.
    """
    for i in df.columns:
        for j in df.index:
            try:
                df[i][j] = _truncate_datetime(datepart, df[i][j])
            except KeyError:
                pass
            except TypeError:
                pass
            except AttributeError:
                pass

    return df


def _truncate_datetime(datepart: str, timestamp: dt.datetime):
    """
    :param datepart: Truncation precision, YEAR, MONTH, DAY,
        HOUR, MINUTE, SECOND. (String is automagically
        capitalized)
    :param timestamp: expecting a datetime from python datetime class (dt)
    """
    recurrence = [0, 1, 1, 0, 0, 0]  # [YEAR, MONTH, DAY, HOUR, MINUTE, SECOND]
    datepart = datepart.upper()
    ENUM = {
        "YEAR": 0,
        "MONTH": 1,
        "DAY": 2,
        "HOUR": 3,
        "MINUTE:": 4,
        "SECOND": 5,
        0: timestamp.year,
        1: timestamp.month,
        2: timestamp.day,
        3: timestamp.hour,
        4: timestamp.minute,
        5: timestamp.second,
    }
    try:
        ENUM[datepart]
    # Capture the error but replace it with explicit instructions.
    except KeyError:
        msg = (
            "Invalid truncation. Please enter any one of 'year', "
            "'month', 'day', 'hour', 'minute' or 'second'."
        )
        raise KeyError(msg)

    for i in range(ENUM.get(datepart) + 1):
        recurrence[i] = ENUM.get(i)

    return dt.datetime(
        recurrence[0],
        recurrence[1],
        recurrence[2],
        recurrence[3],
        recurrence[4],
        recurrence[5],
    )
