"""Implementation of the `truncate_datetime` family of functions."""

import numpy as np
import pandas as pd
import pandas_flavor as pf
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

    Args:
        df: The pandas DataFrame on which to truncate datetime.
        datepart: Truncation precision, YEAR, MONTH, DAY,
            HOUR, MINUTE, SECOND. (String is automagically
            capitalized)

    Raises:
        ValueError: If an invalid `datepart` precision is passed in.

    Returns:
        A pandas DataFrame with all valid datetimes truncated down
            to the specified precision.
    """
    # idea from Stack Overflow
    # https://stackoverflow.com/a/28783971/7175713
    # https://numpy.org/doc/stable/reference/arrays.datetime.html
    ACCEPTABLE_DATEPARTS = {
        "YEAR": "datetime64[Y]",
        "MONTH": "datetime64[M]",
        "DAY": "datetime64[D]",
        "HOUR": "datetime64[h]",
        "MINUTE": "datetime64[m]",
        "SECOND": "datetime64[s]",
    }
    datepart = datepart.upper()
    if datepart not in ACCEPTABLE_DATEPARTS:
        raise ValueError(
            "Received an invalid `datepart` precision. "
            f"Please enter any one of {ACCEPTABLE_DATEPARTS}."
        )

    dictionary = {}

    for label, series in df.items():
        if is_datetime64_any_dtype(series):
            dtype = ACCEPTABLE_DATEPARTS[datepart]
            # TODO: add branch for pyarrow arrays
            series = np.array(series._values, dtype=dtype)
        dictionary[label] = series

    return pd.DataFrame(dictionary)
