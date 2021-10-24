from typing import Set
import pandas_flavor as pf
import pandas as pd


@pf.register_series_method
def toset(series: pd.Series) -> Set:
    """Return a set of the values.

    These are each a scalar type, which is a Python scalar
    (for str, int, float) or a pandas scalar
    (for Timestamp/Timedelta/Interval/Period)

    Functional usage syntax:



        import pandas as pd
        import janitor as jn

        series = pd.Series(...)
        s = jn.functions.toset(series=series)

    Method chaining usage example:



        import pandas as pd
        import janitor

        series = pd.Series(...)
        s = series.toset()

    :param series: A pandas series.
    :returns: A set of values.
    """

    return set(series.tolist())
