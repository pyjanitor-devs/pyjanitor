from functools import reduce
from typing import Dict, Hashable, List, Optional
import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias
import datetime as dt
import numpy as np
import warnings


@pf.register_dataframe_method
@deprecated_alias(column="column_name", start="start_date", end="end_date")
def filter_date(
    df: pd.DataFrame,
    column_name: Hashable,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
    years: Optional[List] = None,
    months: Optional[List] = None,
    days: Optional[List] = None,
    column_date_options: Optional[Dict] = None,
    format: Optional[str] = None,  # skipcq: PYL-W0622
) -> pd.DataFrame:
    """
    Filter a date-based column based on certain criteria.

    This method does not mutate the original DataFrame.

    Dates may be finicky and this function builds on top of the *magic* from
    the pandas `to_datetime` function that is able to parse dates well.

    Additional options to parse the date type of your column may be found at
    the official pandas [documentation][datetime]

    [datetime]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

    !!!note

        This method will cast your column to a Timestamp!

    !!!note

        This only affects the format of the `start_date` and `end_date`
        parameters. If there's an issue with the format of the DataFrame being
        parsed, you would pass `{'format': your_format}` to `column_date_options`.

    :param df: A pandas dataframe.
    :param column_name: The column which to apply the fraction transformation.
    :param start_date: The beginning date to use to filter the DataFrame.
    :param end_date: The end date to use to filter the DataFrame.
    :param years: The years to use to filter the DataFrame.
    :param months: The months to use to filter the DataFrame.
    :param days: The days to use to filter the DataFrame.
    :param column_date_options: 'Special options to use when parsing the date
        column in the original DataFrame. The options may be found at the
        official Pandas documentation.'
    :param format: 'If you're using a format for `start_date` or `end_date`
        that is not recognized natively by pandas' `to_datetime` function, you
        may supply the format yourself. Python date and time formats may be
        found at [link](http://strftime.org/).
    :returns: A filtered pandas DataFrame.
    """  # noqa: E501

    # TODO: need to convert this to notebook.
    #     :Setup:
    # ```python

    #     import pandas as pd
    #     import janitor

    #     date_list = [
    #         [1, "01/28/19"], [2, "01/29/19"], [3, "01/30/19"],
    #         [4, "01/31/19"], [5, "02/01/19"], [6, "02/02/19"],
    #         [7, "02/03/19"], [8, "02/04/19"], [9, "02/05/19"],
    #         [10, "02/06/19"], [11, "02/07/20"], [12, "02/08/20"],
    #         [13, "02/09/20"], [14, "02/10/20"], [15, "02/11/20"],
    #         [16, "02/12/20"], [17, "02/07/20"], [18, "02/08/20"],
    #         [19, "02/09/20"], [20, "02/10/20"], [21, "02/11/20"],
    #         [22, "02/12/20"], [23, "03/08/20"], [24, "03/09/20"],
    #         [25, "03/10/20"], [26, "03/11/20"], [27, "03/12/20"]]

    #     example_dataframe = pd.DataFrame(date_list,
    #                                      columns = ['AMOUNT', 'DATE'])

    # :Example 1: Filter dataframe between two dates

    # ```python

    #     start_date = "01/29/19"
    #     end_date = "01/30/19"

    #     example_dataframe.filter_date(
    #         'DATE', start_date=start_date, end_date=end_date
    #     )

    # :Output:

    # ```python

    #        AMOUNT       DATE
    #     1       2 2019-01-29
    #     2       3 2019-01-30

    # :Example 2: Using a different date format for filtering

    # ```python

    #     end_date = "01$$$30$$$19"
    #     format = "%m$$$%d$$$%y"

    #     example_dataframe.filter_date(
    #         'DATE', end_date=end_date, format=format
    #     )

    # :Output:

    # ```python

    #        AMOUNT       DATE
    #     0       1 2019-01-28
    #     1       2 2019-01-29
    #     2       3 2019-01-30

    # :Example 3: Filtering by year

    # ```python

    #     years = [2019]

    #     example_dataframe.filter_date('DATE', years=years)

    # :Output:

    # ```python

    #        AMOUNT       DATE
    #     0       1 2019-01-28
    #     1       2 2019-01-29
    #     2       3 2019-01-30
    #     3       4 2019-01-31
    #     4       5 2019-02-01
    #     5       6 2019-02-02
    #     6       7 2019-02-03
    #     7       8 2019-02-04
    #     8       9 2019-02-05
    #     9      10 2019-02-06

    # :Example 4: Filtering by year and month

    # ```python

    #     years = [2020]
    #     months = [3]

    #     example_dataframe.filter_date('DATE', years=years, months=months)

    # :Output:

    # ```python

    #         AMOUNT       DATE
    #     22      23 2020-03-08
    #     23      24 2020-03-09
    #     24      25 2020-03-10
    #     25      26 2020-03-11
    #     26      27 2020-03-12

    # :Example 5: Filtering by year and day

    # ```python

    #     years = [2020]
    #     days = range(10,12)

    #     example_dataframe.filter_date('DATE', years=years, days=days)

    # :Output:

    # ```python

    #         AMOUNT       DATE
    #     13      14 2020-02-10
    #     14      15 2020-02-11
    #     19      20 2020-02-10
    #     20      21 2020-02-11
    #     24      25 2020-03-10
    #     25      26 2020-03-11

    def _date_filter_conditions(conditions):
        """Taken from: https://stackoverflow.com/a/13616382."""
        return reduce(np.logical_and, conditions)

    if column_date_options:
        df.loc[:, column_name] = pd.to_datetime(
            df.loc[:, column_name], **column_date_options
        )
    else:
        df.loc[:, column_name] = pd.to_datetime(df.loc[:, column_name])

    _filter_list = []

    if start_date:
        start_date = pd.to_datetime(start_date, format=format)
        _filter_list.append(df.loc[:, column_name] >= start_date)

    if end_date:
        end_date = pd.to_datetime(end_date, format=format)
        _filter_list.append(df.loc[:, column_name] <= end_date)

    if years:
        _filter_list.append(df.loc[:, column_name].dt.year.isin(years))

    if months:
        _filter_list.append(df.loc[:, column_name].dt.month.isin(months))

    if days:
        _filter_list.append(df.loc[:, column_name].dt.day.isin(days))

    if start_date and end_date and start_date > end_date:
        warnings.warn(
            f"Your start date of {start_date} is after your end date of "
            f"{end_date}. Is this intended?"
        )

    return df.loc[_date_filter_conditions(_filter_list), :]
