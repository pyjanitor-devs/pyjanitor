import datetime as dt
import warnings
from functools import reduce
from typing import Dict, Hashable, Iterable, List, Optional

import numpy as np
import pandas as pd
import pandas_flavor as pf
from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_string(
    df: pd.DataFrame,
    column_name: Hashable,
    search_string: str,
    complement: bool = False,
) -> pd.DataFrame:
    """
    Filter a string-based column according to whether it contains a substring.

    This is super sugary syntax that builds on top of
    `pandas.Series.str.contains`.

    Because this uses internally `pandas.Series.str.contains`, which allows a
    regex string to be passed into it, thus `search_string` can also be a regex
    pattern.

    This method does not mutate the original DataFrame.

    This function allows us to method chain filtering operations:

    ```python
        df = (pd.DataFrame(...)
              .filter_string('column', search_string='pattern', complement=False)
              ...)  # chain on more data preprocessing.
    ```

    This stands in contrast to the in-place syntax that is usually used:

    ```python
        df = pd.DataFrame(...)
        df = df[df['column'].str.contains('pattern')]]
    ```

    As can be seen here, the API design allows for a more seamless flow in
    expressing the filtering operations.

    Functional usage syntax:

    ```python
        df = filter_string(df,
                           column_name='column',
                           search_string='pattern',
                           complement=False)
    ```

    Method chaining syntax:

    ```python
        df = (pd.DataFrame(...)

              .filter_string(column_name='column',
                             search_string='pattern',
                             complement=False)
              ...)
    ```

    :param df: A pandas DataFrame.
    :param column_name: The column to filter. The column should contain strings.
    :param search_string: A regex pattern or a (sub-)string to search.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """  # noqa: E501
    criteria = df[column_name].str.contains(search_string)
    if complement:
        return df[~criteria]
    return df[criteria]


@pf.register_dataframe_method
def filter_on(
    df: pd.DataFrame, criteria: str, complement: bool = False
) -> pd.DataFrame:
    """
    Return a dataframe filtered on a particular criteria.

    This method does not mutate the original DataFrame.

    This is super-sugary syntax that wraps the pandas `.query()` API, enabling
    users to use strings to quickly specify filters for filtering their
    dataframe. The intent is that `filter_on` as a verb better matches the
    intent of a pandas user than the verb `query`.

    Let's say we wanted to filter students based on whether they failed an exam
    or not, which is defined as their score (in the "score" column) being less
    than 50.

    ```python
        df = (pd.DataFrame(...)
              .filter_on('score < 50', complement=False)
              ...)  # chain on more data preprocessing.
    ```

    This stands in contrast to the in-place syntax that is usually used:

    ```python
        df = pd.DataFrame(...)
        df = df[df['score'] < 3]
    ```

    As with the `filter_string` function, a more seamless flow can be expressed
    in the code.

    Functional usage syntax:

    ```python
        df = filter_on(df,
                       'score < 50',
                       complement=False)
    ```

    Method chaining syntax:
              .filter_on('score < 50', complement=False))
    ```

    Credit to Brant Peterson for the name.

    :param df: A pandas DataFrame.
    :param criteria: A filtering criteria that returns an array or Series of
        booleans, on which pandas can filter on.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """
    if complement:
        return df.query("not " + criteria)
    return df.query(criteria)


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

    :param df: The dataframe to filter on.
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


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_column_isin(
    df: pd.DataFrame,
    column_name: Hashable,
    iterable: Iterable,
    complement: bool = False,
) -> pd.DataFrame:
    """
    Filter a dataframe for values in a column that exist in another iterable.

    This method does not mutate the original DataFrame.

    Assumes exact matching; fuzzy matching not implemented.

    The below example syntax will filter the DataFrame such that we only get
    rows for which the `names` are exactly `James` and `John`.

    ```python
        df = (
            pd.DataFrame(...)
            .clean_names()
            .filter_column_isin(column_name="names", iterable=["James", "John"]
            )
        )
    ```

    This is the method chaining alternative to:

    ```python
        df = df[df['names'].isin(['James', 'John'])]
    ```

    If `complement` is `True`, then we will only get rows for which the names
    are not `James` or `John`.

    :param df: A pandas DataFrame
    :param column_name: The column on which to filter.
    :param iterable: An iterable. Could be a list, tuple, another pandas
        Series.
    :param complement: Whether to return the complement of the selection or
        not.
    :returns: A filtered pandas DataFrame.
    :raises ValueError: if `iterable` does not have a length of `1`
        or greater.
    """
    if len(iterable) == 0:
        raise ValueError(
            "`iterable` kwarg must be given an iterable of length 1 or greater"
        )
    criteria = df[column_name].isin(iterable)

    if complement:
        return df[~criteria]
    return df[criteria]
