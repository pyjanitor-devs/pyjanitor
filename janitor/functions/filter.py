import datetime as dt
import warnings
from functools import reduce
from typing import Any, Dict, Hashable, Iterable, List, Optional

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import (
    deprecated_alias,
    find_stack_level,
    refactored_function,
)

warnings.simplefilter("always", DeprecationWarning)


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_string(
    df: pd.DataFrame,
    column_name: Hashable,
    search_string: str,
    complement: bool = False,
    case: bool = True,
    flags: int = 0,
    na: Any = None,
    regex: bool = True,
) -> pd.DataFrame:
    """Filter a string-based column according to whether it contains a substring.

    This is super sugary syntax that builds on top of `pandas.Series.str.contains`.
    It is meant to be the method-chaining equivalent of the following:

    ```python
    df = df[df[column_name].str.contains(search_string)]]
    ```

    This method does not mutate the original DataFrame.

    Examples:
        Retain rows whose column values contain a particular substring.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": range(3, 6), "b": ["bear", "peeL", "sail"]})
        >>> df
           a     b
        0  3  bear
        1  4  peeL
        2  5  sail
        >>> df.filter_string(column_name="b", search_string="ee")
           a     b
        1  4  peeL
        >>> df.filter_string(column_name="b", search_string="L", case=False)
           a     b
        1  4  peeL
        2  5  sail

        Filter names does not contain `'.'` (disable regex mode).

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.Series(["JoseChen", "Brian.Salvi"], name="Name").to_frame()
        >>> df
                  Name
        0     JoseChen
        1  Brian.Salvi
        >>> df.filter_string(column_name="Name", search_string=".", regex=False, complement=True)
               Name
        0  JoseChen

    Args:
        df: A pandas DataFrame.
        column_name: The column to filter. The column should contain strings.
        search_string: A regex pattern or a (sub-)string to search.
        complement: Whether to return the complement of the filter or not. If
            set to True, then the rows for which the string search fails are retained
            instead.
        case: If True, case sensitive.
        flags: Flags to pass through to the re module, e.g. re.IGNORECASE.
        na: Fill value for missing values. The default depends on dtype of
            the array. For object-dtype, `numpy.nan` is used. For `StringDtype`,
            `pandas.NA` is used.
        regex: If True, assumes `search_string` is a regular expression. If False,
            treats the `search_string` as a literal string.

    Returns:
        A filtered pandas DataFrame.
    """  # noqa: E501

    criteria = df[column_name].str.contains(
        pat=search_string,
        case=case,
        flags=flags,
        na=na,
        regex=regex,
    )

    if complement:
        return df[~criteria]

    return df[criteria]


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `pd.DataFrame.query` instead."
    )
)
def filter_on(
    df: pd.DataFrame,
    criteria: str,
    complement: bool = False,
) -> pd.DataFrame:
    """Return a dataframe filtered on a particular criteria.

    This method does not mutate the original DataFrame.

    This is super-sugary syntax that wraps the pandas `.query()` API, enabling
    users to use strings to quickly specify filters for filtering their
    dataframe. The intent is that `filter_on` as a verb better matches the
    intent of a pandas user than the verb `query`.

    This is intended to be the method-chaining equivalent of the following:

    ```python
    df = df[df["score"] < 3]
    ```

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `pd.DataFrame.query` instead.


    Examples:
        Filter students who failed an exam (scored less than 50).

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "student_id": ["S1", "S2", "S3"],
        ...     "score": [40, 60, 85],
        ... })
        >>> df
          student_id  score
        0         S1     40
        1         S2     60
        2         S3     85
        >>> df.filter_on("score < 50", complement=False)
          student_id  score
        0         S1     40

    Credit to Brant Peterson for the name.

    Args:
        df: A pandas DataFrame.
        criteria: A filtering criteria that returns an array or Series of
            booleans, on which pandas can filter on.
        complement: Whether to return the complement of the filter or not.
            If set to True, then the rows for which the criteria is False are
            retained instead.

    Returns:
        A filtered pandas DataFrame.
    """

    warnings.warn(
        "This function will be deprecated in a 1.x release. "
        "Kindly use `pd.DataFrame.query` instead.",
        DeprecationWarning,
        stacklevel=find_stack_level(),
    )

    if complement:
        return df.query(f"not ({criteria})")
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
    """Filter a date-based column based on certain criteria.

    This method does not mutate the original DataFrame.

    Dates may be finicky and this function builds on top of the *magic* from
    the pandas `to_datetime` function that is able to parse dates well.

    Additional options to parse the date type of your column may be found at
    the official pandas [documentation][datetime].

    [datetime]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": range(5, 9),
        ...     "dt": ["2021-11-12", "2021-12-15", "2022-01-03", "2022-01-09"],
        ... })
        >>> df
           a          dt
        0  5  2021-11-12
        1  6  2021-12-15
        2  7  2022-01-03
        3  8  2022-01-09
        >>> df.filter_date("dt", start_date="2021-12-01", end_date="2022-01-05")
           a         dt
        1  6 2021-12-15
        2  7 2022-01-03
        >>> df.filter_date("dt", years=[2021], months=[12])
           a         dt
        1  6 2021-12-15

    !!!note

        This method will cast your column to a Timestamp!

    !!!note

        This only affects the format of the `start_date` and `end_date`
        parameters. If there's an issue with the format of the DataFrame being
        parsed, you would pass `{'format': your_format}` to `column_date_options`.

    Args:
        df: The dataframe to filter on.
        column_name: The column which to apply the fraction transformation.
        start_date: The beginning date to use to filter the DataFrame.
        end_date: The end date to use to filter the DataFrame.
        years: The years to use to filter the DataFrame.
        months: The months to use to filter the DataFrame.
        days: The days to use to filter the DataFrame.
        column_date_options: Special options to use when parsing the date
            column in the original DataFrame. The options may be found at the
            official Pandas documentation.
        format: If you're using a format for `start_date` or `end_date`
            that is not recognized natively by pandas' `to_datetime` function, you
            may supply the format yourself. Python date and time formats may be
            found [here](http://strftime.org/).

    Returns:
        A filtered pandas DataFrame.
    """  # noqa: E501

    def _date_filter_conditions(conditions):
        """Taken from: https://stackoverflow.com/a/13616382."""
        return reduce(np.logical_and, conditions)

    if column_date_options is None:
        column_date_options = {}
    df[column_name] = pd.to_datetime(df[column_name], **column_date_options)

    _filter_list = []

    if start_date:
        start_date = pd.to_datetime(start_date, format=format)
        _filter_list.append(df[column_name] >= start_date)

    if end_date:
        end_date = pd.to_datetime(end_date, format=format)
        _filter_list.append(df[column_name] <= end_date)

    if years:
        _filter_list.append(df[column_name].dt.year.isin(years))

    if months:
        _filter_list.append(df[column_name].dt.month.isin(months))

    if days:
        _filter_list.append(df[column_name].dt.day.isin(days))

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
    """Filter a dataframe for values in a column that exist in the given iterable.

    This method does not mutate the original DataFrame.

    Assumes exact matching; fuzzy matching not implemented.

    Examples:
        Filter the dataframe to retain rows for which `names`
        are exactly `James` or `John`.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"names": ["Jane", "Jeremy", "John"], "foo": list("xyz")})
        >>> df
            names foo
        0    Jane   x
        1  Jeremy   y
        2    John   z
        >>> df.filter_column_isin(column_name="names", iterable=["James", "John"])
          names foo
        2  John   z

        This is the method-chaining alternative to:

        ```python
        df = df[df["names"].isin(["James", "John"])]
        ```

        If `complement=True`, then we will only get rows for which the names
        are neither `James` nor `John`.

    Args:
        df: A pandas DataFrame.
        column_name: The column on which to filter.
        iterable: An iterable. Could be a list, tuple, another pandas
            Series.
        complement: Whether to return the complement of the selection or
            not.

    Raises:
        ValueError: If `iterable` does not have a length of `1`
            or greater.

    Returns:
        A filtered pandas DataFrame.
    """  # noqa: E501
    if len(iterable) == 0:
        raise ValueError(
            "`iterable` kwarg must be given an iterable of length 1 "
            "or greater."
        )
    criteria = df[column_name].isin(iterable)

    if complement:
        return df[~criteria]
    return df[criteria]
