import numbers
import re
import warnings
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Hashable, List, Tuple, Union

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_list_like, is_scalar
from pandas.errors import OutOfBoundsDatetime

from janitor.utils import deprecated_alias, find_stack_level, refactored_function

_EXCEL_EPOCH = datetime(1899, 12, 30)
_EXCEL_DATE_PATTERN = re.compile(r"^[0-9]{5}(?:\.[0-9]*)?$")
_TIME_NUMBER_PATTERN = re.compile(r"^0(?:\.[0-9]*)?$")
_TIME_SCI_NUMBER_PATTERN = re.compile(r"^[1-9](?:\\.[0-9]*)?E-[0-9]+$", re.IGNORECASE)
_TIME_12HR_PATTERN = re.compile(
    r"^([0]?[1-9]|1[0-2]):([0-5][0-9])(?::([0-5][0-9]))?\s*([AP]M)$",
    re.IGNORECASE,
)
_TIME_24HR_PATTERN = re.compile(
    r"^([0-1]?[0-9]|2[0-3]):([0-5][0-9])(?::([0-5][0-9]))?$"
)
_TIME_POSIX_PATTERN = re.compile(
    r"^1899-12-31 (?:([0-1]?[0-9]|2[0-3]):([0-5][0-9])(?::([0-5][0-9]))?)?.*$"
)


def _prepare_input(
    values: Any,
) -> Tuple[List[Any], Callable[[List[Any]], Any]]:
    if isinstance(values, pd.Series):
        return values.tolist(), lambda result: pd.Series(
            result, index=values.index, name=values.name
        )
    if is_scalar(values):
        return [values], lambda result: result[0]
    if is_list_like(values):
        return list(values), lambda result: result
    return [values], lambda result: result[0]


def _apply_timezone(timestamp: pd.Timestamp, tz: str | None) -> pd.Timestamp:
    if tz is None:
        return timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(tz)
    return timestamp.tz_convert(tz)


def _excel_numeric_to_timestamp(value: float) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, unit="D", origin="1899-12-30")
    return timestamp.round("s")


def _looks_like_time(value: str) -> bool:
    return bool(
        _TIME_NUMBER_PATTERN.match(value)
        or _TIME_SCI_NUMBER_PATTERN.match(value)
        or _TIME_12HR_PATTERN.match(value)
        or _TIME_24HR_PATTERN.match(value)
        or _TIME_POSIX_PATTERN.match(value)
    )


def _format_failed_strings(values: List[str], out_class: str) -> str:
    unique_values = sorted(set(values))
    if len(unique_values) > 10:
        preview = ", ".join(f'"{val}"' for val in unique_values[:9])
        remainder = len(unique_values) - 9
        not_converted_values = f"{preview} ... and {remainder} other values."
    else:
        not_converted_values = ", ".join(f'"{val}"' for val in unique_values)
    return (
        "Not all character strings converted to class "
        f"{out_class}. Values not converted were: {not_converted_values}"
    )


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
        >>> df.convert_excel_date("date")
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
        column_name: pd.to_datetime(df[column_name], unit="D", origin="1899-12-30")
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
        >>> df.convert_matlab_date("date")
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


def excel_time_to_numeric(time_value: Any, round_seconds: bool = True) -> Any:
    """Convert Excel time representations to numeric seconds.

    The returned values represent the number of seconds between 0 and 86400.

    Examples:
        >>> import janitor as jn
        >>> jn.excel_time_to_numeric("0.5")
        43200
        >>> jn.excel_time_to_numeric("12:30 PM")
        45000
        >>> jn.excel_time_to_numeric("14:30:15")
        52215

    Args:
        time_value: A scalar or list-like of time values to convert.
        round_seconds: Whether to round output seconds to integers.

    Returns:
        Numeric seconds (scalar or list) corresponding to `time_value`.

    Raises:
        ValueError: If numeric values are outside the 0-1 range or strings
            do not match supported time formats.
        TypeError: If `time_value` contains unsupported types.
    """
    if not isinstance(round_seconds, bool):
        raise TypeError("round_seconds should be a boolean.")

    values, finalize = _prepare_input(time_value)
    results: List[Any] = []
    invalid_strings: List[str] = []

    for value in values:
        if pd.isna(value):
            results.append(None)
            continue

        if isinstance(value, bool):
            raise TypeError("If given as a boolean, all values must be missing.")

        if isinstance(value, numbers.Number):
            numeric_value = float(value)
            if not (0 <= numeric_value < 1):
                raise ValueError(
                    "When numeric, all `time_value`s must be between 0 and 1 "
                    "(exclusive of 1)."
                )
            seconds = numeric_value * 86400
            if round_seconds:
                seconds = int(round(seconds))
            results.append(seconds)
            continue

        if isinstance(value, (pd.Timestamp, datetime)):
            seconds = (
                value.hour * 3600
                + value.minute * 60
                + value.second
                + value.microsecond / 1_000_000
            )
            if round_seconds:
                seconds = int(round(seconds))
            results.append(seconds)
            continue

        if isinstance(value, time):
            seconds = (
                value.hour * 3600
                + value.minute * 60
                + value.second
                + value.microsecond / 1_000_000
            )
            if round_seconds:
                seconds = int(round(seconds))
            results.append(seconds)
            continue

        if isinstance(value, str):
            text = value.strip()
            if _TIME_NUMBER_PATTERN.match(text) or _TIME_SCI_NUMBER_PATTERN.match(
                text
            ):
                numeric_value = float(text)
                if not (0 <= numeric_value < 1):
                    raise ValueError(
                        "When numeric, all `time_value`s must be between 0 and 1 "
                        "(exclusive of 1)."
                    )
                seconds = numeric_value * 86400
            elif _TIME_POSIX_PATTERN.match(text):
                match = _TIME_POSIX_PATTERN.match(text)
                hours = match.group(1) or "0"
                minutes = match.group(2) or "0"
                seconds = match.group(3) or "0"
                seconds = (
                    int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                )
            elif _TIME_12HR_PATTERN.match(text):
                match = _TIME_12HR_PATTERN.match(text)
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = int(match.group(3) or 0)
                meridiem = match.group(4).lower()
                if hours == 12:
                    hours = 0
                if meridiem == "pm":
                    hours += 12
                seconds = hours * 3600 + minutes * 60 + seconds
            elif _TIME_24HR_PATTERN.match(text):
                match = _TIME_24HR_PATTERN.match(text)
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = int(match.group(3) or 0)
                seconds = hours * 3600 + minutes * 60 + seconds
            else:
                invalid_strings.append(value)
                results.append(None)
                continue

            if round_seconds:
                seconds = int(round(seconds))
            results.append(seconds)
            continue

        raise TypeError("time_value entries must be numeric, time-like, or strings.")

    if invalid_strings:
        invalid_values = ", ".join(sorted(set(invalid_strings)))
        raise ValueError(
            "The following character strings did not match an interpretable "
            f"character format for time conversion: {invalid_values}"
        )

    return finalize(results)


def _coerce_character_result(
    value: Any, out_class: str
) -> date | pd.Timestamp:
    if out_class == "date":
        if isinstance(value, pd.Timestamp):
            return value.date()
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
    else:
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, datetime):
            return pd.Timestamp(value)
        if isinstance(value, date):
            return pd.Timestamp(datetime.combine(value, time.min))
    raise TypeError(
        "`character_fun` must return date or datetime objects for string inputs."
    )


def _timestamp_to_output(
    timestamp: pd.Timestamp, out_class: str, tz: str | None
) -> Any:
    if out_class == "date":
        return timestamp.date()
    timestamp = _apply_timezone(timestamp, tz)
    return timestamp.to_pydatetime()


def _convert_string_value(
    value: str,
    out_class: str,
    tz: str | None,
    character_fun: Callable[[str], Any],
    failed_strings: List[str],
) -> Any:
    text = value.strip()
    if _EXCEL_DATE_PATTERN.match(text):
        try:
            timestamp = _excel_numeric_to_timestamp(float(text))
        except (OutOfBoundsDatetime, ValueError):
            failed_strings.append(value)
            return None
        return _timestamp_to_output(timestamp, out_class, tz)

    if _looks_like_time(text):
        if out_class == "date":
            failed_strings.append(value)
            return None
        try:
            seconds = excel_time_to_numeric(text, round_seconds=True)
        except ValueError:
            failed_strings.append(value)
            return None
        timestamp = pd.Timestamp(_EXCEL_EPOCH + timedelta(seconds=seconds))
        return _timestamp_to_output(timestamp, out_class, tz)

    try:
        converted = character_fun(text)
    except Exception:
        failed_strings.append(value)
        return None

    if pd.isna(converted):
        failed_strings.append(value)
        return None

    coerced = _coerce_character_result(converted, out_class)
    if out_class == "date":
        return coerced
    return _timestamp_to_output(coerced, out_class, tz)


def _convert_value(
    value: Any,
    out_class: str,
    tz: str | None,
    character_fun: Callable[[str], Any],
    failed_strings: List[str],
) -> Any:
    if pd.isna(value):
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return _timestamp_to_output(pd.Timestamp(value), out_class, tz)

    if isinstance(value, date) and not isinstance(value, datetime):
        if out_class == "date":
            return value
        timestamp = pd.Timestamp(datetime.combine(value, time.min))
        return _timestamp_to_output(timestamp, out_class, tz)

    if isinstance(value, time):
        if out_class == "date":
            raise TypeError("time values cannot be converted to dates.")
        timestamp = pd.Timestamp(datetime.combine(_EXCEL_EPOCH.date(), value))
        return _timestamp_to_output(timestamp, out_class, tz)

    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        timestamp = _excel_numeric_to_timestamp(float(value))
        return _timestamp_to_output(timestamp, out_class, tz)

    if isinstance(value, str):
        return _convert_string_value(
            value,
            out_class=out_class,
            tz=tz,
            character_fun=character_fun,
            failed_strings=failed_strings,
        )

    raise TypeError(f"Unsupported value type: {type(value).__name__}")


def _convert_to_date_or_datetime(
    x: Any,
    out_class: str,
    tz: str | None,
    character_fun: Callable[[str], Any] | None,
    string_conversion_failure: str,
) -> Any:
    if string_conversion_failure not in {"error", "warning"}:
        raise ValueError(
            "string_conversion_failure must be one of 'error' or 'warning'."
        )

    if character_fun is None:
        character_fun = lambda value: pd.to_datetime(value, errors="coerce")

    values, finalize = _prepare_input(x)
    failed_strings: List[str] = []
    results: List[Any] = []

    for value in values:
        results.append(
            _convert_value(
                value,
                out_class=out_class,
                tz=tz,
                character_fun=character_fun,
                failed_strings=failed_strings,
            )
        )

    if failed_strings:
        message = _format_failed_strings(failed_strings, out_class)
        if string_conversion_failure == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=find_stack_level())

    return finalize(results)


def convert_to_date(
    x: Any,
    character_fun: Callable[[str], Any] | None = None,
    string_conversion_failure: str = "error",
) -> Any:
    """Parse mixed date formats into Python ``date`` objects.

    Examples:
        >>> import janitor as jn
        >>> dates = ["2009-07-06", "40000", "40000.1", None]
        >>> jn.convert_to_date(dates)  # doctest: +ELLIPSIS
        [datetime.date(2009, 7, 6), ..., None]

    Args:
        x: A scalar or list-like of mixed date values.
        character_fun: Optional function used to parse non-numeric strings.
        string_conversion_failure: Whether to raise an error or warn when
            string values fail to parse.

    Returns:
        Parsed dates as ``datetime.date`` objects, with ``None`` for missing.
    """
    return _convert_to_date_or_datetime(
        x,
        out_class="date",
        tz=None,
        character_fun=character_fun,
        string_conversion_failure=string_conversion_failure,
    )


def convert_to_datetime(
    x: Any,
    tz: str | None = "UTC",
    character_fun: Callable[[str], Any] | None = None,
    string_conversion_failure: str = "error",
) -> Any:
    """Parse mixed datetime formats into Python ``datetime`` objects.

    Time-only values are anchored to Excel's origin (1899-12-30).

    Examples:
        >>> import janitor as jn
        >>> datetimes = ["2009-07-06", "40000.1", "40000", None]
        >>> jn.convert_to_datetime(datetimes, tz=None)  # doctest: +ELLIPSIS
        [datetime.datetime(2009, 7, 6, 0, 0), ..., None]

    Args:
        x: A scalar or list-like of mixed datetime values.
        tz: Timezone to assign to naive datetime values. Use ``None`` to
            return naive datetimes.
        character_fun: Optional function used to parse non-numeric strings.
        string_conversion_failure: Whether to raise an error or warn when
            string values fail to parse.

    Returns:
        Parsed datetimes as ``datetime.datetime`` objects, with ``None`` for missing.
    """
    if tz is not None and not isinstance(tz, str):
        raise TypeError("tz should be a string or None.")

    return _convert_to_date_or_datetime(
        x,
        out_class="datetime",
        tz=tz,
        character_fun=character_fun,
        string_conversion_failure=string_conversion_failure,
    )


def sas_numeric_to_date(
    date_num: Any = None,
    datetime_num: Any = None,
    time_num: Any = None,
    tz: str = "UTC",
) -> Any:
    """Convert SAS date, time, or datetime numbers to Python objects.

    Examples:
        >>> import janitor as jn
        >>> jn.sas_numeric_to_date(date_num=15639)
        datetime.date(2002, 10, 26)
        >>> jn.sas_numeric_to_date(datetime_num=1217083532, tz="UTC")
        datetime.datetime(1998, 7, 26, 14, 45, 32, tzinfo=...)
        >>> jn.sas_numeric_to_date(time_num=3600)
        datetime.time(1, 0)

    Args:
        date_num: Days since 1960-01-01.
        datetime_num: Seconds since 1960-01-01.
        time_num: Seconds since midnight.
        tz: Timezone for datetime outputs.

    Returns:
        Date, datetime, or time objects corresponding to SAS numeric values.

    Raises:
        ValueError: If invalid argument combinations are provided.
        TypeError: If inputs are non-numeric.
    """
    has_date = date_num is not None
    has_datetime = datetime_num is not None
    has_time = time_num is not None

    if not isinstance(tz, str):
        raise TypeError("tz should be a string.")
    if tz != "UTC":
        warnings.warn(
            "SAS may not properly store timezones other than UTC. "
            "Consider confirming the accuracy of the resulting data.",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    if has_date and has_datetime:
        raise ValueError("Must not give both `date_num` and `datetime_num`")
    if has_time and has_datetime:
        raise ValueError("Must not give both `time_num` and `datetime_num`")
    if not (has_date or has_datetime or has_time):
        raise ValueError(
            "Must give one of `date_num`, `datetime_num`, `time_num`, "
            "or `date_num` and `time_num`."
        )

    def _as_series(values: Any, name: str) -> pd.Series:
        if isinstance(values, pd.Series):
            return values.copy()
        if is_scalar(values):
            return pd.Series([values], name=name)
        if is_list_like(values):
            return pd.Series(list(values), name=name)
        return pd.Series([values], name=name)

    def _numeric_series(values: Any, name: str) -> pd.Series:
        series = _as_series(values, name)
        if series.dropna().apply(lambda val: isinstance(val, bool)).any():
            raise TypeError(f"`{name}` must be numeric.")
        try:
            return pd.to_numeric(series, errors="raise")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"`{name}` must be numeric.") from exc

    base_index = None
    if has_date and isinstance(date_num, pd.Series):
        base_index = date_num.index
    if base_index is None and has_datetime and isinstance(datetime_num, pd.Series):
        base_index = datetime_num.index
    if base_index is None and has_time and isinstance(time_num, pd.Series):
        base_index = time_num.index

    if has_date and has_time:
        date_series = _numeric_series(date_num, "date_num")
        time_series = _numeric_series(time_num, "time_num")

        if len(date_series) != len(time_series):
            if len(date_series) == 1:
                time_series = time_series.reindex(
                    date_series.index, fill_value=time_series.iloc[0]
                )
            elif len(time_series) == 1:
                date_series = date_series.reindex(
                    time_series.index, fill_value=date_series.iloc[0]
                )
            else:
                raise ValueError(
                    "`date_num` and `time_num` must be the same length."
                )

        if not (date_series.isna() == time_series.isna()).all():
            raise ValueError(
                "The same values are not NA for both `date_num` and `time_num`"
            )

        if not ((time_series.isna()) | (time_series >= 0)).all():
            raise ValueError("`time_num` must be non-negative")
        if not ((time_series.isna()) | (time_series <= 86400)).all():
            raise ValueError(
                "`time_num` must be within the number of seconds in a day (<= 86400)"
            )

        datetime_series = date_series * 86400 + time_series
        timestamps = pd.to_datetime(
            datetime_series, unit="s", origin="1960-01-01"
        )
        results = [
            None
            if pd.isna(ts)
            else _timestamp_to_output(pd.Timestamp(ts), "datetime", tz)
            for ts in timestamps
        ]
    elif has_datetime:
        datetime_series = _numeric_series(datetime_num, "datetime_num")
        timestamps = pd.to_datetime(
            datetime_series, unit="s", origin="1960-01-01"
        )
        results = [
            None
            if pd.isna(ts)
            else _timestamp_to_output(pd.Timestamp(ts), "datetime", tz)
            for ts in timestamps
        ]
    elif has_date:
        date_series = _numeric_series(date_num, "date_num")
        timestamps = pd.to_datetime(date_series, unit="D", origin="1960-01-01")
        results = [
            None if pd.isna(ts) else pd.Timestamp(ts).date()
            for ts in timestamps
        ]
    else:
        time_series = _numeric_series(time_num, "time_num")
        if not ((time_series.isna()) | (time_series >= 0)).all():
            raise ValueError("`time_num` must be non-negative")
        if not ((time_series.isna()) | (time_series <= 86400)).all():
            raise ValueError(
                "`time_num` must be within the number of seconds in a day (<= 86400)"
            )
        results = []
        for seconds in time_series:
            if pd.isna(seconds):
                results.append(None)
                continue
            seconds = float(seconds)
            if seconds == 86400:
                seconds = 0
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining = seconds % 60
            whole_seconds = int(remaining)
            microseconds = int(round((remaining - whole_seconds) * 1_000_000))
            results.append(
                time(
                    hour=hours,
                    minute=minutes,
                    second=whole_seconds,
                    microsecond=microseconds,
                )
            )

    if base_index is not None:
        return pd.Series(results, index=base_index)
    if len(results) == 1:
        return results[0]
    return results


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
        >>> df.convert_unix_date("date")
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
