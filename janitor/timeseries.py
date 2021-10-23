""" Time series-specific data cleaning functions. """

import itertools
from typing import Dict, Union

import pandas as pd
import pandas_flavor as pf

from .utils import check
from .errors import JanitorError


@pf.register_dataframe_method
def fill_missing_timestamps(
    df: pd.DataFrame,
    frequency: str,
    first_time_stamp: pd.Timestamp = None,
    last_time_stamp: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Fills a DataFrame with missing timestamps based on a defined frequency.

    If timestamps are missing, this function will re-index the DataFrame.
    If timestamps are not missing, then the function will return the DataFrame
    unmodified.

    Functional usage example:

    ```python
    import pandas as pd
    import janitor.timeseries

    df = pd.DataFrame(...)

    df = janitor.timeseries.fill_missing_timestamps(
        df=df,
        frequency="1H",
    )
    ```

    Method chaining example:

    ```python
    import pandas as pd
    import janitor.timeseries

    df = (
        pd.DataFrame(...)
        .fill_missing_timestamps(frequency="1H")
    )
    ```

    :param df: DataFrame which needs to be tested for missing timestamps
    :param frequency: sampling frequency of the data.
        Acceptable frequency strings are available
        [here](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases).
        Check offset aliases under time series in user guide
    :param first_time_stamp: timestamp expected to start from;
        defaults to `None`. If no input is provided, assumes the
        minimum value in `time_series`.
    :param last_time_stamp: timestamp expected to end with; defaults to `None`.
        If no input is provided, assumes the maximum value in `time_series`.
    :returns: DataFrame that has a complete set of contiguous datetimes.
    """
    # Check all the inputs are the correct data type
    check("frequency", frequency, [str])
    check("first_time_stamp", first_time_stamp, [pd.Timestamp, type(None)])
    check("last_time_stamp", last_time_stamp, [pd.Timestamp, type(None)])

    if first_time_stamp is None:
        first_time_stamp = df.index.min()
    if last_time_stamp is None:
        last_time_stamp = df.index.max()

    # Generate expected timestamps
    expected_timestamps = pd.date_range(
        start=first_time_stamp, end=last_time_stamp, freq=frequency
    )

    return df.reindex(expected_timestamps)


def _get_missing_timestamps(
    df: pd.DataFrame,
    frequency: str,
    first_time_stamp: pd.Timestamp = None,
    last_time_stamp: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Return the timestamps that are missing in a DataFrame.

    This function takes in a DataFrame, and checks its index
    against a DataFrame that contains the expected timestamps.
    Here, we assume that the expected timestamps are going to be
    of a larger size than the timestamps available in the input
    DataFrame `df`.

    If there are any missing timestamps in the input DataFrame,
    this function will return those missing timestamps from the
    expected DataFrame.
    """
    expected_df = df.fill_missing_timestamps(
        frequency, first_time_stamp, last_time_stamp
    )

    missing_timestamps = expected_df.index.difference(df.index)

    return expected_df.loc[missing_timestamps]


@pf.register_dataframe_method
def sort_timestamps_monotonically(
    df: pd.DataFrame, direction: str = "increasing", strict: bool = False
) -> pd.DataFrame:
    """
    Sort DataFrame such that index is monotonic.

    If timestamps are monotonic, this function will return
    the DataFrame unmodified. If timestamps are not monotonic,
    then the function will sort the DataFrame.

    Functional usage example:

    ```python
    import pandas as pd
    import janitor.timeseries

    df = pd.DataFrame(...)

    df = janitor.timeseries.sort_timestamps_monotonically(
        direction="increasing"
    )
    ```

    Method chaining example:

    ```python
    import pandas as pd
    import janitor.timeseries

    df = (
        pd.DataFrame(...)
        .sort_timestamps_monotonically(direction="increasing")
    )
    ```

    :param df: DataFrame which needs to be tested for monotonicity.
    :param direction: type of monotonicity desired.
        Acceptable arguments are `'increasing'` or `'decreasing'`.
    :param strict: flag to enable/disable strict monotonicity.
        If set to `True`, will remove duplicates in the index
        by retaining first occurrence of value in index.
        If set to `False`, will not test for duplicates in the index;
        defaults to `False`.
    :returns: DataFrame that has monotonically increasing
        (or decreasing) timestamps.
    """
    # Check all the inputs are the correct data type
    check("df", df, [pd.DataFrame])
    check("direction", direction, [str])
    check("strict", strict, [bool])

    # Remove duplicates if requested
    if strict:
        df = df[~df.index.duplicated(keep="first")]

    # Sort timestamps
    if direction == "increasing":
        df = df.sort_index()
    else:
        df = df.sort_index(ascending=False)

    # Return the DataFrame
    return df


def _flag_jumps_single_col(
    df: pd.DataFrame,
    col: str,
    scale: str,
    direction: str,
    threshold: Union[int, float],
) -> pd.Series:
    """
    Creates a boolean column that flags whether or not the change
    between consecutive rows in the provided DataFrame column exceeds a
    provided threshold.

    Comparisons are always performed utilizing a GREATER THAN
    threshold check. Thus, flags correspond to values that EXCEED
    the provided threshold.

    The method used to create consecutive row comparisons is set by the
    `scale` argument. A `scale=absolute` corresponds to a difference
    method (`.diff()`) and a `scale=percentage` corresponds to a
    percentage change methods (`pct_change()`).

    A `direction` argument is used to determine how to handle the sign
    of the difference or percentage change methods.
    A `direction=increasing` will only consider consecutive rows that
    are increasing in value and exceeding the provided threshold.
    A `direction=decreasing` will only consider consecutive rows that
    are decreasing in value and exceeding the provided threshold.
    If `direction=any`, the absolute value is taken for both the
    difference method and the percentage change methods and the sign
    between consecutive rows is ignored.
    """
    check("scale", scale, [str])
    check("direction", direction, [str])
    check("threshold", threshold, [int, float])

    scale_types = ["absolute", "percentage"]
    if scale not in scale_types:
        raise JanitorError(
            f"Unrecognized scale: '{scale}'. Must be one of: {scale_types}."
        )

    direction_types = ["increasing", "decreasing", "any"]
    if direction not in direction_types:
        raise JanitorError(
            f"Unrecognized direction: '{direction}'. "
            + f"Must be one of: {direction_types}."
        )

    if threshold < 0:
        raise JanitorError(
            f"Unrecognized threshold: {threshold}. "
            + "This value must be >= 0.0. "
            + "Use 'direction' to specify positive or negative intent."
        )

    single_col = df[col]
    single_col_diffs = single_col.diff()

    if scale == "percentage":
        single_col_pcts = single_col.pct_change()

        if direction == "increasing":
            # Using diffs ensures correct sign is used for incr/decr
            # (see issue #711)
            out = (single_col_diffs > 0) & (single_col_pcts.abs() > threshold)

        elif direction == "decreasing":
            # Using diffs ensures correct sign is used for incr/decr
            # (see issue #711)
            out = (single_col_diffs < 0) & (single_col_pcts.abs() > threshold)

        else:
            out = single_col_pcts.abs() > threshold

    else:
        if direction == "increasing":
            out = single_col_diffs > threshold

        elif direction == "decreasing":
            out = (single_col_diffs < 0) & (single_col_diffs.abs() > threshold)

        else:
            out = single_col_diffs.abs() > threshold

    out = out.astype(int)

    return out


@pf.register_dataframe_method
def flag_jumps(
    df: pd.DataFrame,
    scale: Union[str, Dict[str, str]] = "percentage",
    direction: Union[str, Dict[str, str]] = "any",
    threshold: Union[int, float, Dict[str, Union[int, float]]] = 0.0,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Create boolean column(s) that flag whether or not the change
    between consecutive rows exceeds a provided threshold.

    Functional usage example:

    ```python
    import pandas as pd
    import janitor.timeseries

    df = pd.DataFrame(...)

    df = flag_jumps(
        df=df,
        scale="absolute",
        direction="any",
        threshold=2,
    )
    ```

    Method chaining example:

    ```python
    import pandas as pd
    import janitor.timeseries

    df = (
        pd.DatFrame(...)
        .flag_jumps(
            scale="absolute",
            direction="any",
            threshold=2,
        )
    )
    ```

    Detailed chaining examples:

    ```python
    # Applies specified criteria across all columns of the DataFrame
    # Appends a flag column for each column in the DataFrame
    df = (
        pd.DataFrame(...)
        .flag_jumps(
            scale="absolute",
            direction="any",
            threshold=2
        )
    )

    # Applies specific criteria to certain DataFrame columns
    # Applies default criteria to columns not specifically listed
    # Appends a flag column for each column in the DataFrame
    df = (
        pd.DataFrame(...)
        .flag_jumps(
            scale=dict(col1="absolute", col2="percentage"),
            direction=dict(col1="increasing", col2="any"),
            threshold=dict(col1=1, col2=0.5),
        )
    )

    # Applies specific criteria to certain DataFrame columns
    # Applies default criteria to columns not specifically listed
    # Appends a flag column for each column in the DataFrame
    df = (
        pd.DataFrame(...)
        .flag_jumps(
            scale=dict(col1="absolute"),
            direction=dict(col2="increasing"),
        )
    )

    # Applies specific criteria to certain DataFrame columns
    # Applies default criteria to columns not specifically listed
    # Appends a flag column for only those columns found in
    #   specified criteria
    df = (
        pd.DataFrame(...)
        .flag_jumps(
            scale=dict(col1="absolute"),
            threshold=dict(col2=1),
            strict=True,
        )
    )
    ```

    :param df: DataFrame which needs to be flagged for changes between
        consecutive rows above a certain threshold.
    :param scale: Type of scaling approach to use.
        Acceptable arguments are `'absolute'` (consider the difference
        between rows) and `'percentage'` (consider the percentage
        change between rows); defaults to `'percentage'`.
    :param direction: Type of method used to handle the sign change when
        comparing consecutive rows.
        Acceptable arguments are `'increasing'` (only consider rows
        that are increasing in value), `'decreasing'` (only consider
        rows that are decreasing in value), and `'any'` (consider rows
        that are either increasing or decreasing; sign is ignored);
        defaults to `'any'`.
    :param threshold: The value to check if consecutive row comparisons
        exceed. Always uses a greater than comparison. Must be `>= 0.0`;
        defaults to `0.0`.
    :param strict: flag to enable/disable appending of a flag column for
        each column in the provided DataFrame. If set to `True`, will
        only append a flag column for those columns found in at least
        one of the input dictionaries. If set to `False`, will appen
        a flag column for each column found in the provided DataFrame.
        If criteria is not specified, the defaults for each criteria
        is used; defaults to `False`.
    :returns: DataFrame that has `flag jump` columns.
    :raises JanitorError: if `strict=True` and at least one of
        `scale`, `direction`, or `threshold` inputs is not a
        dictionary.
    :raises JanitorError: if `scale` is not one of
        `("absolute", "percentage")`.
    :raises JanitorError: if `direction` is not one of
        `("increasing", "decreasing", "any")`.
    :raises JanitorError: if `threshold` is less than `0.0`.
    """
    df = df.copy()

    if strict:
        if (
            any(isinstance(arg, dict) for arg in (scale, direction, threshold))
            is False
        ):
            raise JanitorError(
                "When enacting 'strict=True', 'scale', 'direction', or "
                + "'threshold' must be a dictionary."
            )

        # Only append a flag col for the cols that appear
        # in at least one of the input dicts
        arg_keys = [
            arg.keys()
            for arg in (scale, direction, threshold)
            if isinstance(arg, dict)
        ]
        cols = set(itertools.chain.from_iterable(arg_keys))

    else:
        # Append a flag col for each col in the DataFrame
        cols = df.columns

    columns_to_add = {}
    for col in sorted(cols):

        # Allow arguments to be a mix of dict and single instances
        s = scale.get(col, "percentage") if isinstance(scale, dict) else scale
        d = (
            direction.get(col, "any")
            if isinstance(direction, dict)
            else direction
        )
        t = (
            threshold.get(col, 0.0)
            if isinstance(threshold, dict)
            else threshold
        )

        columns_to_add[f"{col}_jump_flag"] = _flag_jumps_single_col(
            df, col, scale=s, direction=d, threshold=t
        )

    df = df.assign(**columns_to_add)

    return df
