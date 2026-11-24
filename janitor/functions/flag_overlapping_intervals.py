"""Implementation source for `flag_overlapping_intervals`."""

from typing import Hashable, List, Optional, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check_column


@pf.register_dataframe_method
def flag_overlapping_intervals(
    df: pd.DataFrame,
    start_column: Hashable,
    end_column: Hashable,
    column_name: Hashable = "overlap_flag",
    by: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Flag rows whose time intervals overlap with another interval.

    Intervals are treated as closed, meaning that two intervals sharing
    an endpoint are considered to overlap.

    Args:
        df: The DataFrame containing the intervals.
        start_column: Column containing interval start values.
        end_column: Column containing interval end values.
        column_name: Name of the output flag column.
        by: Optional column or list of columns used to compare intervals
            within separate groups.

    Returns:
        A copy of the DataFrame with an integer flag column. A value of
        1 indicates that the interval overlaps with at least one other
        interval; 0 indicates that it does not.

    Raises:
        ValueError: If a required column is missing, the output column
            already exists, or an interval starts after it ends.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "start": [1, 3, 6],
        ...         "end": [4, 5, 8],
        ...     }
        ... )
        >>> df.flag_overlapping_intervals("start", "end")
    """
    check_column(df, [start_column, end_column])
    check_column(df, [column_name], present=False)

    if by is not None:
        check_column(df, by)

    result = df.copy()
    flags = np.zeros(len(df), dtype=int)

    if by is None:
        groups = [np.arange(len(df))]
    else:
        groups = df.groupby(
            by,
            sort=False,
            dropna=False,
            observed=True,
        ).indices.values()

    starts = df[start_column]
    ends = df[end_column]

    for positions in groups:
        positions = np.asarray(positions, dtype=int)

        valid_positions = positions[
            starts.iloc[positions].notna().to_numpy()
            & ends.iloc[positions].notna().to_numpy()
        ]

        if len(valid_positions) < 2:
            continue

        valid_starts = starts.iloc[valid_positions]
        valid_ends = ends.iloc[valid_positions]

        invalid_intervals = valid_starts.gt(valid_ends)

        if invalid_intervals.any():
            raise ValueError(
                "Each interval's start value must be less than or "
                "equal to its end value."
            )

        order = np.argsort(valid_starts.to_numpy(), kind="stable")

        sorted_positions = valid_positions[order]
        sorted_starts = valid_starts.iloc[order].to_numpy()
        sorted_ends = valid_ends.iloc[order].to_numpy()

        max_end = None

        for i, (start, end) in enumerate(zip(sorted_starts, sorted_ends)):
            if max_end is not None and start <= max_end:
                flags[sorted_positions[i]] = 1

            if i + 1 < len(sorted_starts) and sorted_starts[i + 1] <= end:
                flags[sorted_positions[i]] = 1

            if max_end is None or end > max_end:
                max_end = end

    result[column_name] = flags

    return result
