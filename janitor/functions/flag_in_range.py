"""Implementation source for `flag_in_range`."""

from typing import Hashable, Optional, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check_column


@pf.register_dataframe_method
def flag_in_range(
    df: pd.DataFrame,
    column_name: Hashable,
    low: Union[int, float],
    high: Union[int, float],
    inclusive: bool = True,
    flag_column_name: Optional[Hashable] = "in_range_flag",
) -> pd.DataFrame:
    """Creates a new column to indicate whether values in a column fall
    within (or outside) a given range.

    A flag value of `1` indicates the row's value in `column_name` falls
    outside the `[low, high]` range; `0` indicates it falls within range,
    consistent with the convention used by `flag_nulls`.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({"a": [1, 5, 10, 15, 20]})
        >>> df.flag_in_range(column_name="a", low=5, high=15)
           a  in_range_flag
        0   1              1
        1   5              0
        2  10              0
        3  15              0
        4  20              1

    Args:
        df: Input pandas DataFrame.
        column_name: Name of the column to check.
        low: Lower bound of the range.
        high: Upper bound of the range.
        inclusive: Whether the range bounds (`low` and `high`) are
            themselves considered "in range". Defaults to True.
        flag_column_name: Name for the output flag column.

    Raises:
        ValueError: If `column_name` is not present in the DataFrame.
        ValueError: If `flag_column_name` is already present in the
            DataFrame.

    Returns:
        Input dataframe with the range flag column appended.

    <!--
    # noqa: DAR402
    -->
    """
    check_column(df, [column_name])
    check_column(df, [flag_column_name], present=False)

    series = df[column_name]

    if inclusive:
        in_range = series.between(low, high)
    else:
        in_range = (series > low) & (series < high)

    out_of_range = np.logical_not(in_range)

    df = df.copy()
    df[flag_column_name] = out_of_range.astype(int)
    return df
