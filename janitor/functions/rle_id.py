"""Implementation of the `rle_id` function."""

import copy
from typing import Hashable, Iterable, Union

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_list_like
from pandas.core.groupby.generic import DataFrameGroupBy

from janitor.utils import check_column


@pf.register_dataframe_groupby_method
@pf.register_dataframe_method
def rle_id(
    df: Union[pd.DataFrame, DataFrameGroupBy],
    column_names: Union[Hashable, Iterable[Hashable]],
    new_column_name: Hashable = "rle_id",
) -> Union[pd.DataFrame, DataFrameGroupBy]:
    """Generate run-length encoding IDs for consecutive identical values.

    This function assigns a unique ID to each consecutive run of identical
    values. When the value changes, the ID increments. This is useful for
    grouping consecutive runs separately, even if they have the same value.

    This method does not mutate the original DataFrame.

    Examples:
        Single column example:

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "grp": ["A", "A", "B", "B", "A", "A"],
        ...         "value": [1, 2, 3, 4, 5, 6],
        ...     }
        ... )
        >>> df
          grp  value
        0   A      1
        1   A      2
        2   B      3
        3   B      4
        4   A      5
        5   A      6
        >>> df.rle_id("grp")
          grp  value  rle_id
        0   A      1       1
        1   A      2       1
        2   B      3       2
        3   B      4       2
        4   A      5       3
        5   A      6       3

        Multiple columns example:

        >>> df = pd.DataFrame(
        ...     {
        ...         "a": [1, 1, 2, 2, 1],
        ...         "b": ["x", "x", "x", "y", "x"],
        ...         "value": [10, 20, 30, 40, 50],
        ...     }
        ... )
        >>> df
           a  b  value
        0  1  x     10
        1  1  x     20
        2  2  x     30
        3  2  y     40
        4  1  x     50
        >>> df.rle_id(["a", "b"])
           a  b  value  rle_id
        0  1  x     10       1
        1  1  x     20       1
        2  2  x     30       2
        3  2  y     40       3
        4  1  x     50       4

        Using result for grouped aggregation:

        >>> df = pd.DataFrame(
        ...     {
        ...         "grp": ["A", "A", "B", "B", "A"],
        ...         "value": [1, 2, 3, 4, 5],
        ...     }
        ... )
        >>> df.rle_id("grp").groupby(["grp", "rle_id"])["value"].sum()
        grp  rle_id
        A    1          3
             3          5
        B    2          7
        Name: value, dtype: int64

    Args:
        df: A pandas DataFrame or GroupBy object.
        column_names: A column name or an iterable of column names to
            use for computing the run-length encoding IDs.
        new_column_name: The name of the new column containing the
            run-length encoding IDs. Defaults to `"rle_id"`.

    Raises:
        ValueError: If any of the specified columns do not exist in the
            DataFrame.
        ValueError: If `new_column_name` already exists in the DataFrame.

    Returns:
        A pandas DataFrame or GroupBy object with the new column.
    """
    grp = None
    if isinstance(df, DataFrameGroupBy):
        grp = copy.copy(df)
        group_keys = grp.keys
        df = grp.obj
    else:
        group_keys = None

    if not is_list_like(column_names):
        column_names = [column_names]
    elif isinstance(column_names, tuple) and column_names in df.columns:
        column_names = [column_names]
    else:
        column_names = list(column_names)

    check_column(df, column_names)
    check_column(df, [new_column_name], present=False)

    def _values_changed(col):
        prev = col.shift()
        both_notna = col.notna() & prev.notna()
        return (both_notna & (col != prev)) | (col.isna() != prev.isna())

    def _compute_rle(frame):
        if len(column_names) == 1:
            changed = _values_changed(frame[column_names[0]])
        else:
            changed = pd.Series(False, index=frame.index)
            for col in column_names:
                changed = changed | _values_changed(frame[col])
        if len(changed) > 0:
            changed.iloc[0] = True
        return changed.cumsum()

    if group_keys is not None:
        rle_ids = df.groupby(group_keys, group_keys=False).apply(
            _compute_rle, include_groups=False
        )
        if isinstance(rle_ids, pd.DataFrame):
            rle_ids = rle_ids.stack().droplevel(0)
    else:
        rle_ids = _compute_rle(df)

    result = df.assign(**{new_column_name: rle_ids})
    if grp is not None:
        grp.obj = result
        return grp
    return result
