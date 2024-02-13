"""Implementation of the `groupby_topk` function"""

from typing import Hashable, Union

import pandas as pd
import pandas_flavor as pf

from janitor.utils import check, check_column, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(groupby_column_name="by", sort_column_name="column")
def groupby_topk(
    df: pd.DataFrame,
    by: Union[list, Hashable],
    column: Hashable,
    k: int,
    dropna: bool = True,
    ascending: bool = True,
    ignore_index: bool = True,
) -> pd.DataFrame:
    """Return top `k` rows from a groupby of a set of columns.

    Returns a DataFrame that has the top `k` values per `column`,
    grouped by `by`. Under the hood it uses `nlargest/nsmallest`,
    for numeric columns, which avoids sorting the entire dataframe,
    and is usually more performant. For non-numeric columns, `pd.sort_values`
    is used.
    No sorting is done to the `by` column(s); the order is maintained
    in the final output.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "age": [20, 23, 22, 43, 21],
        ...         "id": [1, 4, 6, 2, 5],
        ...         "result": ["pass", "pass", "fail", "pass", "fail"],
        ...     }
        ... )
        >>> df
           age  id result
        0   20   1   pass
        1   23   4   pass
        2   22   6   fail
        3   43   2   pass
        4   21   5   fail

        Ascending top 3:

        >>> df.groupby_topk(by="result", column="age", k=3)
           age  id result
        0   20   1   pass
        1   23   4   pass
        2   43   2   pass
        3   21   5   fail
        4   22   6   fail

        Descending top 2:

        >>> df.groupby_topk(
        ...     by="result", column="age", k=2, ascending=False, ignore_index=False
        ... )
           age  id result
        3   43   2   pass
        1   23   4   pass
        2   22   6   fail
        4   21   5   fail

    Args:
        df: A pandas DataFrame.
        by: Column name(s) to group input DataFrame `df` by.
        column: Name of the column that determines `k` rows
            to return.
        k: Number of top rows to return for each group.
        dropna: If `True`, and `NA` values exist in `by`, the `NA`
            values are not used in the groupby computation to get the relevant
            `k` rows. If `False`, and `NA` values exist in `by`, then the `NA`
            values are used in the groupby computation to get the relevant
            `k` rows.
        ascending: If `True`, the smallest top `k` rows,
            determined by `column` are returned; if `False, the largest top `k`
            rows, determined by `column` are returned.
        ignore_index: If `True`, the original index is ignored.
            If `False`, the original index for the top `k` rows is retained.

    Raises:
        ValueError: If `k` is less than 1.

    Returns:
        A pandas DataFrame with top `k` rows per `column`, grouped by `by`.
    """  # noqa: E501

    if isinstance(by, Hashable):
        by = [by]

    check("by", by, [Hashable, list])

    check_column(df, [column])
    check_column(df, by)

    if k < 1:
        raise ValueError(
            "Numbers of rows per group "
            "to be returned must be greater than 0."
        )

    indices = df.groupby(by=by, dropna=dropna, sort=False, observed=True)
    indices = indices[column]

    try:
        if ascending:
            indices = indices.nsmallest(n=k)
        else:
            indices = indices.nlargest(n=k)
    except TypeError:
        indices = indices.apply(
            lambda d: d.sort_values(ascending=ascending).head(k)
        )

    indices = indices.index.get_level_values(-1)
    if ignore_index:
        return df.loc[indices].reset_index(drop=True)
    return df.loc[indices]
