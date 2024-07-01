"""Alternative function to pd.agg for summarizing data."""

from typing import Any

import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def summarise(
    df: pd.DataFrame,
    aggfuncs: list,
    by: Any = None,
) -> pd.DataFrame:
    """

    !!! info "New in version 0.25.0"

    !!!note

        Before reaching for `summarize`, try `pd.DataFrame.agg`.

    `summarise` is a reduction operation that returns one row
    for each combination of grouping variables.
    It will contain one column for each grouping variable
    and one column for each of the summary statistics
    that is specified.

    If a dictionary is passed, the key of the dictionary
    will be the name of the aggregated column. If the value
    of the dictionary is a tuple, it should be of the form
    `(column_name, aggfunc)`, where `column_name` is the
    name of an existing column in the DataFrame, while
    `aggfunc` is the function to be applied. If the value of
    the dictionary is not a tuple, then it is treated as
    a callable, or a valid pandas aggregation function.

    If a tuple is passed, it should be of the form
    `(columns, aggfunc, names)`.  `columns` can be
    a single column, or multiple columns. column selection
    is possible with the
    [`select`][janitor.functions.select.select] syntax.
    `aggfunc` can be a single function, or a list of
    aggregation functions that is supported by Pandas.
    `names` is optional; it allows for setting the names
    of the aggregated columns. It can be a list of names,
    which should match the number of aggregated columns.
    It can also be a string, which can serve as the name
    of the new column. A single string can also be used
    as a glue to rename the aggregated columns,
    with `_col` as a placeholder for column name,
    and `_fn` as a placeholder for function name.

    It can also be a callabe, which will be called directly
    on the dataframe, or the grouped object.

    `by` can be anything supported in the pandas'
    [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby)
    `by` parameter.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor as jn
        >>> from janitor import col
        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'avg_swim': [2, 1, 2, 2, 3, 4],
        ...         'combine_id': [100200, 100200,
        ...                        101200, 101200,
        ...                        102201, 103202],
        ...         'category': ['heats', 'heats',
        ...                      'finals', 'finals',
        ...                      'heats', 'finals']}
        >>> df = pd.DataFrame(data)
        >>> arg = col("avg_run").compute("mean")
        >>> df.summarize(arg, by=['combine_id', 'category'])
                                avg_run
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

        Summarize with a new column name:

        >>> arg = col("avg_run").compute("mean").rename("avg_run_2")
        >>> df.summarize(arg)
            avg_run_2
        0   2.833333
        >>> df.summarize(arg, by=['combine_id', 'category'])
                            avg_run_2
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

        Summarize with the placeholders when renaming:

        >>> cols = col("avg*").compute("mean").rename("{_col}_{_fn}")
        >>> df.summarize(cols)
            avg_jump_mean  avg_run_mean  avg_swim_mean
        0       2.833333      2.833333       2.333333
        >>> df.summarize(cols, by=['combine_id', 'category'])
                                avg_jump_mean  avg_run_mean  avg_swim_mean
        combine_id category
        100200     heats               3.5           3.5            1.5
        101200     finals              1.5           2.0            2.0
        102201     heats               3.0           2.0            3.0
        103202     finals              4.0           4.0            4.0

        Pass the `col` class to `by`:

        >>> df.summarize(cols, by=col("c*"))
                                avg_jump_mean  avg_run_mean  avg_swim_mean
        combine_id category
        100200     heats               3.5           3.5            1.5
        101200     finals              1.5           2.0            2.0
        102201     heats               3.0           2.0            3.0
        103202     finals              4.0           4.0            4.0

    Args:
        df: A pandas DataFrame.
        args: instance(s) of the `janitor.col` class.
        by: Column(s) to group by.

    Raises:
        ValueError: If a function is not provided.

    Returns:
        A pandas DataFrame with summarized columns.
    """  # noqa: E501

    return df
