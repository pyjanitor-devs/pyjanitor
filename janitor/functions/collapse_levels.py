"""Implementation of the `collapse_levels` function."""

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_string_dtype

from janitor.utils import check


@pf.register_dataframe_method
def collapse_levels(
    df: pd.DataFrame,
    sep: str = None,
    glue: str = None,
    axis: str = "columns",
) -> pd.DataFrame:
    """Flatten multi-level index/column dataframe to a single level.

    This method does not mutate the original DataFrame.

    Given a DataFrame containing multi-level index/columns, flatten to single-level
    by string-joining the labels in each level.

    After a `groupby` / `aggregate` operation where `.agg()` is passed a
    list of multiple aggregation functions, a multi-level DataFrame is
    returned with the name of the function applied in the second level.

    It is sometimes convenient for later indexing to flatten out this
    multi-level configuration back into a single level. This function does
    this through a simple string-joining of all the names across different
    levels in a single column.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "class": ["bird", "bird", "bird", "mammal", "mammal"],
        ...     "max_speed": [389, 389, 24, 80, 21],
        ...     "type": ["falcon", "falcon", "parrot", "Lion", "Monkey"],
        ... })
        >>> df
            class  max_speed    type
        0    bird        389  falcon
        1    bird        389  falcon
        2    bird         24  parrot
        3  mammal         80    Lion
        4  mammal         21  Monkey
        >>> grouped_df = df.groupby("class")[['max_speed']].agg(["mean", "median"])
        >>> grouped_df  # doctest: +NORMALIZE_WHITESPACE
                 max_speed
                      mean median
        class
        bird    267.333333  389.0
        mammal   50.500000   50.5
        >>> grouped_df.collapse_levels(sep="_")  # doctest: +NORMALIZE_WHITESPACE
                max_speed_mean  max_speed_median
        class
        bird        267.333333             389.0
        mammal       50.500000              50.5

        Before applying `.collapse_levels`, the `.agg` operation returns a
        multi-level column DataFrame whose columns are `(level 1, level 2)`:

        ```python
        [("max_speed", "mean"), ("max_speed", "median")]
        ```

        `.collapse_levels` then flattens the column MultiIndex into a single
        level index with names:

        ```python
        ["max_speed_mean", "max_speed_median"]
        ```

        For more control, a `glue` specification can be passed,
        where the names of the levels are used to control the output of the
        flattened index:
        >>> (grouped_df
        ...  .rename_axis(columns=['column_name', 'agg_name'])
        ...  .collapse_levels(glue="{agg_name}_{column_name}")
        ... )
                mean_max_speed  median_max_speed
        class
        bird        267.333333             389.0
        mammal       50.500000              50.5

        Note that for `glue` to work, the keyword arguments
        in the glue specification
        should be the names of the levels in the MultiIndex.

    !!! abstract "Version Changed"

        - 0.27.0
            - Added `glue` and `axis` parameters.

    Args:
        df: A pandas DataFrame.
        sep: String separator used to join the column level names.
        glue: A specification on how the column levels should be combined.
            It allows for a more granular composition,
            and serves as an alternative to `sep`.
        axis: Determines whether to collapse the
            levels on the index or columns.

    Returns:
        A pandas DataFrame with single-level column index.
    """  # noqa: E501
    if (sep is not None) and (glue is not None):
        raise ValueError("Only one of sep or glue should be provided.")
    if sep is not None:
        check("sep", sep, [str])
    if glue is not None:
        check("glue", glue, [str])
    check("axis", axis, [str])
    if axis not in {"index", "columns"}:
        raise ValueError(
            "axis argument should be either 'index' or 'columns'."
        )

    if not isinstance(getattr(df, axis), pd.MultiIndex):
        return df

    # TODO: Pyarrow offers faster string computations
    # future work should take this into consideration,
    # which would require a different route from python's string.join
    # since work is only on the columns
    # it is safe, and more efficient to slice/view the dataframe
    # plus Pandas creates a new Index altogether
    # as such, the original dataframe is not modified
    df = df[:]
    new_index = getattr(df, axis)
    if glue is not None:
        new_index = [dict(zip(new_index.names, entry)) for entry in new_index]
        new_index = [glue.format_map(mapping) for mapping in new_index]
        setattr(df, axis, new_index)
        return df
    sep = "_" if sep is None else sep
    levels = [level for level in new_index.levels]
    all_strings = all(map(is_string_dtype, levels))
    if all_strings:
        no_empty_string = all((entry != "").all() for entry in levels)
        if no_empty_string:
            new_index = new_index.map(sep.join)
            setattr(df, axis, new_index)
            return df
    new_index = (map(str, entry) for entry in new_index)
    new_index = [
        # faster to use a list comprehension within string.join
        # compared to a generator
        # https://stackoverflow.com/a/37782238
        sep.join([entry for entry in word if entry])
        for word in new_index
    ]
    setattr(df, axis, new_index)
    return df
