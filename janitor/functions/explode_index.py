"""Implementation of the `explode_index` function."""

from __future__ import annotations

import re
from typing import Union

import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


@pf.register_dataframe_method
def explode_index(
    df: pd.DataFrame,
    names_sep: Union[str, None] = None,
    names_pattern: Union[str, None] = None,
    axis: str = "columns",
    level_names: list = None,
) -> pd.DataFrame:
    """Explode a single index DataFrame into a MultiIndex DataFrame.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...          {'max_speed_mean': [267.3333333333333, 50.5],
        ...           'max_speed_median': [389.0, 50.5]})
        >>> df
           max_speed_mean  max_speed_median
        0      267.333333             389.0
        1       50.500000              50.5
        >>> df.explode_index(names_sep='_',axis='columns')  # doctest: +NORMALIZE_WHITESPACE
                  max
                speed
                 mean median
        0  267.333333  389.0
        1   50.500000   50.5
        >>> df.explode_index(names_pattern=r"(.+speed)_(.+)",axis='columns') # doctest: +NORMALIZE_WHITESPACE
            max_speed
                 mean median
        0  267.333333  389.0
        1   50.500000   50.5
        >>> df.explode_index(
        ...     names_pattern=r"(?P<measurement>.+speed)_(?P<aggregation>.+)",
        ...     axis='columns'
        ... ) # doctest: +NORMALIZE_WHITESPACE
        measurement   max_speed
        aggregation        mean median
        0            267.333333  389.0
        1             50.500000   50.5
        >>> df.explode_index(
        ...     names_sep='_',
        ...     axis='columns',
        ...     level_names = ['min or max', 'measurement','aggregation']
        ... ) # doctest: +NORMALIZE_WHITESPACE
        min or max          max
        measurement       speed
        aggregation        mean median
        0            267.333333  389.0
        1             50.500000   50.5

    Args:
        df: A pandas DataFrame.
        names_sep: string or compiled regex used to split the column/index into levels.
        names_pattern: regex to extract new levels from the column/index.
        axis: 'index/columns'. Determines which axis to explode.
        level_names: names of the levels in the MultiIndex.

    Returns:
        A pandas DataFrame with a MultiIndex.
    """  # noqa: E501
    check("axis", axis, [str])
    if axis not in {"index", "columns"}:
        raise ValueError("axis should be either index or columns.")
    if (names_sep is None) and (names_pattern is None):
        raise ValueError(
            "Provide argument for either names_sep or names_pattern."
        )
    if (names_sep is not None) and (names_pattern is not None):
        raise ValueError(
            "Provide argument for either names_sep or names_pattern, not both."
        )
    if names_sep is not None:
        check("names_sep", names_sep, [str])
    if names_pattern is not None:
        check("names_pattern", names_pattern, [str])
    if level_names is not None:
        check("level_names", level_names, [list])

    new_index = getattr(df, axis)
    if isinstance(new_index, pd.MultiIndex):
        return df
    # avoid a copy - Index is immutable; a slice is safe to use.
    df = df[:]
    if names_sep:
        new_index = new_index.str.split(names_sep, expand=True)
    else:
        named_groups = re.compile(names_pattern).groupindex
        if named_groups and not level_names:
            level_names = list(named_groups)
        new_index = new_index.str.extract(names_pattern)
        new_index = [arr.array for _, arr in new_index.items()]
        new_index = pd.MultiIndex.from_arrays(new_index)
    if level_names:
        new_index.names = level_names

    setattr(df, axis, new_index)
    return df
