from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check, check_column, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", dest_col_name="dest_column_name")
def transform_column(
    df: pd.DataFrame,
    column_name: Hashable,
    function: Callable,
    dest_column_name: Optional[str] = None,
    elementwise: bool = True,
) -> pd.DataFrame:
    """Transform the given column using the provided function.

    Meant to be the method-chaining equivalent of:
    ```python
    df[dest_column_name] = df[column_name].apply(function)
    ```

    Functions can be applied in one of two ways:

    - **Element-wise** (default; `elementwise=True`). Then, the individual
    column elements will be passed in as the first argument of `function`.
    - **Column-wise** (`elementwise=False`). Then, `function` is expected to
    take in a pandas Series and return a sequence that is of identical length
    to the original.

    If `dest_column_name` is provided, then the transformation result is stored
    in that column. Otherwise, the transformed result is stored under the name
    of the original column.

    This method does not mutate the original DataFrame.

    Example: Transform a column in-place with an element-wise function.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": [2, 3, 4],
        ...     "b": ["area", "pyjanitor", "grapefruit"],
        ... })
        >>> df
           a           b
        0  2        area
        1  3   pyjanitor
        2  4  grapefruit
        >>> df.transform_column(
        ...     column_name="a",
        ...     function=lambda x: x**2 - 1,
        ... )
            a           b
        0   3        area
        1   8   pyjanitor
        2  15  grapefruit

    Example: Transform a column in-place with an column-wise function.

        >>> df.transform_column(
        ...     column_name="b",
        ...     function=lambda srs: srs.str[:5],
        ...     elementwise=False,
        ... )
           a      b
        0  2   area
        1  3  pyjan
        2  4  grape

    :param df: A pandas DataFrame.
    :param column_name: The column to transform.
    :param function: A function to apply on the column.
    :param dest_column_name: The column name to store the transformation result
        in. Defaults to None, which will result in the original column
        name being overwritten. If a name is provided here, then a new column
        with the transformed values will be created.
    :param elementwise: Whether to apply the function elementwise or not.
        If `elementwise` is True, then the function's first argument
        should be the data type of each datum in the column of data,
        and should return a transformed datum.
        If `elementwise` is False, then the function's should expect
        a pandas Series passed into it, and return a pandas Series.

    :returns: A pandas DataFrame with a transformed column.
    """
    check_column(df, column_name)

    if dest_column_name is None:
        dest_column_name = column_name

    if elementwise:
        result = df[column_name].apply(function)
    else:
        result = function(df[column_name])

    df = df.assign(**{dest_column_name: result})
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_names="new_column_names")
def transform_columns(
    df: pd.DataFrame,
    column_names: Union[List[str], Tuple[str]],
    function: Callable,
    suffix: Optional[str] = None,
    elementwise: bool = True,
    new_column_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Transform multiple columns through the same transformation.

    This method does not mutate the original DataFrame.

    Super syntactic sugar!
    Basically wraps [`transform_column`][janitor.functions.transform_columns.transform_column]
    and calls it repeatedly over all column names provided.

    User can optionally supply either a suffix to create a new set of columns
    with the specified suffix, or provide a dictionary mapping each original
    column name in `column_names` to its corresponding new column name.
    Note that all column names must be strings.

    Example: log10 transform a list of columns, replacing original columns.

        >>> import numpy as np
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "col1": [5, 10, 15],
        ...     "col2": [3, 6, 9],
        ...     "col3": [10, 100, 1_000],
        ... })
        >>> df
           col1  col2  col3
        0     5     3    10
        1    10     6   100
        2    15     9  1000
        >>> df.transform_columns(["col1", "col2", "col3"], np.log10)
               col1      col2  col3
        0  0.698970  0.477121   1.0
        1  1.000000  0.778151   2.0
        2  1.176091  0.954243   3.0

    Example: Using the `suffix` parameter to create new columns.

        >>> df.transform_columns(["col1", "col3"], np.log10, suffix="_log")
           col1  col2  col3  col1_log  col3_log
        0     5     3    10  0.698970       1.0
        1    10     6   100  1.000000       2.0
        2    15     9  1000  1.176091       3.0

    Example: Using the `new_column_names` parameter to create new columns.

        >>> df.transform_columns(
        ...     ["col1", "col3"],
        ...     np.log10,
        ...     new_column_names={"col1": "transform1"},
        ... )
           col1  col2  col3  transform1
        0     5     3   1.0    0.698970
        1    10     6   2.0    1.000000
        2    15     9   3.0    1.176091

    :param df: A pandas DataFrame.
    :param column_names: An iterable of columns to transform.
    :param function: A function to apply on each column.
    :param suffix: Suffix to use when creating new columns to hold
        the transformed values.
    :param elementwise: Passed on to `transform_column`; whether or not
        to apply the transformation function elementwise (True)
        or columnwise (False).
    :param new_column_names: An explicit mapping of old column names in
        `column_names` to new column names.
    :returns: A pandas DataFrame with transformed columns.
    :raises ValueError: If both `suffix` and `new_column_names` are
        specified.
    """  # noqa: E501
    check("column_names", column_names, [list, tuple])

    if suffix is not None and new_column_names is not None:
        raise ValueError(
            "Only one of `suffix` or `new_column_names` should be specified."
        )

    dest_column_names = dict(zip(column_names, column_names))

    if suffix:
        check("suffix", suffix, [str])
        for col in column_names:
            dest_column_names[col] = col + suffix
    elif new_column_names:
        check("new_column_names", new_column_names, [dict])
        dest_column_names = new_column_names

    for old_col, new_col in dest_column_names.items():
        df = transform_column(
            df,
            old_col,
            function,
            new_col,
            elementwise=elementwise,
        )

    return df
