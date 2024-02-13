"""Implementation of the `sort_naturally` function."""

from typing import Any

import pandas as pd
import pandas_flavor as pf
from natsort import index_natsorted


@pf.register_dataframe_method
def sort_naturally(
    df: pd.DataFrame, column_name: str, **natsorted_kwargs: Any
) -> pd.DataFrame:
    """Sort a DataFrame by a column using *natural* sorting.

    Natural sorting is distinct from
    the default lexiographical sorting provided by `pandas`.
    For example, given the following list of items:

    ```python
    ["A1", "A11", "A3", "A2", "A10"]
    ```

    Lexicographical sorting would give us:

    ```python
    ["A1", "A10", "A11", "A2", "A3"]
    ```

    By contrast, "natural" sorting would give us:

    ```python
    ["A1", "A2", "A3", "A10", "A11"]
    ```

    This function thus provides *natural* sorting
    on a single column of a dataframe.

    To accomplish this, we do a natural sort
    on the unique values that are present in the dataframe.
    Then, we reconstitute the entire dataframe
    in the naturally sorted order.

    Natural sorting is provided by the Python package
    [natsort](https://natsort.readthedocs.io/en/master/index.html).

    All keyword arguments to `natsort` should be provided
    after the column name to sort by is provided.
    They are passed through to the `natsorted` function.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "Well": ["A21", "A3", "A21", "B2", "B51", "B12"],
        ...         "Value": [1, 2, 13, 3, 4, 7],
        ...     }
        ... )
        >>> df
          Well  Value
        0  A21      1
        1   A3      2
        2  A21     13
        3   B2      3
        4  B51      4
        5  B12      7
        >>> df.sort_naturally("Well")
          Well  Value
        1   A3      2
        0  A21      1
        2  A21     13
        3   B2      3
        5  B12      7
        4  B51      4

    Args:
        df: A pandas DataFrame.
        column_name: The column on which natural sorting should take place.
        **natsorted_kwargs: Keyword arguments to be passed
            to natsort's `natsorted` function.

    Returns:
        A sorted pandas DataFrame.
    """
    new_order = index_natsorted(df[column_name], **natsorted_kwargs)
    return df.iloc[new_order, :]
