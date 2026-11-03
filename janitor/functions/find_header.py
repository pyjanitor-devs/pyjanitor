"""Implementation of find_header."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np
import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


@pf.register_dataframe_method
def find_header(
    df: pd.DataFrame,
    target: Any = None,
    column: Hashable | int | None = None,
    **kwargs: Any,
) -> int:
    """Find the row index of a header in a DataFrame.

    Useful for spreadsheets where titles or notes precede the real table headers.
    When called without arguments, locates the first row without missing values.
    Alternatively searches for a specific string or value in a given column.

    Examples:
        Find first complete row:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     [
        ...         ["Title", np.nan],
        ...         ["ID", "Name"],
        ...         [1, "Alice"],
        ...     ]
        ... )
        >>> df.find_header()
        1

        Search for a label in column zero:

        >>> df.find_header("ID")
        1

    Args:
        df: Input pandas DataFrame.
        target: Optional value to search for in a column.
        column: Column position or column label to search.
            Defaults to column zero when target is provided.
        **kwargs: Optional keyword argument mapping target value
            to column position. Matches R janitor syntax.

    Raises:
        ValueError: If DataFrame is empty, if no complete row
            exists, or if target is not found.

    Returns:
        Integer position of the detected header row.
    """
    check("df", df, [pd.DataFrame])

    if df.empty:
        raise ValueError("Cannot find header in an empty DataFrame.")

    if kwargs and target is None:
        target, column = next(iter(kwargs.items()))

    if target is None:
        complete_mask = df.notna().all(axis=1)
        if not complete_mask.any():
            raise ValueError("No complete row found without missing values.")
        return int(np.flatnonzero(complete_mask.to_numpy())[0])

    if column is None:
        series = df.iloc[:, 0]
        col_desc = "column 0"
    elif column in df.columns:
        series = df[column]
        col_desc = f"column '{column}'"
    elif isinstance(column, int) and 0 <= column < df.shape[1]:
        series = df.iloc[:, column]
        col_desc = f"column position {column}"
    else:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    match_mask = series == target
    if not match_mask.any():
        raise ValueError(f"Target value '{target}' was not found in {col_desc}.")

    return int(np.flatnonzero(match_mask.to_numpy())[0])
