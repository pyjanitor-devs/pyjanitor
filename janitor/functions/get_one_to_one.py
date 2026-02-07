"""Implementation of `get_one_to_one`."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _get_one_to_one_value_order(series: pd.Series) -> np.ndarray:
    """Convert series values to ordered integer codes."""
    codes, _ = pd.factorize(series, sort=False)
    return codes


def get_one_to_one(df: pd.DataFrame) -> List[List[str]]:
    """Find column groups that map one-to-one with each other.

    Columns are grouped if they share identical value orderings, indicating
    a one-to-one mapping across rows.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "Lab_Test_Long": ["Cholesterol, LDL", "Cholesterol, LDL", "Glucose"],
        ...         "Lab_Test_Short": ["CLDL", "CLDL", "GLUC"],
        ...         "LOINC": [12345, 12345, 54321],
        ...         "Person": ["Sam", "Bill", "Sam"],
        ...     }
        ... )
        >>> janitor.get_one_to_one(df)
        [['Lab_Test_Long', 'Lab_Test_Short', 'LOINC']]

    Args:
        df: A pandas DataFrame.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `df` has no columns or duplicate column names.

    Returns:
        A list of column name groups that map one-to-one with each other.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.shape[1] == 0:
        raise ValueError("df must have at least one column.")
    if df.columns.duplicated().any():
        raise ValueError("df must have unique column names.")

    encoded = {col: _get_one_to_one_value_order(df[col]) for col in df.columns}
    remaining_cols = list(df.columns)
    groups: List[List[str]] = []

    while remaining_cols:
        nm1 = remaining_cols.pop(0)
        current_group = [nm1]
        for nm2 in remaining_cols[:]:
            if np.array_equal(encoded[nm1], encoded[nm2]):
                current_group.append(nm2)
                remaining_cols.remove(nm2)
        if len(current_group) > 1:
            groups.append(current_group)

    return groups
