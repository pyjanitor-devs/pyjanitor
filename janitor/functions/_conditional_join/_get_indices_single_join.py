# helper function for a single join

import pandas as pd

from janitor.functions._conditional_join._greater_than_indices import (
    _greater_than_indices,
)
from janitor.functions._conditional_join._helpers import (
    greater_than_join_types,
    less_than_join_types,
)
from janitor.functions._conditional_join._less_than_indices import (
    _less_than_indices,
)
from janitor.functions._conditional_join._not_equal_indices import (
    _not_equal_indices,
)


def _single_join(
    df: pd.DataFrame,
    right: pd.DataFrame,
    condition: tuple,
    keep: str,
) -> dict | None:
    """
    Compute indices for a single join
    """
    left_on, right_on, op = condition
    if op in less_than_join_types:
        return _less_than_indices(
            left=df[left_on],
            right=right[right_on],
            strict=op == "<",
            keep=keep,
        )
    if op in greater_than_join_types:
        return _greater_than_indices(
            left=df[left_on],
            right=right[right_on],
            strict=op == ">",
            keep=keep,
        )
    if op == "!=":
        return _not_equal_indices(
            left=df[left_on],
            right=right[right_on],
            keep=keep,
        )
