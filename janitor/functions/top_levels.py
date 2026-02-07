"""Implementation of `top_levels`."""

from __future__ import annotations

from numbers import Real
from typing import Dict, List

import pandas as pd


def _get_level_groups(
    levels: List[str],
    n: int,
    num_levels: int,
) -> Dict[str, str | None]:
    """Return grouped level labels for top, middle, and bottom categories."""
    top_levels = ", ".join(map(str, levels[:n]))
    bottom_levels = ", ".join(map(str, levels[num_levels - n : num_levels]))

    if num_levels > 2 * n:
        middle_levels = ", ".join(map(str, levels[n : num_levels - n]))
    else:
        middle_levels = None

    if middle_levels is not None and len(middle_levels) > 30:
        middle_levels = f"<<< Middle Group ({num_levels - 2 * n} categories) >>>"
    if len(top_levels) > 30:
        top_levels = f"{top_levels[:27]}..."
    if len(bottom_levels) > 30:
        bottom_levels = f"{bottom_levels[:27]}..."

    return {"top": top_levels, "mid": middle_levels, "bot": bottom_levels}


def top_levels(
    input_vec: pd.Series | pd.Categorical,
    n: int = 2,
    show_na: bool = False,
) -> pd.DataFrame:
    """Generate a grouped frequency table for a categorical variable.

    Groups levels into top-n, middle, and bottom-n categories by level order.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> s = pd.Categorical(["A", "B", "C", "D", "E", "A", "B", "C"])
        >>> janitor.top_levels(s, n=2)
          level  n  percent
        0  A, B  4     0.50
        1     C  2     0.25
        2  D, E  2     0.25

    Args:
        input_vec: A pandas Categorical or Series with categorical dtype.
        n: Number of levels in each of the top and bottom groups.
        show_na: Whether to include NA values in the output.

    Raises:
        ValueError: If input is not categorical or `n` is invalid.

    Returns:
        A DataFrame with grouped counts and percentages.
    """
    if isinstance(input_vec, pd.Series):
        series = input_vec
    elif isinstance(input_vec, pd.Categorical):
        series = pd.Series(input_vec)
    elif isinstance(input_vec, pd.Index):
        series = pd.Series(input_vec)
    else:
        raise ValueError("factor_vec is not of type 'factor'")

    if not isinstance(series.dtype, pd.CategoricalDtype):
        raise ValueError("factor_vec is not of type 'factor'")

    num_levels = len(series.cat.categories)

    if num_levels <= 2:
        raise ValueError("input factor variable must have at least 3 levels")

    if not isinstance(n, Real) or n < 1 or n % 1 != 0:
        raise ValueError("n must be a whole number at least 1")

    n = int(n)
    if num_levels < 2 * n:
        raise ValueError(
            "there are "
            f"{num_levels} levels in the variable and {n} levels in each of the "
            "top and bottom groups.\nSince 2 * "
            f"{n} = {2 * n} is greater than {num_levels}, "
            "there would be overlap in the top and bottom groups and some "
            "records will be double-counted."
        )
    var_name = series.name or "level"
    levels = list(series.cat.categories)
    groups = _get_level_groups(levels, n, num_levels)

    codes = series.cat.codes
    grouped = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = codes != -1

    grouped.loc[valid & (codes < n)] = groups["top"]
    grouped.loc[valid & (codes >= num_levels - n)] = groups["bot"]

    if groups["mid"] is not None:
        middle_mask = valid & (codes >= n) & (codes < num_levels - n)
        grouped.loc[middle_mask] = groups["mid"]

    if groups["mid"] is None:
        categories = [groups["top"], groups["bot"]]
    else:
        categories = [groups["top"], groups["mid"], groups["bot"]]

    grouped = pd.Categorical(grouped, categories=categories, ordered=True)

    from .tabyl import _tabyl_one

    result = _tabyl_one(
        pd.DataFrame({var_name: grouped}),
        var_name,
        show_na=show_na,
        show_missing_levels=True,
    )

    if show_na and result[var_name].isna().any():
        non_na = result[~result[var_name].isna()]
        na_rows = result[result[var_name].isna()]
        result = pd.concat([non_na, na_rows], ignore_index=True)

    return result
