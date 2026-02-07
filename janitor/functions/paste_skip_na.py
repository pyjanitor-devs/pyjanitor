"""Implementation of `paste_skip_na`."""

from __future__ import annotations

from typing import Any, List

import pandas as pd
from pandas.api.types import is_list_like, is_scalar


def _as_listlike(value: Any) -> List[Any]:
    """Normalize a value to a list-like container."""
    if is_list_like(value) and not is_scalar(value):
        return list(value)
    return [value]


def paste_skip_na(*args, sep: str = " ", collapse: str | None = None):
    """Paste values together while skipping missing entries.

    This mirrors R janitor's `paste_skip_na`, preserving missing values
    when all entries are missing for a position.

    Examples:
        >>> import janitor
        >>> janitor.paste_skip_na("A", None)
        'A'
        >>> janitor.paste_skip_na("A", None, ["B", None], sep=",")
        ['A,B', 'A']
        >>> janitor.paste_skip_na(None, None, None) is None
        True

    Args:
        *args: Values to paste. Scalars will be recycled to match vector lengths.
        sep: Separator used between values.
        collapse: If provided, collapse the final vector to a scalar string.

    Raises:
        ValueError: If arguments have incompatible lengths.

    Returns:
        A scalar string/None or a list of strings/None values.
    """
    force_list = any(is_list_like(arg) and not is_scalar(arg) for arg in args)
    return _paste_skip_na(args, sep=sep, collapse=collapse, force_list=force_list)


def _paste_skip_na(
    args: tuple,
    *,
    sep: str,
    collapse: str | None,
    force_list: bool,
):
    if len(args) == 0:
        return ""

    if len(args) == 1:
        values = _as_listlike(args[0])
        if collapse is not None:
            if all(pd.isna(value) for value in values):
                return values[0] if values else None
            non_missing = [str(value) for value in values if not pd.isna(value)]
            return collapse.join(non_missing)

        if force_list:
            return [value if pd.isna(value) else str(value) for value in values]
        return values[0] if pd.isna(values[0]) else str(values[0])

    a1 = _as_listlike(args[0])
    a2 = _as_listlike(args[1])

    if len(a1) != len(a2):
        if len(a1) == 1:
            a1 = a1 * len(a2)
        elif len(a2) == 1:
            a2 = a2 * len(a1)
        else:
            raise ValueError(
                "Arguments must be the same length or one argument must be a scalar."
            )

    first_two = []
    for value1, value2 in zip(a1, a2):
        missing1 = pd.isna(value1)
        missing2 = pd.isna(value2)
        if not missing1 and not missing2:
            first_two.append(f"{value1}{sep}{value2}")
        elif missing1 and not missing2:
            first_two.append(str(value2))
        elif not missing1 and missing2:
            first_two.append(str(value1))
        else:
            first_two.append(value1)

    new_args = (first_two, *args[2:])
    return _paste_skip_na(
        new_args,
        sep=sep,
        collapse=collapse,
        force_list=force_list,
    )
