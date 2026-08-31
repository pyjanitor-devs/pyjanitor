"""Shared dtype -> janitor_rs function dispatch for conditional_join."""

from functools import lru_cache
from typing import Callable

import janitor_rs

# Maps a numpy dtype name to the suffix used in the corresponding
# `janitor_rs` function name (float64/float32 shorten to f64/f32; every
# other dtype keeps its numpy name).
_DTYPE_SUFFIXES = {
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "int8": "int8",
    "uint64": "uint64",
    "uint32": "uint32",
    "uint16": "uint16",
    "uint8": "uint8",
    "float64": "f64",
    "float32": "f32",
}


@lru_cache(maxsize=None)
def _rs_func(family: str, dtype_name: str) -> Callable:
    """
    ELI5: for a given operation `family` (e.g. "compute_sum_start"), look up
    the `janitor_rs` function built for `dtype_name` once, then remember it
    so later calls for the same (family, dtype) pair skip the lookup.
    """
    try:
        suffix = _DTYPE_SUFFIXES[dtype_name]
    except KeyError:
        raise KeyError(f"Unsupported data type -> {dtype_name}") from None
    return getattr(janitor_rs, f"{family}_{suffix}")
