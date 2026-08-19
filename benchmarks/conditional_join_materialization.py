"""Benchmark conditional_join result materialization.

ELI5: the join first finds matching row numbers, then turns those numbers into
the returned DataFrame. This script can time either that second step alone or
the complete public operation. It also records Python/NumPy peak allocations
with tracemalloc so commits can be compared using the same generated inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc

import numpy as np
import pandas as pd

from janitor.functions.conditional_join import _create_frame, get_join_indices


def _frame(prefix: str, key_name: str, keys: np.ndarray, width: int) -> pd.DataFrame:
    """Create a repeatable mix of NumPy and pandas extension dtypes."""
    size = keys.size
    data = {
        key_name: keys,
        f"{prefix}_guard": np.ones(size) if prefix == "left" else np.zeros(size),
        f"{prefix}_int": np.arange(size),
        f"{prefix}_nullable_int": pd.array(np.arange(size), dtype="Int64"),
        f"{prefix}_boolean": pd.array(np.arange(size) % 2 == 0, dtype="boolean"),
        f"{prefix}_category": pd.Categorical(np.arange(size) % 4, categories=range(4)),
        f"{prefix}_string": pd.array(np.arange(size).astype(str), dtype="string"),
        f"{prefix}_datetime": pd.date_range(
            "2000-01-01", periods=size, freq="s", tz="UTC"
        ),
        f"{prefix}_timedelta": pd.to_timedelta(np.arange(size), unit="s"),
    }
    while len(data) < width:
        number = len(data)
        data[f"{prefix}_float_{number}"] = np.arange(size, dtype=float)
    return pd.DataFrame(data)


def _inputs(rows: int, width: int, density: str):
    """Build inputs with duplicate keys so every keep mode does real work."""
    left_keys = np.arange(rows) // 2
    right_keys = left_keys.copy()
    matched = {
        "zero": 0,
        "sparse": max(1, rows // 100),
        "dense": rows * 9 // 10,
        "full": rows,
    }[density]
    right_keys[matched:] += rows
    return _frame("left", "left_key", left_keys, width), _frame(
        "right", "right_key", right_keys, width
    )


def _measure(operation, repeats: int):
    """Return individual runtimes and peak allocated bytes."""
    operation()
    elapsed = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = operation()
        elapsed.append(time.perf_counter() - start)
        del result
    peaks = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(peak)
        tracemalloc.stop()
        del result
    return elapsed, peaks


def main() -> None:
    """Parse options, execute one benchmark case, and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["materialize", "end-to-end"], default="materialize"
    )
    parser.add_argument(
        "--how",
        choices=[
            "inner",
            "left",
            "right",
            "outer",
            "left_anti",
            "right_anti",
        ],
        default="outer",
    )
    parser.add_argument(
        "--density",
        choices=["zero", "sparse", "dense", "full"],
        default="sparse",
    )
    parser.add_argument("--keep", choices=["first", "last", "all"], default="all")
    args = parser.parse_args()
    if args.width < 9:
        parser.error("--width must be at least 9 for the mixed-dtype schema")

    left, right = _inputs(args.rows, args.width, args.density)
    conditions = (
        ("left_key", "right_key", "=="),
        ("left_guard", "right_guard", ">"),
    )
    if args.mode == "end-to-end":

        def operation():
            return left.conditional_join(
                right,
                *conditions,
                how=args.how,
                keep=args.keep,
                indicator=True,
            )

    else:
        match_keep = "all" if args.how in {"left_anti", "right_anti"} else args.keep
        indices = get_join_indices(left, right, *conditions, keep=match_keep)

        def operation():
            return _create_frame(
                left,
                right,
                indices["left_index"],
                indices["right_index"],
                args.how,
                slice(None),
                slice(None),
                True,
                False,
            )

    elapsed, peaks = _measure(operation, args.repeats)
    print(
        json.dumps(
            {
                **vars(args),
                "seconds_median": statistics.median(elapsed),
                "seconds_min": min(elapsed),
                "peak_bytes_max": max(peaks),
                "memory_method": "tracemalloc",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
