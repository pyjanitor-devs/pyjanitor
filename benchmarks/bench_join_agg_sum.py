"""Benchmark for the integer prefix-sum `join_agg` kernels (Issue #1648).

Reproduces the shape of the issue's local numbers:

- Kernel-level: the old Rust `compute_sum_start*`/`compute_sum_end*`/
  `compute_sum_start_end*` kernels (O(sum of interval widths)) versus the
  new NumPy prefix-sum kernels in `_agg_functions` (O(n + m)), for
  "suffixes" (`<`), "prefixes" (`>`), and "arbitrary intervals" (a range
  join with no equality condition).
- End-to-end: a real `join_agg(..., aggfunc=[("value", "sum")])` call,
  which now dispatches through the new prefix-sum kernels automatically.

Run with:

    pixi run python benchmarks/bench_join_agg_sum.py
    pixi run python benchmarks/bench_join_agg_sum.py --large  # 10M/50M rows

Large-N results (Apple Silicon, single run, int64), kernel-level only --
the old Rust kernels are the O(sum of interval widths) algorithm this issue
replaces, so they are not re-run here; they are already intractable at the
20,000-row scale (see the small-N results above)::

    n = 10,000,000
    kernel suffixes (prefix-sum):     0.066 s
    kernel prefixes (prefix-sum):     0.056 s
    kernel intervals (prefix-sum):    0.098 s
    end-to-end join_agg sum '<':      2.027 s   (peak RSS ~1.6 GB)

    n = 50,000,000
    kernel suffixes (prefix-sum):     0.419 s
    kernel prefixes (prefix-sum):     0.351 s
    kernel intervals (prefix-sum):    0.652 s
    end-to-end join_agg sum '<':     18.795 s   (peak RSS ~6.1 GB)

The prefix-sum kernel itself stays linear and sub-second through 50M rows.
The end-to-end number is dominated by `conditional_join`'s match-finding
(building `starts`/`right_index`), which is unrelated to this issue's
scope (the sum kernel) and unchanged by this PR.
"""

import timeit

import janitor_rs
import numpy as np
import pandas as pd

import janitor  # noqa: F401 registers the DataFrame accessor
from janitor.functions._conditional_join import _agg_functions as agg


def _bench(label: str, fn, number: int = 5) -> float:
    """Time `fn`, print the per-call average, and return it in seconds."""
    seconds = timeit.timeit(fn, number=number) / number
    print(f"{label:<28} {seconds * 1000:10.3f} ms")
    return seconds


def kernel_level(n: int) -> None:
    """Compare the old Rust kernels against the new prefix-sum kernels."""
    print(f"\n-- kernel-level, n={n:,} (int64) --")
    rng = np.random.default_rng(0)
    arr = rng.integers(-1000, 1000, size=n, dtype="int64")
    booleans = np.zeros(n, dtype=bool)

    # suffixes: every row starts somewhere and runs to the end
    starts = rng.integers(0, n, size=n, dtype="int64")
    _bench(
        "suffixes (rust)",
        lambda: janitor_rs.compute_sum_start_int64(
            arr=arr, starts=starts, booleans=booleans
        ),
    )
    _bench(
        "suffixes (prefix-sum)",
        lambda: agg._sum_starts(arr=arr, starts=starts, booleans=booleans),
    )

    # prefixes: every row runs from the start to somewhere
    ends = rng.integers(0, n, size=n, dtype="int64")
    _bench(
        "prefixes (rust)",
        lambda: janitor_rs.compute_sum_end_int64(arr=arr, ends=ends, booleans=booleans),
    )
    _bench(
        "prefixes (prefix-sum)",
        lambda: agg._sum_ends(arr=arr, ends=ends, booleans=booleans),
    )

    # arbitrary intervals
    lo = rng.integers(0, n, size=n, dtype="int64")
    hi = rng.integers(0, n, size=n, dtype="int64")
    starts2, ends2 = np.minimum(lo, hi), np.maximum(lo, hi)
    _bench(
        "intervals (rust)",
        lambda: janitor_rs.compute_sum_start_end_int64(
            arr=arr, starts=starts2, ends=ends2, booleans=booleans
        ),
    )
    _bench(
        "intervals (prefix-sum)",
        lambda: agg._sum_starts_ends(
            arr=arr, starts=starts2, ends=ends2, booleans=booleans
        ),
    )


def end_to_end(n: int) -> None:
    """Time a real `join_agg(..., aggfunc=[("value", "sum")])` call."""
    print(f"\n-- end-to-end join_agg, n={n:,} (int64, '<') --")
    rng = np.random.default_rng(0)
    left = pd.DataFrame({"key": rng.integers(0, n, size=n)})
    right = pd.DataFrame(
        {
            "key": np.sort(rng.integers(0, n, size=n)),
            "value": rng.integers(-1000, 1000, size=n),
        }
    )

    def run():
        """Run one `join_agg` sum call."""
        return left.join_agg(right, ("key", "key", "<"), aggfunc=[("value", "sum")])

    _bench("join_agg sum '<'", run, number=3)


def large_scale(n: int) -> None:
    """Prefix-sum kernel only (the old Rust kernels are intractable here)."""
    print(f"\n-- kernel-level, n={n:,} (int64), prefix-sum only --")
    rng = np.random.default_rng(0)
    arr = rng.integers(-1000, 1000, size=n, dtype="int64")
    booleans = np.zeros(n, dtype=bool)
    starts = rng.integers(0, n, size=n, dtype="int64")
    ends = rng.integers(0, n, size=n, dtype="int64")
    lo = rng.integers(0, n, size=n, dtype="int64")
    hi = rng.integers(0, n, size=n, dtype="int64")
    starts2, ends2 = np.minimum(lo, hi), np.maximum(lo, hi)

    _bench(
        "suffixes (prefix-sum)",
        lambda: agg._sum_starts(arr=arr, starts=starts, booleans=booleans),
        number=3,
    )
    _bench(
        "prefixes (prefix-sum)",
        lambda: agg._sum_ends(arr=arr, ends=ends, booleans=booleans),
        number=3,
    )
    _bench(
        "intervals (prefix-sum)",
        lambda: agg._sum_starts_ends(
            arr=arr, starts=starts2, ends=ends2, booleans=booleans
        ),
        number=3,
    )
    end_to_end(n)


if __name__ == "__main__":
    import sys

    if "--large" in sys.argv:
        for n in (10_000_000, 50_000_000):
            large_scale(n)
    else:
        for n in (1_000, 20_000):
            kernel_level(n)
        for n in (1_000, 20_000):
            end_to_end(n)
