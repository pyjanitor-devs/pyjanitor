"""Compare normal_cdf with its pre-#1468 implementation on the same machine.

Run from an installed checkout with:
    pixi run python scripts/benchmark_normal_cdf.py

Inputs and imports are prepared before timing. Both variants include Series
construction and are warmed up. Report median per-call times over seven
alternating rounds; timings are evidence, not a CI performance assertion.
"""

import platform
from statistics import median
from timeit import Timer

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from pandas.testing import assert_series_equal

from janitor.math import normal_cdf


def baseline(s):
    """Run the original normal_cdf implementation, including its local imports."""
    import pandas as pd
    import scipy

    return pd.Series(scipy.stats.norm.cdf(s), index=s.index)


def main():
    """Check equality and measure both implementations for each input size."""
    print(platform.platform())
    print(platform.processor())
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"pandas {pd.__version__}, SciPy {scipy.__version__}"
    )
    print("dtype, size, calls/round, baseline us, ndtr us, speedup")
    rng = np.random.default_rng(1468)
    for dtype in ("float32", "float64"):
        for size in (0, 10, 1_000, 100_000, 1_000_000):
            s = pd.Series(rng.standard_normal(size).astype(dtype), name="values")
            assert_series_equal(normal_cdf(s), baseline(s), check_exact=True)
            number = max(1, min(5_000, 1_000_000 // max(size, 1)))
            timers = [Timer(lambda: baseline(s)), Timer(lambda: normal_cdf(s))]
            samples = [[], []]
            for round_number in range(7):
                for variant in (0, 1) if round_number % 2 == 0 else (1, 0):
                    samples[variant].append(timers[variant].timeit(number) / number)
            old, new = (median(times) for times in samples)
            print(
                f"{dtype}, {size}, {number}, {old * 1e6:.2f}, "
                f"{new * 1e6:.2f}, {old / new:.2f}x"
            )


if __name__ == "__main__":
    main()
