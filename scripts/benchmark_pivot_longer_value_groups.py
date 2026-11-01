"""Compare baseline vs homogeneous NumPy fast path for pyjanitor #1821.

Requires an installed pyjanitor checkout. No network or pytest is required.
Examples:
  python scripts/benchmark_pivot_longer_value_groups.py --profile smoke
  python scripts/benchmark_pivot_longer_value_groups.py --profile issue
  Use --repeat 2 and --out result.json to save repeated measurements.

Each variant runs in a fresh child process, avoiding cross-case lifetime
allocation effects when reporting peak RSS. Timing is measured with tracing
disabled across repeated calls; a separate call with tracemalloc measures
Python allocations. Native NumPy buffers are not included in traced memory.
The baseline is the same branch with only the new fast-path entry disabled
(not a separate upstream revision). Compare results only on the same machine.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd


def _api():
    from importlib import import_module

    from janitor import pivot_longer, pivot_longer_spec

    return pivot_longer, pivot_longer_spec, import_module("janitor.functions.pivot")


def _peak_rss_mib():
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        handle = get_process()
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        if get_memory(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024**2)
        return None

    import resource

    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return high / (1024 if sys.platform.startswith("linux") else 1024**2)


def _dataset(rows, groups, repeats):
    width = groups * repeats
    names = [f"v{g}_{part}" for g in range(groups) for part in range(repeats)]
    values = np.arange(rows * width, dtype=np.float64).reshape(rows, width)
    frame = pd.DataFrame(values, columns=names)
    spec = pd.DataFrame(
        {
            ".name": names,
            ".value": [f"v{g}" for g in range(groups) for _ in range(repeats)],
            "part": [part for _ in range(groups) for part in range(repeats)],
        }
    )
    return frame, spec


def _execute(frame, spec, case, mode, pivot_longer, pivot_longer_spec, module):
    original = module._fast_numpy_value_matrix
    try:
        if mode == "baseline":
            module._fast_numpy_value_matrix = lambda _: None
        if case == "value_only":
            return pivot_longer_spec(frame, spec[[".name", ".value"]])
        if case == "value_names":
            return pivot_longer_spec(frame, spec)
        if case == "names_transform":
            return pivot_longer(
                frame,
                names_to=(".value", "part"),
                names_pattern=r"(v[0-9]+)_([0-9]+)",
                names_transform={"part": "int64"},
            )
        raise ValueError(f"Unknown case: {case}")
    finally:
        module._fast_numpy_value_matrix = original


def _worker(args):
    pivot_longer, pivot_longer_spec, module = _api()
    frame, spec = _dataset(args.rows, args.groups, args.repeats)
    # NumPy buffers in frame are allocated before measurements for both modes.
    durations = []
    peak_start = _peak_rss_mib()
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = _execute(
            frame, spec, args.case, args.mode, pivot_longer, pivot_longer_spec, module
        )
        durations.append(time.perf_counter() - start)
        del result
    # Measure memory separately: tracing inflates legacy per-label timings.
    tracemalloc.start()
    memory_result = _execute(
        frame, spec, args.case, args.mode, pivot_longer, pivot_longer_spec, module
    )
    traced_peak = tracemalloc.get_traced_memory()[1] / (1024**2)
    tracemalloc.stop()
    del memory_result
    peak_after = _peak_rss_mib()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "rows": args.rows,
                "columns": args.groups * args.repeats,
                "case": args.case,
                "median_seconds": round(statistics.median(durations), 6),
                "peak_rss_mib": (
                    round(peak_after, 1) if peak_after is not None else None
                ),
                "baseline_process_rss_peak_mib": (
                    round(peak_start, 1) if peak_start is not None else None
                ),
                "peak_traced_python_mib": round(traced_peak, 1),
            }
        )
    )


def _config(profile):
    if profile == "smoke":
        return [("wide", 100, 80, 5), ("tall", 3000, 4, 5)]
    return [
        ("wide", 100, 4000, 5),
        ("medium", 10000, 400, 5),
        ("tall", 1000000, 4, 5),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "issue"], default="smoke")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--case", choices=["value_only", "value_names", "names_transform"]
    )
    parser.add_argument("--mode", choices=["baseline", "fast"])
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.worker:
        _worker(args)
        return

    records = []
    for name, rows, groups, repeats in _config(args.profile):
        for case in ("value_only", "value_names", "names_transform"):
            outcomes = {}
            for mode in ("baseline", "fast"):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--rows",
                    str(rows),
                    "--groups",
                    str(groups),
                    "--repeats",
                    str(repeats),
                    "--case",
                    case,
                    "--mode",
                    mode,
                    "--repeat",
                    str(args.repeat),
                ]
                completed = subprocess.run(
                    command, capture_output=True, text=True, check=True
                )
                outcomes[mode] = json.loads(completed.stdout)
            baseline, fast = outcomes["baseline"], outcomes["fast"]
            record = {
                "shape": name,
                "rows": rows,
                "columns": groups * repeats,
                "case": case,
                "baseline_seconds": baseline["median_seconds"],
                "fast_seconds": fast["median_seconds"],
                "speedup": round(
                    baseline["median_seconds"] / fast["median_seconds"], 2
                ),
                "baseline_peak_rss_mib": baseline["peak_rss_mib"],
                "fast_peak_rss_mib": fast["peak_rss_mib"],
                "baseline_peak_traced_python_mib": baseline["peak_traced_python_mib"],
                "fast_peak_traced_python_mib": fast["peak_traced_python_mib"],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    if args.out:
        args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
