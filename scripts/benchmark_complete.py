# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "janitor-rs==0.6.6",
#   "multipledispatch",
#   "natsort>=8.4.0,<9",
#   "numpy",
#   "pandas>=3.0.0",
#   "pandas-flavor>=0.8.1,<0.9",
#   "scipy",
#   "typer>=0.12",
#   "xarray",
# ]
# ///
"""Benchmark ``complete`` on two git revisions of this repository.

The timed code is ``janitor.complete``. Comparisons are end to end, including
grouped frames and dictionary ``fill_value`` cases. A revision that raises is
reported as an error; it is not treated as a faster successful call.

Examples:
    uv run scripts/benchmark_complete.py
    uv run scripts/benchmark_complete.py --baseline dev --candidate HEAD
    uv run scripts/benchmark_complete.py --candidate WORKTREE --out /tmp/complete.json

``WORKTREE`` times the ``janitor`` package in the current directory. Every
other ref is exported with ``git archive``. Each revision runs in its own
process.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path

import typer


def repo_root() -> Path:
    """Return the repository root."""
    output = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    )
    return Path(output.strip())


def materialize(ref: str, dest: Path) -> None:
    """Export one revision's ``janitor`` package into ``dest``.

    Args:
        ref: Git ref, or ``WORKTREE`` for the current checkout.
        dest: Directory that will contain the ``janitor`` package.
    """
    if ref == "WORKTREE":
        shutil.copytree(
            repo_root() / "janitor",
            dest / "janitor",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return
    archive = subprocess.check_output(["git", "archive", ref, "janitor"])
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def import_janitor():
    """Import janitor without requiring an installed distribution.

    Returns:
        The imported janitor module.
    """
    import importlib.metadata as metadata

    original = metadata.version

    def version(name: str) -> str:
        """Return a package version, with a fallback for this checkout."""
        if name == "pyjanitor":
            try:
                return original(name)
            except metadata.PackageNotFoundError:
                return "0+benchmark"
        return original(name)

    metadata.version = version
    import janitor

    return janitor


def frame_token(frame) -> dict:
    """Summarize a frame so two revisions can be compared.

    Args:
        frame: DataFrame returned by ``complete``.

    Returns:
        Shape and a hash of the frame contents.
    """
    import pandas as pd

    hashed = pd.util.hash_pandas_object(frame, index=True)
    return {"shape": list(frame.shape), "hash": int(hashed.sum())}


def sample_call(func, repeats: int) -> list[float]:
    """Time ``func`` and return sample durations in milliseconds.

    Args:
        func: Callable to time. Its return value is discarded.
        repeats: Number of timed calls. One untimed warmup runs first.

    Returns:
        Elapsed milliseconds for each timed call.
    """
    func()
    gc.collect()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        func()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def run_case(func, repeats: int, warn: str) -> dict:
    """Run one case, recording either timings or the exception type.

    Args:
        func: Callable under test.
        repeats: Number of timed calls after a warmup.
        warn: ``ignore`` to hide warnings, or ``count`` to let them be built
            without writing them to stderr.

    Returns:
        A JSON-ready result dictionary.
    """
    warnings.resetwarnings()
    if warn == "count":
        warnings.simplefilter("always")
        warnings.showwarning = lambda *args, **kwargs: None
    else:
        warnings.simplefilter("ignore")
    try:
        result = func()
    except Exception as exc:
        started = time.perf_counter()
        try:
            func()
        except Exception:
            pass
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "ok": False,
            "error": type(exc).__name__,
            "median_ms": elapsed,
        }
    samples = sample_call(func, repeats)
    return {
        "ok": True,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "token": frame_token(result),
    }


def cases() -> list[dict]:
    """Build the benchmark cases against the imported janitor package.

    Returns:
        Case specifications. Callables are invoked by :func:`run_case`.
    """
    import numpy as np
    import pandas as pd

    import_janitor()
    rng = np.random.default_rng(0)
    specs = []

    def add(name, func, repeats, note, warn="ignore"):
        """Append one case specification."""
        specs.append(
            {
                "name": name,
                "func": func,
                "repeats": repeats,
                "note": note,
                "warn": warn,
            }
        )

    diagonal = pd.DataFrame(
        {
            "left": np.arange(1000),
            "right": np.arange(1000),
            "value": np.arange(1000),
        }
    )
    shuffled = diagonal.sample(frac=1, random_state=0).reset_index(drop=True)
    add(
        "cartesian_sorted_1m",
        lambda: diagonal.complete("left", "right"),
        3,
        "1,000 by 1,000 sorted domains, about 1M output rows",
    )
    add(
        "cartesian_unsorted_1m",
        lambda: shuffled.complete("left", "right"),
        3,
        "same domains in random row order",
    )

    small = pd.DataFrame(
        {
            "group": np.resize(np.arange(3), 600),
            "item": np.arange(600),
            "value": np.arange(600),
        }
    )
    add(
        "by_600_rows",
        lambda: small.complete("item", by="group"),
        20,
        "deprecated by= on 600 rows and 3 groups; warnings ignored",
    )
    add(
        "by_warning_overhead",
        lambda: small.complete("item", by="group"),
        30,
        "same call, building deprecation warnings with no stderr",
        warn="count",
    )

    groups = 2000
    grouped = pd.DataFrame(
        {
            "group": np.repeat(np.arange(groups), 2),
            "item": np.tile([1, 2], groups),
            "value": np.arange(groups * 2),
        }
    )
    add(
        "groupby_2000",
        lambda: grouped.groupby("group").complete({"item": [1, 2, 3]}),
        5,
        "2,000 groups expanding one missing item each",
    )
    add(
        "groupby_scalar_fill",
        lambda: grouped.groupby("group").complete(
            {"item": [1, 2, 3]},
            fill_value=0,
        ),
        5,
        "same grouped expansion with a scalar fill",
    )

    partial_groups = pd.DataFrame(
        {
            "state": np.arange(200),
            "year": 2020,
            "sales": np.arange(200),
            "profit": np.arange(200),
        }
    )
    add(
        "groupby_partial_fill",
        lambda: partial_groups.groupby("state").complete(
            {"year": [2020, 2021]},
            fill_value={"sales": 0},
        ),
        5,
        "grouped partial dictionary; dev raises AttributeError",
    )

    rows = 20_000
    width = 64
    payload = {f"v{i}": rng.random(rows) for i in range(width)}
    wide = pd.DataFrame({"key": np.arange(rows), **payload})
    names = [f"v{i}" for i in range(width)]
    add(
        "wide_partial_no_nulls",
        lambda: wide.complete("key", fill_value={"v0": 0.0}),
        5,
        "20,000 x 64 already complete; dictionary names one column",
    )
    extended = np.arange(rows + 500)
    add(
        "wide_partial_missing_rows",
        lambda: wide.complete({"key": extended}, fill_value={"v0": 0.0}),
        5,
        "adds 500 rows; dictionary names one of 64 payload columns",
    )
    full = {name: 0.0 for name in names}
    add(
        "wide_full_dict",
        lambda: wide.complete({"key": extended}, fill_value=full),
        5,
        "same 500 added rows, dictionary names every payload column",
    )
    add(
        "wide_scalar_fill",
        lambda: wide.complete({"key": extended}, fill_value=0.0),
        5,
        "same 500 added rows with a scalar fill",
    )
    return specs


def worker() -> None:
    """Execute every case and write JSON to stdout."""
    results = []
    for spec in cases():
        outcome = run_case(spec["func"], spec["repeats"], spec["warn"])
        outcome["name"] = spec["name"]
        outcome["note"] = spec["note"]
        results.append(outcome)
    json.dump(results, sys.stdout)
    sys.stdout.write("\n")


def run_revision(ref: str) -> list[dict]:
    """Time one git revision in a child process.

    Args:
        ref: Git ref or ``WORKTREE``.

    Returns:
        Case results for that revision.

    Raises:
        RuntimeError: If the worker process fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialize(ref, root)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark worker failed for {ref}.\n"
            f"{completed.stderr}\n{completed.stdout}"
        )
    return json.loads(completed.stdout)


def combine(baseline: list[dict], candidate: list[dict]) -> list[dict]:
    """Pair baseline and candidate results by case name.

    Args:
        baseline: Results from the baseline revision.
        candidate: Results from the candidate revision.

    Returns:
        One row per case with a ratio when both calls succeeded.
    """
    candidate_by_name = {row["name"]: row for row in candidate}
    rows = []
    for base in baseline:
        other = candidate_by_name[base["name"]]
        row = {
            "name": base["name"],
            "note": base["note"],
            "baseline": base,
            "candidate": other,
        }
        if base["ok"] and other["ok"]:
            row["ratio"] = other["median_ms"] / base["median_ms"]
            row["same_result"] = base["token"] == other["token"]
        else:
            row["ratio"] = None
            row["same_result"] = None
        rows.append(row)
    return rows


def format_status(result: dict) -> str:
    """Format one revision's median time or error.

    Args:
        result: A case result from :func:`run_case`.

    Returns:
        A short status string.
    """
    if result["ok"]:
        return f"{result['median_ms']:.2f} ms"
    return f"{result['error']} in {result['median_ms']:.2f} ms"


def render(rows: list[dict], baseline_name: str, candidate_name: str) -> str:
    """Render the comparison as a text table.

    Args:
        rows: Paired case results.
        baseline_name: Label for the baseline revision.
        candidate_name: Label for the candidate revision.

    Returns:
        The table, including a note when results differ.
    """
    header = (
        f"{'case':<28} {baseline_name:>22} {candidate_name:>22} "
        f"{'candidate/baseline':>18}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        ratio = "" if row["ratio"] is None else f"{row['ratio']:.2f}x"
        lines.append(
            f"{row['name']:<28} {format_status(row['baseline']):>22} "
            f"{format_status(row['candidate']):>22} {ratio:>18}"
        )
        lines.append(f"    {row['note']}")
        if row["same_result"] is False:
            lines.append("    results differ")
    lines.append("")
    lines.append(
        textwrap.fill(
            "Ratios below 1 mean the candidate returned faster. "
            "Cases where either revision raises are not speed comparisons. "
            "Medians use one untimed warmup. The warning-overhead case builds "
            "warnings and discards them instead of printing.",
            width=88,
        )
    )
    return "\n".join(lines)


def main(
    baseline: str = typer.Option("dev", help="Git ref for the baseline."),
    candidate: str = typer.Option(
        "HEAD",
        help="Git ref for the candidate, or WORKTREE for the working tree.",
    ),
    out: Path | None = typer.Option(None, help="Optional path for the JSON results."),
    run_worker: bool = typer.Option(False, "--worker", hidden=True),
) -> None:
    """Compare two revisions and print the table.

    Args:
        baseline: Git ref for the baseline.
        candidate: Git ref for the candidate, or ``WORKTREE``.
        out: Optional path for the JSON results.
        run_worker: Run the benchmark worker when true.
    """
    if run_worker:
        worker()
        return
    os.chdir(repo_root())
    baseline_rows = run_revision(baseline)
    candidate_rows = run_revision(candidate)
    rows = combine(baseline_rows, candidate_rows)
    print(render(rows, baseline, candidate))
    if out:
        out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)
