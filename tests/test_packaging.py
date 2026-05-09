"""Smoke tests for packaging metadata.

These tests guard the contract that ``pip install pyjanitor`` brings in a
working set of runtime dependencies. They catch regressions like #1597,
where pyjanitor relied on pandas APIs (e.g. ``pd.col``) but the
``pyproject.toml`` did not declare a pandas dependency at all, allowing
``pip`` to leave the resolved environment with an unsupported pandas (or
none).
"""

from __future__ import annotations

from importlib.metadata import requires

from packaging.requirements import Requirement


def _runtime_requirements() -> dict[str, Requirement]:
    """Return the parsed runtime requirements of the installed pyjanitor.

    We filter out PEP 508 ``extra ==`` markers so optional-dependency groups
    (``dev``, ``docs``, ``biology`` …) don't pollute the runtime view.
    """
    raw = requires("pyjanitor") or []
    runtime: dict[str, Requirement] = {}
    for line in raw:
        req = Requirement(line)
        if req.marker is not None and "extra" in str(req.marker):
            continue
        runtime[req.name.lower()] = req
    return runtime


def test_pandas_is_a_runtime_dependency():
    """pandas must be listed as a runtime dep, not just an optional/dev one.

    Without this, ``pip install pyjanitor`` would silently leave the user's
    environment with no pandas at all (or with an old pandas the package
    cannot run against). Issue #1597.
    """
    reqs = _runtime_requirements()
    assert "pandas" in reqs, (
        "pandas is missing from pyjanitor's runtime dependencies; "
        "see issue #1597 for context."
    )


def test_pandas_runtime_dependency_pins_to_pandas_3():
    """The declared pandas range must include >=3.0 (which the code requires).

    pyjanitor's modern code path uses ``pd.col`` and other pandas-3-only
    APIs (see CHANGELOG entry for #1590), so allowing a resolution of
    pandas<3 would yield ImportError-like failures at first use.
    """
    reqs = _runtime_requirements()
    pandas_req = reqs.get("pandas")
    assert pandas_req is not None, "pandas missing from runtime deps (#1597)"
    # The declared specifier must reject pandas 2.x. Probe the boundary
    # rather than parse the spec string directly: that way the assertion
    # tracks intent ("must not allow pandas 2.2.3") instead of the exact
    # operator/version we happened to write.
    assert not pandas_req.specifier.contains("2.2.3"), (
        f"pandas requirement {pandas_req.specifier!s} still admits "
        "pandas 2.x; pyjanitor needs pandas>=3 (#1597)."
    )
    assert pandas_req.specifier.contains("3.0.0"), (
        f"pandas requirement {pandas_req.specifier!s} excludes pandas 3.0.0, "
        "which is the minimum supported version (#1597)."
    )
