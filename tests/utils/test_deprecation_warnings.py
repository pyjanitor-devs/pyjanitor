"""Regression tests for deprecation warning behavior."""

import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd


def test_import_does_not_override_user_warning_filter():
    """Importing janitor must respect the user's warning filters."""
    script = """
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
before = list(warnings.filters)

import janitor

added_filters = [
    entry for entry in warnings.filters if entry not in before
]

assert not any(
    action == "always" and issubclass(DeprecationWarning, category)
    for action, _, category, _, _ in added_filters
), "Importing janitor installed an always-on DeprecationWarning filter"

with warnings.catch_warnings(record=True) as recorded:
    warnings.warn("should remain hidden", DeprecationWarning)

assert not recorded, (
    "Importing janitor overrode the user's warning preference"
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_dataframe_deprecation_warning_points_to_caller():
    """A deprecated DataFrame method should warn at the user's call."""
    import janitor  # noqa: F401

    df = pd.DataFrame({"a": [1], "b": [2]})

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")

        expected_line = inspect.currentframe().f_lineno + 1
        df.remove_columns(["a"])

    deprecations = [
        warning for warning in recorded if "deprecated" in str(warning.message).lower()
    ]

    assert deprecations, "Expected a deprecation warning"

    warning = deprecations[0]

    assert Path(warning.filename).resolve() == Path(__file__).resolve()
    assert warning.lineno == expected_line


def _assert_future_warning_at_caller(call, message):
    """Check warning category and the exact caller location."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", FutureWarning)
        call()

    matches = [
        item
        for item in recorded
        if item.category is FutureWarning and message in str(item.message)
    ]

    assert len(matches) == 1, (
        f"Expected one FutureWarning containing {message!r}; found {len(matches)}"
    )

    warning = matches[0]
    assert Path(warning.filename).resolve() == Path(__file__).resolve()
    assert warning.lineno == call.__code__.co_firstlineno


def test_deprecated_kwargs_warning_location():
    """Deprecated keyword warnings should identify user code."""
    from janitor.utils import deprecated_kwargs

    @deprecated_kwargs("old", error=False)
    def example(*, old):
        return old

    _assert_future_warning_at_caller(
        lambda: example(old=1),
        "'old'",
    )


def test_deprecated_alias_warning_location():
    """Deprecated aliases should identify user code."""
    from janitor.utils import deprecated_alias

    @deprecated_alias(old="new")
    def example(*, new):
        return new

    _assert_future_warning_at_caller(
        lambda: example(old=1),
        "old is deprecated",
    )


def test_label_encode_future_warning_location():
    """label_encode should emit FutureWarning at the caller."""
    import janitor  # noqa: F401

    _assert_future_warning_at_caller(
        lambda: pd.DataFrame({"a": ["x", "y"]}).label_encode(column_names="a"),
        "Please use `janitor.factorize_columns` instead.",
    )


def test_conditional_join_deprecated_parameters_warning_location():
    """Each deprecated conditional_join parameter should warn at the caller."""
    import janitor  # noqa: F401

    left = pd.DataFrame({"a": [1, 2]})
    right = pd.DataFrame({"b": [1, 2]})

    options = [
        ("use_numba", False),
        ("df_columns", ["a"]),
        ("right_columns", ["b"]),
    ]

    for option, value in options:
        _assert_future_warning_at_caller(
            lambda: left.conditional_join(right, ("a", "b", "<"), **{option: value}),
            f"'{option}'",
        )


def test_get_join_indices_use_numba_warning_location():
    """get_join_indices should warn at the caller."""
    import janitor

    left = pd.DataFrame({"a": [1, 2]})
    right = pd.DataFrame({"b": [1, 2]})

    _assert_future_warning_at_caller(
        lambda: janitor.get_join_indices(
            left, right, ("a", "b", "=="), use_numba=False
        ),
        "'use_numba'",
    )


def test_refactored_methods_emit_one_future_warning():
    """Refactored methods should not emit duplicate FutureWarnings."""
    import janitor  # noqa: F401

    df = pd.DataFrame(
        {
            "group": ["a", "b", "a"],
            "value": [1, 2, 3],
        }
    )

    calls = {
        "label_encode": lambda: df.copy().label_encode("group"),
        "filter_on": lambda: df.filter_on("value > 1"),
        "filter_column_isin": lambda: df.filter_column_isin("group", ["a"]),
        "ungroup": lambda: df.groupby("group").ungroup(),
    }

    for name, call in calls.items():
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", FutureWarning)
            call()

        future_warnings = [item for item in recorded if item.category is FutureWarning]
        assert len(future_warnings) == 1, (
            f"{name} emitted {len(future_warnings)} FutureWarnings"
        )
