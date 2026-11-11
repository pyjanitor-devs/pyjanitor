"""Regression tests for complete and expand column-label handling.

Covers issue #1860: a one-column frame with ``explicit=False``, and integer
column labels in both the scalar and list-of-columns forms.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401


@pytest.fixture
def integer_labels_df():
    """Frame with integer column labels from issue 1860."""
    return pd.DataFrame({0: [1, 2], 1: [1, 2], 2: [1.0, None]})


@pytest.mark.functions
def test_complete_single_column_explicit_false():
    """A one-column frame can be completed with explicit=False.

    ``"".join`` of the only label equals that label, so merge used to reject
    the indicator. ``explicit=True`` and a missing fill value already worked.
    """
    frame = pd.DataFrame({"a": [1, 3]})

    assert_frame_equal(frame.complete("a", fill_value=0, explicit=False), frame)
    assert_frame_equal(frame.complete("a", fill_value=0, explicit=True), frame)
    assert_frame_equal(frame.complete("a"), frame)


@pytest.mark.functions
def test_complete_single_integer_column_explicit_false():
    """One integer column, including the falsy label 0, with explicit=False."""
    frame = pd.DataFrame({0: [1, 3]})

    assert_frame_equal(frame.complete(0, fill_value=0, explicit=False), frame)

    result = frame.complete(
        pd.Index([1, 2, 3], name=0),
        fill_value=0,
        explicit=False,
    )
    expected = pd.DataFrame({0: [1, 2, 3]})
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_complete_single_column_explicit_false_adds_missing_value():
    """Missing values of a one-column frame are exposed when explicit=False."""
    frame = pd.DataFrame({"a": [1, 3]})
    result = frame.complete(
        pd.Index([1, 2, 3], name="a"),
        fill_value=0,
        explicit=False,
    )
    expected = pd.DataFrame({"a": [1, 2, 3]})
    assert_frame_equal(result, expected)


@pytest.mark.functions
@pytest.mark.parametrize("explicit", [True, False])
def test_complete_indicator_skips_names_already_present(explicit):
    """The indicator is not an existing column, and only new rows are filled.

    ``explicit=False`` fills the new row and leaves the original null.
    ``explicit=True`` fills both. A column named ``_merge`` must survive.
    """
    frame = pd.DataFrame({"_merge": [1, 3], "v": [10.0, None]})
    result = frame.complete(
        pd.Index([1, 2, 3], name="_merge"),
        fill_value=0,
        explicit=explicit,
    )
    original_null = 0.0 if explicit else np.nan
    expected = pd.DataFrame({"_merge": [1, 2, 3], "v": [10.0, 0.0, original_null]})
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_complete_indicator_skips_null_character_label():
    """A column whose label is the first candidate indicator is not reused."""
    frame = pd.DataFrame({"\0": [1, 3]})
    result = frame.complete(
        pd.Index([1, 2, 3], name="\0"),
        fill_value=0,
        explicit=False,
    )
    expected = pd.DataFrame({"\0": [1, 2, 3]})
    assert_frame_equal(result, expected)


@pytest.mark.functions
@pytest.mark.parametrize("columns", [(0, 1), ([0], [1])])
@pytest.mark.parametrize("explicit", [True, False])
def test_complete_integer_labels_without_fill(integer_labels_df, columns, explicit):
    """Integer labels complete, with or without a list around each label."""
    original = integer_labels_df.copy()
    result = integer_labels_df.complete(*columns, explicit=explicit)
    expected = pd.DataFrame(
        {0: [1, 1, 2, 2], 1: [1, 2, 1, 2], 2: [1.0, np.nan, np.nan, np.nan]}
    )

    assert_frame_equal(integer_labels_df, original)
    assert_frame_equal(result, expected)


@pytest.mark.functions
@pytest.mark.parametrize("columns", [(0, 1), ([0], [1])])
@pytest.mark.parametrize("explicit", [True, False])
def test_complete_integer_labels_with_fill(integer_labels_df, columns, explicit):
    """Fill integer-labeled frames, including the list-of-columns form.

    ``explicit=False`` fills only combinations that were not already present,
    so the original null in column 2 stays missing.
    """
    original = integer_labels_df.copy()
    result = integer_labels_df.complete(
        *columns,
        fill_value=0,
        explicit=explicit,
    )
    original_null = 0.0 if explicit else np.nan
    expected = pd.DataFrame(
        {0: [1, 1, 2, 2], 1: [1, 2, 1, 2], 2: [1.0, 0.0, 0.0, original_null]}
    )

    assert_frame_equal(integer_labels_df, original)
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_expand_integer_labels(integer_labels_df):
    """expand accepts an integer label, including the falsy label 0."""
    assert_frame_equal(integer_labels_df.expand(0), pd.DataFrame({0: [1, 2]}))
    crossed = pd.DataFrame({0: [1, 1, 2, 2], 1: [1, 2, 1, 2]})
    assert_frame_equal(integer_labels_df.expand(0, 1), crossed)
    assert_frame_equal(integer_labels_df.expand([0], [1]), crossed)
