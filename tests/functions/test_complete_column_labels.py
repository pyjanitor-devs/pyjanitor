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
@pytest.mark.parametrize("explicit", [True, False])
def test_complete_groupby_integer_label(explicit):
    """Grouped complete keeps an integer group key, including the label 0.

    Within each group the observed values of column 1 are already complete,
    so the result is the input. ``explicit=False`` must not turn the key
    into an all-null column.
    """
    frame = pd.DataFrame({0: [1, 1, 2], 1: [1, 2, 1], 2: [1.0, 2, 3]})
    original = frame.copy()
    result = frame.groupby(0).complete(1, fill_value=0, explicit=explicit)

    assert_frame_equal(frame, original)
    assert_frame_equal(result, frame)


@pytest.mark.functions
def test_complete_groupby_integer_label_without_fill():
    """A grouped complete with no fill keeps the integer group key."""
    frame = pd.DataFrame({0: [1, 1, 2], 1: [1, 2, 1], 2: [1.0, 2, 3]})
    assert_frame_equal(frame.groupby(0).complete(1), frame)
    assert_frame_equal(frame.groupby(0).complete([1]), frame)


@pytest.mark.functions
def test_complete_groupby_integer_label_past_nlevels():
    """An integer group key above the index depth is still a column name."""
    frame = pd.DataFrame({5: [1, 1, 2], 6: [1, 2, 1], 7: [1.0, None, 3.0]})
    original = frame.copy()
    implicit = frame.groupby(5).complete(6, fill_value=0, explicit=False)
    explicit = frame.groupby(5).complete(6, fill_value=0, explicit=True)
    filled = frame.copy()
    filled.loc[filled[7].isna(), 7] = 0

    assert_frame_equal(frame, original)
    assert_frame_equal(implicit, frame)
    assert_frame_equal(explicit, filled)


@pytest.mark.functions
@pytest.mark.parametrize("explicit", [True, False])
def test_complete_by_integer_label(explicit):
    """The deprecated ``by`` path uses the same integer group key."""
    frame = pd.DataFrame({0: [1, 1, 2], 1: [1, 2, 1], 2: [1.0, 2, 3]})
    with pytest.warns(FutureWarning, match="by"):
        result = frame.complete(1, by=0, fill_value=0, explicit=explicit)
    assert_frame_equal(result, frame)


@pytest.mark.functions
@pytest.mark.parametrize("explicit", [True, False])
def test_complete_groupby_two_integer_keys(explicit):
    """Both integer group keys survive a grouped complete.

    The null is in the expanded column, so ``fill_value`` does not replace it.
    """
    frame = pd.DataFrame({0: [1, 1, 2], 1: [10, 11, 20], 2: [1.0, None, 3.0]})
    result = frame.groupby([0, 1]).complete(2, fill_value=0, explicit=explicit)
    assert_frame_equal(result, frame)


@pytest.mark.functions
@pytest.mark.parametrize("explicit", [True, False])
def test_complete_groupby_integer_label_adds_missing_pair(explicit):
    """A missing pair inside an integer-keyed group is filled, and only that pair."""
    frame = pd.DataFrame(
        {
            0: [1, 1, 1],
            1: [1, 2, 1],
            2: ["a", "b", "b"],
            3: [10.0, 20.0, None],
        }
    )
    result = frame.groupby(0).complete(1, 2, fill_value=0, explicit=explicit)
    original_null = 0.0 if explicit else np.nan
    expected = pd.DataFrame(
        {
            0: [1, 1, 1, 1],
            1: [1, 1, 2, 2],
            2: ["a", "b", "a", "b"],
            3: [10.0, original_null, 0.0, 20.0],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_expand_integer_labels(integer_labels_df):
    """expand accepts an integer label, including the falsy label 0."""
    assert_frame_equal(integer_labels_df.expand(0), pd.DataFrame({0: [1, 2]}))
    crossed = pd.DataFrame({0: [1, 1, 2, 2], 1: [1, 2, 1, 2]})
    assert_frame_equal(integer_labels_df.expand(0, 1), crossed)
    assert_frame_equal(integer_labels_df.expand([0], [1]), crossed)
