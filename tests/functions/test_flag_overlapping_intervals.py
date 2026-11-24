"""Tests for the flag_overlapping_intervals function."""

import pandas as pd
import pytest

from janitor.functions.flag_overlapping_intervals import (
    flag_overlapping_intervals,
)


def test_flag_overlapping_intervals():
    """Test detection of overlapping intervals."""
    df = pd.DataFrame(
        {
            "start": [1, 3, 10, 12],
            "end": [5, 7, 15, 18],
        }
    )

    result = flag_overlapping_intervals(df, "start", "end")

    assert result["overlap_flag"].tolist() == [1, 1, 1, 1]


def test_no_overlapping_intervals():
    """Test intervals with no overlaps."""
    df = pd.DataFrame(
        {
            "start": [1, 5, 10],
            "end": [3, 7, 12],
        }
    )

    result = flag_overlapping_intervals(df, "start", "end")

    assert result["overlap_flag"].tolist() == [0, 0, 0]


def test_nested_intervals():
    """Test detection of nested intervals."""
    df = pd.DataFrame(
        {
            "start": [1, 2, 4],
            "end": [10, 3, 5],
        }
    )

    result = flag_overlapping_intervals(df, "start", "end")

    assert result["overlap_flag"].tolist() == [1, 1, 1]


def test_duplicate_intervals():
    """Test handling of duplicate intervals."""
    df = pd.DataFrame(
        {
            "start": [1, 1, 5],
            "end": [4, 4, 8],
        }
    )

    result = flag_overlapping_intervals(df, "start", "end")

    assert result["overlap_flag"].tolist() == [1, 1, 0]


def test_shared_endpoint_counts_as_overlap():
    """Test that shared endpoints count as overlaps."""
    df = pd.DataFrame(
        {
            "start": [1, 5],
            "end": [5, 8],
        }
    )

    result = flag_overlapping_intervals(df, "start", "end")

    assert result["overlap_flag"].tolist() == [1, 1]


def test_grouped_intervals():
    """Test overlap detection within groups."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "start": [1, 3, 1, 6],
            "end": [5, 7, 4, 8],
        }
    )

    result = flag_overlapping_intervals(df, "start", "end", by="group")

    assert result["overlap_flag"].tolist() == [1, 1, 0, 0]


def test_missing_interval_values():
    """Test handling of missing interval values."""
    df = pd.DataFrame(
        {
            "start": [1, None, 5],
            "end": [4, 6, None],
        }
    )

    result = flag_overlapping_intervals(df, "start", "end")

    assert result["overlap_flag"].tolist() == [0, 0, 0]


def test_invalid_interval_raises_error():
    """Test validation of invalid intervals."""
    df = pd.DataFrame(
        {
            "start": [5, 1],
            "end": [3, 4],
        }
    )

    with pytest.raises(ValueError):
        flag_overlapping_intervals(df, "start", "end")


def test_original_dataframe_is_not_modified():
    """Test that the input DataFrame remains unchanged."""
    df = pd.DataFrame(
        {
            "start": [1, 3],
            "end": [5, 7],
        }
    )
    original = df.copy()

    flag_overlapping_intervals(df, "start", "end")

    pd.testing.assert_frame_equal(df, original)
