"""Tests for scale_mad function."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import janitor  # noqa: F401


@pytest.mark.functions
def test_scale_mad_scales_numeric_columns_default():
    """Scale numeric columns by default; leave constant and non-numeric unchanged."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 10, 10, 10],
            "label": ["a", "b", "c", "d"],
        }
    )
    original = df.copy()

    result = df.scale_mad()

    assert np.isclose(result["x"].median(), 0.0)
    assert_series_equal(result["y"], df["y"])
    assert_series_equal(result["label"], df["label"])
    assert result is not df
    assert df.equals(original)


@pytest.mark.functions
def test_scale_mad_zero_mad_center():
    """With zero_mad='center', constant column is centered to zero."""
    df = pd.DataFrame({"y": [10, 10, 10, 10]})

    result = df.scale_mad(zero_mad="center")

    assert (result["y"] == 0).all()


@pytest.mark.functions
def test_scale_mad_suffix_and_clip():
    """Suffix creates new column and clip bounds scaled values."""
    df = pd.DataFrame({"x": [1, 2, 3, 100]})

    result = df.scale_mad(columns=["x"], clip=3, suffix="_mad")

    assert "x_mad" in result.columns
    assert "x" in result.columns
    assert (result["x_mad"].abs() <= 3).all()


@pytest.mark.functions
def test_scale_mad_callable_column_selector():
    """columns can be a callable that returns column names (e.g. numeric only)."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    result = df.scale_mad(
        columns=lambda d: d.select_dtypes(include=["number"]).columns,
        suffix="_mad",
    )

    assert "a_mad" in result.columns
    assert "b" in result.columns


@pytest.mark.functions
def test_scale_mad_zero_mad_raise():
    """zero_mad='raise' raises ValueError when column has zero MAD."""
    df = pd.DataFrame({"y": [1, 1, 1]})

    with pytest.raises(ValueError, match="MAD is zero"):
        df.scale_mad(columns=["y"], zero_mad="raise")
