import numpy as np
import pandas as pd
import pytest

from janitor.testing_utils.strategies import df_strategy
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays


@pytest.mark.functions
@given(
    df=df_strategy(),
)
def test_add_columns(df):
    """
    Test for adding multiple columns at the same time.
    """
    x_vals = 42
    y_vals = np.linspace(0, 42, len(df))
    # assume(len(y_vals) == len(df))

    df = df.add_columns(x=x_vals, y=y_vals)

    series = pd.Series([x_vals] * len(df))
    series.name = "x"
    pd.testing.assert_series_equal(df["x"], series)

    series = pd.Series(y_vals)
    series.name = "y"
    pd.testing.assert_series_equal(df["y"], series)
