import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pandas.testing import assert_series_equal

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(
    df=df_strategy(),
    x_vals=st.floats(),
    n_yvals=st.integers(min_value=0, max_value=100),
)
@settings(deadline=None)
def test_add_columns(df, x_vals, n_yvals):
    """
    Test for adding multiple columns at the same time.
    """
    y_vals = np.linspace(0, 42, n_yvals)

    if n_yvals != len(df) or n_yvals == 0:
        with pytest.raises(ValueError):
            df = df.add_columns(x=x_vals, y=y_vals)

    else:
        df = df.add_columns(x=x_vals, y=y_vals)
        series = pd.Series([x_vals] * len(df))
        series.name = "x"
        assert_series_equal(df["x"], series)

        series = pd.Series(y_vals)
        series.name = "y"
        assert_series_equal(df["y"], series)
