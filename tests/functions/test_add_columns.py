import numpy as np
import pandas as pd

from janitor.testing_utils.fixtures import dataframe


@pytest.mark.functions
def test_add_columns(dataframe):
    # sanity checking is pretty much handled in test_add_column

    # multiple column addition with scalar and iterable

    x_vals = 42
    y_vals = np.linspace(0, 42, len(dataframe))

    df = dataframe.add_columns(x=x_vals, y=y_vals)

    series = pd.Series([x_vals] * len(dataframe))
    series.name = "x"
    pd.testing.assert_series_equal(df["x"], series)

    series = pd.Series(y_vals)
    series.name = "y"
    pd.testing.assert_series_equal(df["y"], series)
