import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from scipy.stats import norm


@pytest.mark.functions
def test_normal_cdf():
    s = pd.Series([0, 1, 2, 3, -1])
    out = s.normal_cdf()
    assert (out == norm.cdf(s)).all()
    assert (s.index == out.index).all()


@pytest.mark.parametrize(
    "dtype",
    ["int8", "int64", "float16", "float32", "float64", "Int64", "Float32", "Float64"],
)
def test_normal_cdf_dtype_and_metadata(dtype):
    """Keep the normal CDF's promotion, index, name, and input preservation."""
    s = pd.Series([-3, -1, 0, 1, 3], dtype=dtype, index=[4, 2, 2, 1, 0], name="values")
    original = s.copy(deep=True)
    expected = pd.Series(norm.cdf(s), index=s.index)

    assert_series_equal(s.normal_cdf(), expected, check_exact=True)
    assert_series_equal(s, original, check_exact=True)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_normal_cdf_tails_and_missing_values(dtype):
    """Match SciPy at extreme tails, infinities, and missing values."""
    s = pd.Series([-np.inf, -40, -10, -1, 0, 1, 8, 40, np.inf, np.nan], dtype=dtype)
    expected = pd.Series(norm.cdf(s), index=s.index)
    assert_series_equal(s.normal_cdf(), expected, check_exact=True)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float64"])
def test_normal_cdf_empty(dtype):
    """An empty input still produces an unnamed float64 Series."""
    s = pd.Series([], dtype=dtype, name="values")
    assert_series_equal(s.normal_cdf(), pd.Series([], dtype="float64"))


def test_normal_cdf_multiindex():
    """Preserve a named MultiIndex, including duplicate labels."""
    index = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 1), ("b", 2)], names=["group", "row"]
    )
    s = pd.Series([-1.0, 0.0, 1.0], index=index, name="values")
    assert_series_equal(
        s.normal_cdf(), pd.Series(norm.cdf(s), index=index), check_exact=True
    )


@pytest.mark.parametrize("values", [[True, False], np.array([0, 1, 2], dtype="uint64")])
def test_normal_cdf_unsigned_and_boolean(values):
    """Promote unsigned integers and booleans as the distribution API does."""
    s = pd.Series(values, name="values")
    assert_series_equal(s.normal_cdf(), pd.Series(norm.cdf(s)), check_exact=True)
