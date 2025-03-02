import pandas as pd
from pandas.testing import assert_series_equal


def test_alias_no_name():
    """Test output if Series does not have a name"""
    series = pd.Series([1, 2, 3])
    assert_series_equal(series, series.alias())


def test_alias_callable():
    """Test output if alias is a callable"""
    series = pd.Series([1, 2, 3], name="UPPER")
    assert_series_equal(series.rename("upper"), series.alias(str.lower))


def test_alias_scalar():
    """Test output if alias is a scalar"""
    series = pd.Series([1, 2, 3], name="UPPER")
    assert_series_equal(series.rename("upper"), series.alias("upper"))
