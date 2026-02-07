import pandas as pd
import pytest

import janitor


@pytest.mark.functions
def test_single_value_basic():
    assert janitor.single_value([1, 1, 1, None]) == 1


@pytest.mark.functions
def test_single_value_all_missing_returns_first_missing():
    result = janitor.single_value([None, "a"], missing=["a", None])
    assert result == "a"


@pytest.mark.functions
def test_single_value_warns_if_all_missing():
    with pytest.warns(UserWarning, match="All values are missing"):
        result = janitor.single_value([None, None], warn_if_all_missing=True)
    assert result is None


@pytest.mark.functions
def test_single_value_with_series_input():
    series = pd.Series([1, 1, pd.NA])
    assert janitor.single_value(series) == 1


@pytest.mark.functions
def test_single_value_info_in_error():
    with pytest.raises(
        ValueError,
        match=r"More than one \(2\) value found \(1, 2\): group A=1",
    ):
        janitor.single_value([1, 2], info="group A=1")
