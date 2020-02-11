import pandas as pd
import pytest

from janitor.utils import (
    _replace_empty_string_with_none,
    _replace_original_empty_string_with_none,
)


@pytest.mark.utils
def test_replace_empty_string_with_none():
    df = pd.DataFrame({"a": ["", 1, 0.34, "6.5", ""]})
    df_expected = pd.DataFrame({"a": [None, 1, 0.34, "6.5", None]})

    df["a"] = _replace_empty_string_with_none(df["a"])
    pd.testing.assert_series_equal(df["a"], df_expected["a"])


@pytest.mark.utils
def test_replace_original_empty_string_with_none():
    df = pd.DataFrame({"a": [1, 0.34, "6.5", None, "ORIGINAL_NA", "foo"]})
    df_expected = pd.DataFrame({"a": [1, 0.34, "6.5", None, None, "foo"]})

    df["a"] = _replace_original_empty_string_with_none(df["a"])
    pd.testing.assert_series_equal(df["a"], df_expected["a"])
