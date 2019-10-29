import pandas as pd
import pytest
from numpy import NaN
from pandas.testing import assert_frame_equal

from janitor.finance import currency_column_to_numeric


@pytest.mark.finance
def test_normal_case():
    df = pd.DataFrame(
        {"whozits": ["9", "3.50", "5"], "whatsits": ["1000000", "13.42", "7"]}
    )
    output = pd.DataFrame(
        {"whozits": [9, 3.50, 5], "whatsits": ["1000000", "13.42", "7"]}
    )
    result = currency_column_to_numeric(df, "whozits")
    assert_frame_equal(result, output, check_dtype=float)


@pytest.mark.finance
def test_df_with_missing_vals():
    df = pd.DataFrame(
        {"whozits": ["9", "", "5"], "whatsits": ["1000000", "13.42", ""]}
    )
    output = pd.DataFrame(
        {"whozits": [9, NaN, 5], "whatsits": ["1000000", "13.42", ""]}
    )
    assert_frame_equal(currency_column_to_numeric(df, "whozits"), output)


@pytest.mark.finance
def test_accounting_ledger():
    df = pd.DataFrame(
        {
            "whozits": ["9", "3,500", "5"],
            "whatsits": ["(1,000,000)", "5,000", "(7.10)"],
        }
    )
    output = pd.DataFrame(
        {"whozits": ["9", "3,500", "5"], "whatsits": [-1000000, 5000, -7.10]}
    )
    result = currency_column_to_numeric(
        df, "whatsits", cleaning_style="accounting"
    )
    assert_frame_equal(result, output, check_dtype=float)


@pytest.mark.finance
def test_casting():
    df = pd.DataFrame(
        {
            "whozits": ["9", "REORDER", "5"],
            "whatsits": ["1000000", "13.42", "REORDER"],
        }
    )
    output = pd.DataFrame(
        {"whozits": [9, 1, 5], "whatsits": ["1000000", "13.42", "REORDER"]}
    )
    result = currency_column_to_numeric(
        df, "whozits", cast_non_numeric={"REORDER": 1}
    )
    assert_frame_equal(output, result)


@pytest.mark.finance
def test_fill_non_numeric():
    df = pd.DataFrame(
        {
            "whozits": ["9", "not currency", "5"],
            "whatsits": ["1000000", "13.42", "also not"],
        }
    )
    output = pd.DataFrame(
        {
            "whozits": ["9", "not currency", "5"],
            "whatsits": [1000000, 13.42, 1],
        }
    )
    result = currency_column_to_numeric(df, "whatsits", fill_all_non_numeric=1)
    assert_frame_equal(output, result)


@pytest.mark.finance
def test_remove_non_numeric():
    df = pd.DataFrame(
        {
            "whozits": ["9", "not currency", "5"],
            "whatsits": ["1000000", "13.42", "also not"],
        }
    )
    output = pd.DataFrame(
        {"whozits": [9.0, 5.0], "whatsits": ["1000000", "also not"]}
    )
    result = currency_column_to_numeric(df, "whozits", remove_non_numeric=True)
    # shape used instead of frame_equal as the ouput keeps original index
    assert result.shape == output.shape
