"""
Below, company_sales and company_sales_2 are both dfs.

company_sales_2 is inverted, April is the first month
    where in company_sales Jan is the first month

The values found in each row are the same
    company_sales's Jan row contains the
    same values as company_sales_2's Jan row

Test 1 asserts that dfcannot be blank

Test 2 asserts that column cannot be blank

Test 3 asserts that company_sales_2 and
    company_sales with columns sorted
    will become equivalent, meaning
    the columns have been successfully ordered.
"""

import numpy as np
import pandas as pd
import pytest

from janitor.functions import sort_column_value_order


@pytest.fixture
def company_sales_df():
    """Fixture for tests below."""
    company_sales = {
        "SalesMonth": ["Jan", "Feb", "Feb", "Mar", "April"],
        "Company1": [150.0, 260.0, 230.0, 300.0, 400.0],
        "Company2": [500.0, 210.0, 250.0, np.nan, 80.0],
        "Company3": [400.0, 500.0, 500.0, 600.0, 675.0],
    }
    df = pd.DataFrame.from_dict(company_sales)
    return df


def test_sort_column_value_order(company_sales_df):
    """Main test for sort_column_value_order."""
    df = company_sales_df.set_index("Company1")
    company_sales_2 = {
        "SalesMonth": reversed(["Jan", "Feb", "Feb", "Mar", "April"]),
        "Company1": reversed([150.0, 230.0, 260.0, 300.0, 400.0]),
        "Company2": reversed([500.0, 250.0, 210.0, np.nan, 80.0]),
        "Company3": reversed([400.0, 500.0, 500.0, 600.0, 675.0]),
    }
    df2 = pd.DataFrame.from_dict(company_sales_2)
    df2 = df2.set_index("Company1")
    with pytest.raises(ValueError):
        assert pd.DataFrame().equals(sort_column_value_order(pd.DataFrame(), "", {}))
    with pytest.raises(ValueError):
        assert pd.DataFrame().equals(
            sort_column_value_order(df, "", {"April": 1, "Mar": 2, "Feb": 3, "Jan": 4})
        )
    assert df2.equals(
        df.sort_column_value_order(
            "SalesMonth", {"April": 1, "Mar": 2, "Feb": 3, "Jan": 4}
        )
    )


def test_sort_column_value_order_without_column_value_order(company_sales_df):
    """
    Test that sort_column_value_order raises a ValueError.

    In this case, it should raise ValueError
    when `column_value_order` is empty.
    """
    with pytest.raises(ValueError):
        company_sales_df.sort_column_value_order("SalesMonth", {})


def test_sort_preserves_cond_order_as_secondary_key():
    df = pd.DataFrame({"group": ["B", "A", "A"], "cond_order": [5, 3, 1]})
    expected = df.iloc[[2, 1, 0]].copy()
    result = df.sort_column_value_order("group", {"A": 1, "B": 2}, ["cond_order"])
    pd.testing.assert_frame_equal(result, expected)


def test_missing_secondary_key_raises():
    df = pd.DataFrame({"group": ["A", "A"], "cond_order": [2, 1]})
    with pytest.raises(KeyError, match="cond_order_"):
        df.sort_column_value_order("group", {"A": 1}, ["cond_order_"])


def test_sort_by_missing_value_column_label():
    df = pd.DataFrame({"group": ["A", "A"], pd.NA: [2, 1]})
    result = df.sort_column_value_order("group", {"A": 1}, [pd.NA])
    pd.testing.assert_frame_equal(result, df.iloc[[1, 0]])


def test_sort_preserves_unrelated_cond_order_columns():
    df = pd.DataFrame(
        {"group": ["B", "A"], "cond_order": [9, 8], "cond_order_": [4, 5]}
    )
    result = df.sort_column_value_order("group", {"A": 1, "B": 2})
    pd.testing.assert_frame_equal(result, df.iloc[[1, 0]])


def test_sort_preserves_index_level_named_cond_order():
    df = pd.DataFrame(
        {"group": ["A", "B", "A"]}, index=pd.Index([3, 2, 1], name="cond_order")
    )
    result = df.sort_column_value_order("group", {"A": 1, "B": 2})
    pd.testing.assert_frame_equal(result, df.iloc[[0, 2, 1]])


def test_sort_column_value_order_with_columns(company_sales_df):
    """Execution test for sort_column_value_order

    Used to test the case when columns is also specified.

    TODO: This test needs to be improved
    to be more than just an execution test.
    """
    _ = company_sales_df.set_index("Company1").sort_column_value_order(
        "SalesMonth",
        {"April": 1, "Mar": 2, "Feb": 3, "Jan": 4},
        columns=["Company2"],
    )
