import numpy as np
import pandas as pd

from janitor.functions import sort_column_value_order

"""
Below, company_sales and company_sales_2 are both dfs.

company_sales_2 is inverted, April is the first month
    where in comapny_sales Jan is the first month

The values found in each row are the same
    company_sales's Jan row contains the
    same values as company_sales_2's Jan row

Test 1 asserts sort_column may have parameters
    which will not alter the df passed without
    issue.

Test 2 asserts that columns may be ordered
    without issue

Test 3 asserts that company_sales_2 and
    company_sales with columns sorted
    will become equivilent, meaning
    the columns have been successfully ordered.
"""


def test_sort_column_value_order():
    company_sales = {
        "SalesMonth": ["Jan", "Feb", "Feb", "Mar", "April"],
        "Company1": [150.0, 200.0, 200.0, 300.0, 400.0],
        "Company2": [180.0, 250.0, 250.0, np.nan, 500.0],
        "Company3": [400.0, 500.0, 500.0, 600.0, 675.0],
    }
    df = pd.DataFrame.from_dict(company_sales)
    df = df.set_index("Company1")
    company_sales_2 = {
        "SalesMonth": ["April", "Mar", "Feb", "Feb", "Jan"],
        "Company1": [400.0, 300.0, 200.0, 200.0, 150.0],
        "Company2": [500.0, np.nan, 250.0, 250.0, 180.0],
        "Company3": [675.0, 600.0, 500.0, 500.0, 400.0],
    }
    df2 = pd.DataFrame.from_dict(company_sales_2)
    df2 = df2.set_index("Company1")
    assert pd.DataFrame().equals(
        sort_column_value_order(pd.DataFrame(), "", {})
    )
    assert pd.DataFrame().equals(
        sort_column_value_order(
            df, "", {"April": 1, "Mar": 2, "Feb": 3, "Jan": 4}
        )
    )
    assert df2.equals(
        sort_column_value_order(
            df, "SalesMonth", {"April": 1, "Mar": 2, "Feb": 3, "Jan": 4}
        )
    )
