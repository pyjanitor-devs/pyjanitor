from janitor.functions import remove_dupes
import pandas as pd, numpy as np
import pytest


@pytest.mark.functions
def test_remove_dupes():
    """
    Tests for remove_dupes():
    Test 1: passing in an object that is not a Pandas DataFrame should
    return an empty DataFrame
    Test 2: passing in a value that is not last or first will return the
    DataFrame using "first" by default
    Test 3: passing in an empty DataFrame should return an empty DataFrame
    """
    company_sales = {
        'SalesMonth': ['Jan', 'Feb', 'Feb', 'Mar', 'April'],
        'Company1': [150.0, 200.0, 200.0, 300.0, 400.0],
        'Company2': [180.0, 250.0, 250.0, np.nan, 500.0],
        'Company3': [400.0, 500.0, 500.0, 600.0, 675.0]
    }
    df = pd.DataFrame.from_dict(company_sales)
    df = df.set_index("Company1")
    company_sales2 = {
        'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
        'Company1': [150.0, 200.0, 300.0, 400.0],
        'Company2': [180.0, 250.0, np.nan, 500.0],
        'Company3': [400.0, 500.0, 600.0, 675.0]
    }
    df_2 = pd.DataFrame.from_dict(company_sales2)
    df_2 = df_2.set_index("Company1")
    # assert (pd.DataFrame() == remove_dupes([], keep="first"))
    assert (df_2.equals(remove_dupes(df, keep="first")))
    assert (pd.DataFrame() == remove_dupes(pd.DataFrame(), keep="first"))
