from janitor import join_conditional
import pandas as pd
import numpy as np
import pytest


@pytest.mark.functions
def test_join_conditional():
    company_sales = {
        "SalesMonth": ["Jan", "Feb", "Mar", "April"],
        "Company1": [150.0, 200.0, 300.0, 400.0],
        "Company2": [180.0, 250.0, np.nan, 500.0],
        "Company3": [400.0, 500.0, 600.0, 675.0],
    }

    df = pd.DataFrame.from_dict(company_sales)

    company_sales2 = {
        "SalesMonth": ["Jan", "Feb", "April"],
        "Company1": [150.0, 200.0, 400.0],
        "Company2": [180.0, 250.0, 500.0],
        "Company3": [400.0, 500.0, 675.0],
    }
    df_2 = pd.DataFrame.from_dict(company_sales2)

    assert df_2.equals(join_conditional(df, "Company1", "Company2", "<"))
