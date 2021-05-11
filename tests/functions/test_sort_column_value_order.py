import pandas as pd
import numpy as np
from janitor.functions import sort_column_value_order
def test_sort_column_value_order():
    company_sales = {
        'SalesMonth': ['Jan', 'Feb', 'Feb', 'Mar', 'April'],
        'Company1': [150.0, 200.0, 200.0, 300.0, 400.0],
        'Company2': [180.0, 250.0, 250.0, np.nan, 500.0],
        'Company3': [400.0, 500.0, 500.0, 600.0, 675.0]
    }
    df = pd.DataFrame.from_dict(company_sales)
    df = df.set_index("Company1")
    company_sales_2 = {
        'SalesMonth': ['April', 'Mar', 'Feb', 'Feb', 'Jan'],
        'Company1': [400.0, 300.0, 200.0, 200.0, 150.0],
        'Company2': [500.0, np.nan, 250.0, 250.0, 180.0],
        'Company3': [675.0, 600.0, 500.0, 500.0, 400.0]
    }
    df2 = pd.DataFrame.from_dict(company_sales_2)
    df2 = df2.set_index("Company1")
    assert (pd.DataFrame().equals(
      sort_column_value_order(pd.DataFrame(),
                                          "", 
                                          {})))
    assert (pd.DataFrame().equals(sort_column_value_order(
                                                   df, 
                                                   "",
                                                   {
                                                     'April': 1, 
                                                     'Mar': 2, 
                                                     'Feb': 3, 
                                                     'Jan': 4
                                                   }
                                                    )))
    assert (
        df2.equals(sort_column_value_order(
                                            df, 
                                           "SalesMonth",
                                           {
                                             'April': 1, 
                                             'Mar': 2, 
                                             'Feb': 3, 
                                             'Jan': 4
                                           }
        )))
