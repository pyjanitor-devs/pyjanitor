import pytest
import pandas as pd

def test_charac():
    table_GDP = pd.read_html('https://en.wikipedia.org/wiki/Economy_of_the_United_States', match='Nominal GDP')
    df = table_GDP[0]

    df = df.clean_names(strip_underscores=True, case_type='lower')

    print(df.columns)
    
    assert 'current_accountbalance_in_%_of_gdp' in df.columns.values  #== True