import pandas as pd
import pytest


@pytest.mark.functions
def test_charac():
    table_GDP = pd.read_html(
        'https://en.wikipedia.org/wiki/Economy_of_the_United_States',
        match='Nominal GDP')
    df = table_GDP[0]

    df = df.clean_names(strip_underscores=True, case_type='lower')


    assert 'current_accountbalance_in_%_of_gdp' in df.columns.values


def test_space():
    table_GDP = pd.read_html(
        'https://en.wikipedia.org/wiki/Economy_of_Russia',
        match='Year')
    df = table_GDP[0]

    df = df.clean_names(strip_underscores=True, case_type='lower')

    assert ("in %" in df.columns.values) is False
