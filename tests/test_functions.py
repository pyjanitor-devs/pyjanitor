import pytest
from janitor import clean_names
import janitor as jn
import pandas as pd


@pytest.fixture
def dataframe():
    data = {
        'a': [1, 2, 3],
        'Bell Chart': [1, 2, 3],
        'decorated-elephant': [1, 2, 3],
    }
    df = pd.DataFrame(data)
    return df


def test_clean_names_functional(dataframe):
    df = clean_names(dataframe)
    expected_columns = ['a', 'bell_chart', 'decorated-elephant']

    assert set(df.columns) == set(expected_columns)


def test_clean_names_method_chain(dataframe):
    df = jn.DataFrame(dataframe).clean_names()
    expected_columns = ['a', 'bell_chart', 'decorated-elephant']
    assert set(df.columns) == set(expected_columns)


def test_clean_names_pipe(dataframe):
    df = dataframe.pipe(clean_names)
    expected_columns = ['a', 'bell_chart', 'decorated-elephant']
    assert set(df.columns) == set(expected_columns)
