import pytest
from janitor import clean_names, remove_empty
import janitor as jn
import pandas as pd
import numpy as np


@pytest.fixture
def dataframe():
    data = {
        'a': [1, 2, 3],
        'Bell Chart': [1, 2, 3],
        'decorated-elephant': [1, 2, 3],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def null_df():
    np.random.seed([3, 1415])
    df = pd.DataFrame(np.random.choice((1, np.nan), (10, 2)))
    df['2'] = np.nan * 10
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


def test_remove_empty(null_df):
    df = remove_empty(null_df)
    assert df.shape == (8, 2)
