import numpy as np
import pandas as pd
import pytest

import janitor as jn
from janitor import clean_names, encode_categorical, get_dupes, remove_empty


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


def test_get_dupes():
    df = pd.DataFrame()
    df['a'] = [1, 2, 1]
    df['b'] = [1, 2, 1]
    df_dupes = get_dupes(df)
    assert df_dupes.shape == (2, 2)

    df2 = pd.DataFrame()
    df2['a'] = [1, 2, 3]
    df2['b'] = [1, 2, 3]
    df2_dupes = get_dupes(df2)
    assert df2_dupes.shape == (0, 2)


def test_encode_categorical():
    df = pd.DataFrame()
    df['class_label'] = ['test1', 'test2', 'test1', 'test2']
    df['numbers'] = [1, 2, 3, 2]
    df = encode_categorical(df, 'class_label')
    assert df['class_label'].dtypes == 'category'
