import numpy as np
import pandas as pd
import pytest

import janitor
from janitor import (clean_names, coalesce, convert_excel_date,
                     encode_categorical, expand_column, get_dupes,
                     remove_empty)


@pytest.fixture
def dataframe():
    data = {
        'a': [1, 2, 3],
        'Bell__Chart': [1, 2, 3],
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


@pytest.fixture
def multiindex_dataframe():
    data = {
        ('a', 'b'): [1, 2, 3],
        ('Bell__Chart', 'Normal  Distribution'): [1, 2, 3],
        ('decorated-elephant', "r.i.p-rhino :'("): [1, 2, 3],
    }
    df = pd.DataFrame(data)
    return df


def test_clean_names_functional(dataframe):
    df = clean_names(dataframe)
    expected_columns = ['a', 'bell_chart', 'decorated_elephant']

    assert set(df.columns) == set(expected_columns)


def test_clean_names_method_chain(dataframe):
    df = dataframe.clean_names()
    expected_columns = ['a', 'bell_chart', 'decorated_elephant']
    assert set(df.columns) == set(expected_columns)


def test_clean_names_pipe(dataframe):
    df = dataframe.pipe(clean_names)
    expected_columns = ['a', 'bell_chart', 'decorated_elephant']
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


def test_get_features_targets(dataframe):
    dataframe = dataframe.clean_names()
    X, y = dataframe.get_features_targets(target_columns='bell_chart')
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_rename_column(dataframe):
    df = dataframe.clean_names().rename_column('a', 'index')
    assert set(df.columns) == set(['index', 'bell_chart', 'decorated_elephant'])  # noqa: E501


def test_coalesce():
    df = pd.DataFrame({'a': [1, np.nan, 3],
                       'b': [2, 3, 1],
                       'c': [2, np.nan, 9]})

    df = coalesce(df, ['a', 'b', 'c'], 'a')
    assert df.shape == (3, 1)
    assert pd.isnull(df).sum().sum() == 0


def test_convert_excel_date():
    df = pd.read_excel('examples/dirty_data.xlsx').clean_names()
    df = convert_excel_date(df, 'hire_date')

    assert df['hire_date'].dtype == 'M8[ns]'


def test_fill_empty(null_df):
    df = null_df.fill_empty(columns=['2'], value=3)
    assert set(df.loc[:, '2']) == set([3])


def test_single_column_label_encode():
    df = pd.DataFrame({'a': ['hello', 'hello', 'sup'],
                       'b': [1, 2, 3]}).label_encode(columns='a')
    assert 'a_enc' in df.columns


def test_single_column_fail_label_encode():
    with pytest.raises(AssertionError):
        df = pd.DataFrame({'a': ['hello', 'hello', 'sup'],
                           'b': [1, 2, 3]}).label_encode(columns='c')


def test_multicolumn_label_encode():
    df = (pd.DataFrame({'a': ['hello', 'hello', 'sup'],
                        'b': [1, 2, 3],
                        'c': ['aloha', 'nihao', 'nihao']})
          .label_encode(columns=['a', 'c']))
    assert 'a_enc' in df.columns
    assert 'c_enc' in df.columns


def test_multiindex_clean_names_functional(multiindex_dataframe):
    df = clean_names(multiindex_dataframe)

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino_']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_multiindex_clean_names_method_chain(multiindex_dataframe):
    df = multiindex_dataframe.clean_names()

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino_']
    ]

    labels = [[0, 1, 2], [0, 1, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_multiindex_clean_names_pipe(multiindex_dataframe):
    df = multiindex_dataframe.pipe(clean_names)

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino_']
    ]

    labels = [[0, 1, 2], [0, 1, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_clean_names_strip_underscores_both(multiindex_dataframe):
    df = multiindex_dataframe.rename(columns=lambda x: '_' + x)
    df = clean_names(multiindex_dataframe, strip_underscores='both')

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_clean_names_strip_underscores_true(multiindex_dataframe):
    df = multiindex_dataframe.rename(columns=lambda x: '_' + x)
    df = clean_names(multiindex_dataframe, strip_underscores=True)

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_clean_names_strip_underscores_right(multiindex_dataframe):
    df = clean_names(multiindex_dataframe, strip_underscores='right')

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_clean_names_strip_underscores_r(multiindex_dataframe):
    df = clean_names(multiindex_dataframe, strip_underscores='r')

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_clean_names_strip_underscores_left(multiindex_dataframe):
    df = multiindex_dataframe.rename(columns=lambda x: '_' + x)
    df = clean_names(multiindex_dataframe, strip_underscores='left')

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino_']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_clean_names_strip_underscores_l(multiindex_dataframe):
    df = multiindex_dataframe.rename(columns=lambda x: '_' + x)
    df = clean_names(multiindex_dataframe, strip_underscores='l')

    levels = [
        ['a', 'bell_chart', 'decorated_elephant'],
        ['b', 'normal_distribution', 'r_i_p_rhino_']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_incorrect_strip_underscores(multiindex_dataframe):
    with pytest.raises(janitor.errors.JanitorError):
        df = clean_names(multiindex_dataframe, strip_underscores='hello')


def test_clean_names_preserve_case_true(multiindex_dataframe):
    df = multiindex_dataframe.rename(columns=lambda x: '_' + x)
    df = clean_names(multiindex_dataframe, preserve_case=True)

    levels = [
        ['a', 'Bell_Chart', 'decorated_elephant'],
        ['b', 'Normal_Distribution', 'r_i_p_rhino_']
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_expand_column():
    data = {'col1': ['A, B', 'B, C, D', 'E, F', 'A, E, F'],
            'col2': [1, 2, 3, 4]}

    df = pd.DataFrame(data)
    expanded = expand_column(df, 'col1', sep=', ', concat=False)
    assert expanded.shape[1] == 6
