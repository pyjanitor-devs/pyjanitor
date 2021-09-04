"""
Author: Eric J. Ma
Date: 18 July 2018

The intent of these tests is to test that dataframe method registration works.
"""
import pandas as pd
import pytest

import janitor  # noqa: F401


@pytest.fixture
def dataframe():
    data = {
        "a": [1, 2, 3],
        "Bell__Chart": [1, 2, 3],
        "decorated-elephant": [1, 2, 3],
    }
    df = pd.DataFrame(data)
    return df


def test_clean_names_registration(dataframe):
    """Test if DataFrame has clean_method method"""
    assert dataframe.__getattr__("clean_names")


def test_remove_empty_registration(dataframe):
    """Test if DataFrame has remove_empty method"""
    assert dataframe.__getattr__("remove_empty")


def test_get_dupes_registration(dataframe):
    """Test if DataFrame has get_dupes method"""
    assert dataframe.__getattr__("get_dupes")


def test_encode_categorical_registration(dataframe):
    """Test if DataFrame has encode_categorical method"""
    assert dataframe.__getattr__("encode_categorical")


def test_label_encode_registration(dataframe):
    """Test if DataFrame has label_encode method"""
    assert dataframe.__getattr__("label_encode")


def test_factorize_columns_registration(dataframe):
    """Test if DataFrame has factorize_columns method"""
    assert dataframe.__getattr__("factorize_columns")


def test_get_features_targets_registration(dataframe):
    """Test if DataFrame has get_features_targets method"""
    assert dataframe.__getattr__("get_features_targets")


def test_rename_column_registration(dataframe):
    """Test if DataFrame has rename_column method"""
    assert dataframe.__getattr__("rename_column")


def test_coalesce_registration(dataframe):
    """Test if DataFrame has coalesce method"""
    assert dataframe.__getattr__("coalesce")


def test_convert_excel_date_registration(dataframe):
    """Test if DataFrame has convert_excel_date method"""
    assert dataframe.__getattr__("convert_excel_date")


def test_convert_matlab_date_registration(dataframe):
    """Test if DataFrame has convert_matlab_date method"""
    assert dataframe.__getattr__("convert_matlab_date")


def test_convert_unix_date_registration(dataframe):
    """Test if DataFrame has convert_unix_date method"""
    assert dataframe.__getattr__("convert_unix_date")


def test_fill_empty_registration(dataframe):
    """Test if DataFrame has fill_empty method"""
    assert dataframe.__getattr__("fill_empty")


def test_expand_column_registration(dataframe):
    """Test if DataFrame has expand_column method"""
    assert dataframe.__getattr__("expand_column")


def test_concatenate_columns_registration(dataframe):
    """Test if DataFrame has concatenate_columns method"""
    assert dataframe.__getattr__("concatenate_columns")


def test_deconcatenate_column_registration(dataframe):
    """Test if DataFrame has deconcatenate_column method"""
    assert dataframe.__getattr__("deconcatenate_column")


def test_filter_string_registration(dataframe):
    """Test if DataFrame has filter_string method"""
    assert dataframe.__getattr__("filter_string")


def test_filter_on_registration(dataframe):
    """Test if DataFrame has filter_on method"""
    assert dataframe.__getattr__("filter_on")


def test_remove_columns_registration(dataframe):
    """Test if DataFrame has remove_columns method"""
    assert dataframe.__getattr__("remove_columns")


def test_change_type_registration(dataframe):
    """Test if DataFrame has change_type method"""
    assert dataframe.__getattr__("change_type")


def test_filter_date_registration(dataframe):
    """Test if DataFrame has filter_date method"""
    assert dataframe.__getattr__("filter_date")


def test_conditional_join_registration(dataframe):
    """Test if DataFrame has conditional_join method"""
    assert dataframe.__getattr__("conditional_join")


def test_pivot_longer_registration(dataframe):
    """Test if DataFrame has pivot_longer method"""
    assert dataframe.__getattr__("pivot_longer")


def test_pivot_wider_registration(dataframe):
    """Test if DataFrame has pivot_wider method"""
    assert dataframe.__getattr__("pivot_wider")


def test_expand_grid_registration(dataframe):
    """Test if DataFrame has expand_grid method"""
    assert dataframe.__getattr__("expand_grid")


def test_process_text_registration(dataframe):
    """Test if DataFrame has process_text method"""
    assert dataframe.__getattr__("process_text")


def test_fill_direction_registration(dataframe):
    """Test if DataFrame has fill_direction method"""
    assert dataframe.__getattr__("fill_direction")


def test_drop_constant_columns_registration(dataframe):
    """Test if DataFrame has drop_constant_columns method"""
    assert dataframe.__getattr__("drop_constant_columns")
