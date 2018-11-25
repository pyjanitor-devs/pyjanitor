"""
Author: Eric J. Ma
Date: 18 July 2018

The intent of these tests is to test that dataframe method registration works.
"""
import pandas as pd
import pytest
import janitor


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
    assert dataframe.__getattr__("clean_names")


def test_remove_empty_registration(dataframe):
    assert dataframe.__getattr__("remove_empty")


def test_get_dupes_registration(dataframe):
    assert dataframe.__getattr__("get_dupes")


def test_encode_categorical_registration(dataframe):
    assert dataframe.__getattr__("encode_categorical")


def test_label_encode_registration(dataframe):
    assert dataframe.__getattr__("label_encode")


def test_get_features_targets_registration(dataframe):
    assert dataframe.__getattr__("get_features_targets")


def test_rename_column_registration(dataframe):
    assert dataframe.__getattr__("rename_column")


def test_coalesce_registration(dataframe):
    assert dataframe.__getattr__("coalesce")


def test_convert_excel_date_registration(dataframe):
    assert dataframe.__getattr__("convert_excel_date")


def test_fill_empty_registration(dataframe):
    assert dataframe.__getattr__("fill_empty")


def test_expand_column_registration(dataframe):
    assert dataframe.__getattr__("expand_column")


def test_concatenate_columns_registration(dataframe):
    assert dataframe.__getattr__("concatenate_columns")


def test_deconcatenate_column_registration(dataframe):
    assert dataframe.__getattr__("deconcatenate_column")


def test_filter_string_registration(dataframe):
    assert dataframe.__getattr__("filter_string")


def test_filter_on_registration(dataframe):
    assert dataframe.__getattr__("filter_on")


def test_remove_columns_registration(dataframe):
    assert dataframe.__getattr__("remove_columns")


def test_change_type_registration(dataframe):
    assert dataframe.__getattr__("change_type")
