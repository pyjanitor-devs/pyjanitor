# import numpy as np
import pandas as pd
import pytest

# from pandas.testing import assert_frame_equal

# import re


@pytest.fixture
def df_checks():
    """fixture dataframe"""
    return pd.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
        }
    )


@pytest.fixture
def df_multi():
    """MultiIndex dataframe fixture."""
    return pd.DataFrame(
        {
            ("name", "a"): {0: "Wilbur", 1: "Petunia", 2: "Gregory"},
            ("names", "aa"): {0: 67, 1: 80, 2: 64},
            ("more_names", "aaa"): {0: 56, 1: 90, 2: 50},
        }
    )


def test_column_level_wrong_type(df_multi):
    """Raise TypeError if wrong type is provided for column_level."""
    with pytest.raises(TypeError):
        df_multi.pivot_longer(index="name", column_level={0})


def test_type_index(df_checks):
    """Raise TypeError if wrong type is provided for the index."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(index=2007)


def test_type_column_names(df_checks):
    """Raise TypeError if wrong type is provided for column_names."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(column_names=2007)


def test_type_names_to(df_checks):
    """Raise TypeError if wrong type is provided for names_to."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(names_to={2007})


def test_subtype_names_to(df_checks):
    """
    Raise TypeError if names_to is a sequence
    and the wrong type is provided for entries
    in names_to.
    """
    with pytest.raises(TypeError):
        df_checks.pivot_longer(names_to=[("famid",)])


def test_duplicate_names_to(df_checks):
    """Raise error if names_to contains duplicates."""
    with pytest.raises(ValueError, match="y already exists in names_to."):
        df_checks.pivot_longer(names_to=["y", "y"], names_pattern="(.+)(.)")


def test_both_names_sep_and_pattern(df_checks):
    """
    Raise ValueError if both names_sep
    and names_pattern is provided.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["rar", "bar"], names_sep="-", names_pattern="(.+)(.)"
        )


def test_name_pattern_wrong_type(df_checks):
    """Raise TypeError if the wrong type provided for names_pattern."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(names_to=["rar", "bar"], names_pattern=2007)


def test_name_pattern_no_names_to(df_checks):
    """Raise ValueError if names_pattern and names_to is None."""
    with pytest.raises(ValueError):
        df_checks.pivot_longer(names_to=None, names_pattern="(.+)(.)")


def test_name_pattern_groups_len(df_checks):
    """
    Raise ValueError if names_pattern
    and the number of groups
    differs from the length of names_to.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(names_to=".value", names_pattern="(.+)(.)")


def test_names_pattern_wrong_subtype(df_checks):
    """
    Raise TypeError if names_pattern is a list/tuple
    and wrong subtype is supplied.
    """
    with pytest.raises(TypeError):
        df_checks.pivot_longer(
            names_to=["ht", "num"], names_pattern=[1, "\\d"]
        )


def test_names_pattern_names_to_unequal_length(df_checks):
    """
    Raise ValueError if names_pattern is a list/tuple
    and wrong number of items in names_to.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["variable"], names_pattern=["^ht", ".+i.+"]
        )


def test_names_pattern_names_to_dot_value(df_checks):
    """
    Raise Error if names_pattern is a list/tuple and
    .value in names_to.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["variable", ".value"], names_pattern=["^ht", ".+i.+"]
        )


def test_name_sep_wrong_type(df_checks):
    """Raise TypeError if the wrong type is provided for names_sep."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(names_to=[".value", "num"], names_sep=["_"])


def test_name_sep_no_names_to(df_checks):
    """Raise ValuError if names_sep and names_to is None."""
    with pytest.raises(ValueError):
        df_checks.pivot_longer(names_to=None, names_sep="_")


def test_values_to_wrong_type(df_checks):
    """Raise TypeError if the wrong type is provided for `values_to`."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(values_to=["salvo"])


def test_values_to_exists_in_columns(df_checks):
    """
    Raise ValueError if values_to already
    exists in the dataframe's columns.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(values_to="birth")


def test_values_to_exists_in_names_to(df_checks):
    """
    Raise ValueError if values_to is in names_to.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(values_to="num", names_to="num")


def test_column_multiindex_names_sep(df_multi):
    """
    Raise ValueError if the dataframe's column is a MultiIndex,
    and names_sep is present.
    """
    with pytest.raises(ValueError):
        df_multi.pivot_longer(
            column_names=[("names", "aa")],
            names_sep="_",
            names_to=["names", "others"],
        )


def test_column_multiindex_names_pattern(df_multi):
    """
    Raise ValueError if the dataframe's column is a MultiIndex,
    and names_pattern is present.
    """
    with pytest.raises(ValueError):
        df_multi.pivot_longer(
            index=[("name", "a")],
            names_pattern=r"(.+)(.+)",
            names_to=["names", "others"],
        )


def test_index_tuple_multiindex(df_multi):
    """
    Raise ValueError if index is a tuple,
    instead of a list of tuples,
    and the dataframe's column is a MultiIndex.
    """
    with pytest.raises(ValueError):
        df_multi.pivot_longer(index=("name", "a"))


def test_column_names_tuple_multiindex(df_multi):
    """
    Raise ValueError if column_names is a tuple,
    instead of a list of tuples,
    and the dataframe's column is a MultiIndex.
    """
    with pytest.raises(ValueError):
        df_multi.pivot_longer(column_names=("names", "aa"))


def test_sort_by_appearance(df_checks):
    """Raise error if sort_by_appearance is not boolean."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(
            names_to=[".value", "value"],
            names_sep="_",
            sort_by_appearance="TRUE",
        )


def test_ignore_index(df_checks):
    """Raise error if ignore_index is not boolean."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(
            names_to=[".value", "value"], names_sep="_", ignore_index="TRUE"
        )


def test_names_to_index(df_checks):
    """
    Raise ValueError if there is no names_sep/names_pattern,
    .value not in names_to and names_to intersects with index.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to="famid",
            index="famid",
        )


def test_names_sep_pattern_names_to_index(df_checks):
    """
    Raise ValueError if names_sep/names_pattern,
    .value not in names_to and names_to intersects with index.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["dim", "famid"],
            names_sep="_",
            index="famid",
        )


def test_dot_value_names_to_columns_intersect(df_checks):
    """
    Raise ValueError if names_sep/names_pattern,
    .value in names_to,
    and names_to intersects with the new columns
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            index="famid", names_to=(".value", "ht"), names_pattern="(.+)(.)"
        )


def test_dot_value_names_to_index_intersect(df_checks):
    """
    Raise ValueError if names_sep/names_pattern,
    .value in names_to,
    and names_to intersects with the index
    """
    with pytest.raises(ValueError):
        df_checks.rename(columns={"famid": "ht"}).pivot_longer(
            index="ht", names_to=(".value", "num"), names_pattern="(.+)(.)"
        )


def test_names_pattern_list_empty_any(df_checks):
    """
    Raise ValueError if names_pattern is a list,
    and not all matches are returned.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            index=["famid", "birth"],
            names_to=["ht"],
            names_pattern=["rar"],
        )


def test_names_pattern_no_match(df_checks):
    """Raise error if names_pattern is a regex and returns no matches."""
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            index="famid",
            names_to=[".value", "value"],
            names_pattern=r"(rar)(.)",
        )


def test_names_pattern_incomplete_match(df_checks):
    """
    Raise error if names_pattern is a regex
    and returns incomplete matches.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            index="famid",
            names_to=[".value", "value"],
            names_pattern=r"(ht)(.)",
        )


def test_names_sep_len(df_checks):
    """
    Raise error if names_sep,
    and the number of  matches returned
    is not equal to the length of names_to.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(names_to=".value", names_sep="(\\d)")
