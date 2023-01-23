import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from pandas import NA


df = [1, 2, 3]


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


@pytest.mark.xfail(reason="checking is done within _select_columns")
def test_type_index(df_checks):
    """Raise TypeError if wrong type is provided for the index."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(index=2007)


@pytest.mark.xfail(reason="checking is done within _select_columns")
def test_type_column_names(df_checks):
    """Raise TypeError if wrong type is provided for column_names."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(column_names=2007)


def test_type_names_to(df_checks):
    """Raise TypeError if wrong type is provided for names_to."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(names_to={2007})


def test_type_dropna(df_checks):
    """Raise TypeError if wrong type is provided for dropna."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(dropna="True")


def test_subtype_names_to(df_checks):
    """
    Raise TypeError if names_to is a sequence
    and the wrong type is provided for entries
    in names_to.
    """
    with pytest.raises(TypeError, match="'1' in names_to.+"):
        df_checks.pivot_longer(names_to=[1])


def test_duplicate_names_to(df_checks):
    """Raise error if names_to contains duplicates."""
    with pytest.raises(ValueError, match="'y' is duplicated in names_to."):
        df_checks.pivot_longer(names_to=["y", "y"], names_pattern="(.+)(.)")


def test_both_names_sep_and_pattern(df_checks):
    """
    Raise ValueError if both names_sep
    and names_pattern is provided.
    """
    with pytest.raises(
        ValueError,
        match="Only one of names_pattern or names_sep should be provided.",
    ):
        df_checks.pivot_longer(
            names_to=["rar", "bar"], names_sep="-", names_pattern="(.+)(.)"
        )


def test_name_pattern_wrong_type(df_checks):
    """Raise TypeError if the wrong type provided for names_pattern."""
    with pytest.raises(TypeError, match="names_pattern should be one of.+"):
        df_checks.pivot_longer(names_to=["rar", "bar"], names_pattern=2007)


def test_name_pattern_no_names_to(df_checks):
    """Raise ValueError if names_pattern and names_to is None."""
    with pytest.raises(
        ValueError, match="Kindly provide values for names_to."
    ):
        df_checks.pivot_longer(names_to=None, names_pattern="(.+)(.)")


def test_name_pattern_seq_no_names_to(df_checks):
    """Raise ValueError if names_pattern is a sequence and names_to is None."""
    with pytest.raises(
        ValueError, match="Kindly provide values for names_to."
    ):
        df_checks.pivot_longer(names_to=None, names_pattern=[".{2}", "\\d"])


def test_name_pattern_groups_len(df_checks):
    """
    Raise ValueError if names_pattern
    and the number of groups
    differs from the length of names_to.
    """
    with pytest.raises(
        ValueError,
        match="The length of names_to does not match "
        "the number of groups in names_pattern.+",
    ):
        df_checks.pivot_longer(names_to=".value", names_pattern="(.+)(.)")


def test_names_pattern_wrong_subtype(df_checks):
    """
    Raise TypeError if names_pattern is a list/tuple
    and wrong subtype is supplied.
    """
    with pytest.raises(TypeError, match="'1' in names_pattern.+"):
        df_checks.pivot_longer(
            names_to=["ht", "num"], names_pattern=[1, "\\d"]
        )


def test_names_pattern_names_to_unequal_length(df_checks):
    """
    Raise ValueError if names_pattern is a list/tuple
    and wrong number of items in names_to.
    """
    with pytest.raises(
        ValueError,
        match="The length of names_to does not match "
        "the number of regexes in names_pattern.+",
    ):
        df_checks.pivot_longer(
            names_to=["variable"], names_pattern=["^ht", ".+i.+"]
        )


def test_names_pattern_names_to_dot_value(df_checks):
    """
    Raise Error if names_pattern is a list/tuple and
    .value in names_to.
    """
    with pytest.raises(
        ValueError,
        match=".value is not accepted in names_to "
        "if names_pattern is a list/tuple.",
    ):
        df_checks.pivot_longer(
            names_to=["variable", ".value"], names_pattern=["^ht", ".+i.+"]
        )


def test_name_sep_wrong_type(df_checks):
    """Raise TypeError if the wrong type is provided for names_sep."""
    with pytest.raises(TypeError, match="names_sep should be one of.+"):
        df_checks.pivot_longer(names_to=[".value", "num"], names_sep=["_"])


def test_name_sep_no_names_to(df_checks):
    """Raise ValueError if names_sep and names_to is None."""
    with pytest.raises(ValueError):
        df_checks.pivot_longer(names_to=None, names_sep="_")


def test_values_to_wrong_type(df_checks):
    """Raise TypeError if the wrong type is provided for `values_to`."""
    with pytest.raises(TypeError, match="values_to should be one of.+"):
        df_checks.pivot_longer(values_to={"salvo"})


def test_values_to_wrong_type_names_pattern(df_checks):
    """
    Raise TypeError if `values_to` is a list,
    and names_pattern is not.
    """
    with pytest.raises(
        TypeError,
        match="values_to can be a list/tuple only "
        "if names_pattern is a list/tuple.",
    ):
        df_checks.pivot_longer(values_to=["salvo"])


def test_values_to_names_pattern_unequal_length(df_checks):
    """
    Raise ValueError if `values_to` is a list,
    and the length of names_pattern
    does not match the length of values_to.
    """
    with pytest.raises(
        ValueError,
        match="The length of values_to does not match "
        "the number of regexes in names_pattern.+",
    ):
        df_checks.pivot_longer(
            values_to=["salvo"],
            names_pattern=["ht", r"\d"],
            names_to=["foo", "bar"],
        )


def test_values_to_names_seq_names_to(df_checks):
    """
    Raise ValueError if `values_to` is a list,
    and intersects with names_to.
    """
    with pytest.raises(
        ValueError, match="'salvo' in values_to already exists in names_to."
    ):
        df_checks.pivot_longer(
            values_to=["salvo"], names_pattern=["ht"], names_to="salvo"
        )


def test_sub_values_to(df_checks):
    """Raise error if values_to is a sequence, and contains non strings."""
    with pytest.raises(TypeError, match="1 in values_to.+"):
        df_checks.pivot_longer(
            names_to=["x", "y"],
            names_pattern=[r"ht", r"\d"],
            values_to=[1, "salvo"],
        )


def test_duplicate_values_to(df_checks):
    """Raise error if values_to is a sequence, and contains duplicates."""
    with pytest.raises(
        ValueError, match="'salvo' is duplicated in values_to."
    ):
        df_checks.pivot_longer(
            names_to=["x", "y"],
            names_pattern=[r"ht", r"\d"],
            values_to=["salvo", "salvo"],
        )


def test_values_to_exists_in_columns(df_checks):
    """
    Raise ValueError if values_to already
    exists in the dataframe's columns.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(index="birth", values_to="birth")


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


def test_column_names_missing_multiindex(df_multi):
    """
    Raise ValueError if column_names is a list of tuples,
    the dataframe's column is a MultiIndex,
    and the tuple cannot be found.
    """
    with pytest.raises(KeyError):
        df_multi.pivot_longer(column_names=[("names", "bb")])


def test_index_missing_multiindex(df_multi):
    """
    Raise ValueError if index is a list of tuples,
    the dataframe's column is a MultiIndex,
    and the tuple cannot be found.
    """
    with pytest.raises(KeyError):
        df_multi.pivot_longer(index=[("names", "bb")])


def test_column_names_not_all_tuples_multiindex(df_multi):
    """
    Raise ValueError if column_names is a list of tuples,
    the dataframe's column is a MultiIndex,
    and one of the entries is not a tuple.
    """
    with pytest.raises(TypeError):
        df_multi.pivot_longer(column_names=[("names", "aa"), "a"])


def test_index_not_all_tuples_multiindex(df_multi):
    """
    Raise ValueError if index is a list of tuples,
    the dataframe's column is a MultiIndex,
    and one of the entries is not a tuple.
    """
    with pytest.raises(TypeError):
        df_multi.pivot_longer(index=[("names", "aa"), "a"])


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
    with pytest.raises(
        ValueError,
        match=r".+in names_to already exist as column labels.+",
    ):
        df_checks.pivot_longer(
            names_to="famid",
            index="famid",
        )


def test_names_sep_pattern_names_to_index(df_checks):
    """
    Raise ValueError if names_sep/names_pattern,
    .value not in names_to and names_to intersects with index.
    """
    with pytest.raises(
        ValueError,
        match=r".+in names_to already exist as column labels.+",
    ):
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
    with pytest.raises(
        ValueError,
        match=r".+in names_to already exist in the new dataframe\'s columns.+",
    ):
        df_checks.pivot_longer(
            index="famid", names_to=(".value", "ht"), names_pattern="(.+)(.)"
        )


def test_values_to_seq_index_intersect(df_checks):
    """
    Raise ValueError if values_to is a sequence,
    and intersects with the index
    """
    match = ".+values_to already exist as column labels assigned "
    match = match + "to the dataframe's index parameter.+"
    with pytest.raises(ValueError, match=rf"{match}"):
        df_checks.pivot_longer(
            index="famid",
            names_to=("value", "ht"),
            names_pattern=["ht", r"\d"],
            values_to=("famid", "foo"),
        )


def test_dot_value_names_to_index_intersect(df_checks):
    """
    Raise ValueError if names_sep/names_pattern,
    .value in names_to,
    and names_to intersects with the index
    """
    match = ".+already exist as column labels assigned "
    match = match + "to the dataframe's index parameter.+"
    with pytest.raises(
        ValueError,
        match=rf"{match}",
    ):
        df_checks.rename(columns={"famid": "ht"}).pivot_longer(
            index="ht", names_to=(".value", "num"), names_pattern="(.+)(.)"
        )


def test_names_pattern_list_empty_any(df_checks):
    """
    Raise ValueError if names_pattern is a list,
    and not all matches are returned.
    """
    with pytest.raises(
        ValueError, match="No match was returned for the regex.+"
    ):
        df_checks.pivot_longer(
            index=["famid", "birth"],
            names_to=["ht"],
            names_pattern=["rar"],
        )


def test_names_pattern_no_match(df_checks):
    """Raise error if names_pattern is a regex and returns no matches."""
    with pytest.raises(
        ValueError, match="Column labels .+ could not be matched with any .+"
    ):
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
    with pytest.raises(
        ValueError, match="Column labels .+ could not be matched with any .+"
    ):
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


def test_pivot_index_only(df_checks):
    """Test output if only index is passed."""
    result = df_checks.pivot_longer(
        index=["famid", "birth"],
        names_to="dim",
        values_to="num",
    )

    actual = df_checks.melt(
        ["famid", "birth"], var_name="dim", value_name="num"
    )

    assert_frame_equal(result, actual)


def test_pivot_column_only(df_checks):
    """Test output if only column_names is passed."""
    result = df_checks.pivot_longer(
        column_names=["ht1", "ht2"],
        names_to="dim",
        values_to="num",
        ignore_index=False,
    )

    actual = df_checks.melt(
        ["famid", "birth"],
        var_name="dim",
        value_name="num",
        ignore_index=False,
    )

    assert_frame_equal(result, actual)


def test_pivot_sort_by_appearance(df_checks):
    """Test output if sort_by_appearance is True."""
    result = df_checks.pivot_longer(
        column_names="ht*",
        names_to="dim",
        values_to="num",
        sort_by_appearance=True,
    )

    actual = (
        df_checks.melt(
            ["famid", "birth"],
            var_name="dim",
            value_name="num",
            ignore_index=False,
        )
        .sort_index()
        .reset_index(drop=True)
    )

    assert_frame_equal(result, actual)


def test_names_pat_str(df_checks):
    """
    Test output when names_pattern is a string,
    and .value is present.
    """
    result = (
        df_checks.pivot_longer(
            column_names="ht*",
            names_to=(".value", "age"),
            names_pattern="(.+)(.)",
            sort_by_appearance=True,
        )
        .reindex(columns=["famid", "birth", "age", "ht"])
        .astype({"age": int})
    )

    actual = pd.wide_to_long(
        df_checks, stubnames="ht", i=["famid", "birth"], j="age"
    ).reset_index()

    assert_frame_equal(result, actual, check_dtype=False)


def test_multiindex_column_level(df_multi):
    """
    Test output from MultiIndex column,
    when column_level is provided.
    """
    result = df_multi.pivot_longer(
        index="name", column_names="names", column_level=0
    )
    expected_output = df_multi.melt(
        id_vars="name", value_vars="names", col_level=0
    )
    assert_frame_equal(result, expected_output)


def test_multiindex(df_multi):
    """
    Test output from MultiIndex column,
    where column_level is not provided,
    and there is no names_sep/names_pattern.
    """
    result = df_multi.pivot_longer(index=[("name", "a")])
    expected_output = df_multi.melt(id_vars=[("name", "a")])
    assert_frame_equal(result, expected_output)


def test_multiindex_names_to(df_multi):
    """
    Test output from MultiIndex column,
    where column_level is not provided,
    there is no names_sep/names_pattern,
    and names_to is provided as a sequence.
    """
    result = df_multi.pivot_longer(
        index=[("name", "a")], names_to=["variable_0", "variable_1"]
    )
    expected_output = df_multi.melt(id_vars=[("name", "a")])
    assert_frame_equal(result, expected_output)


def test_multiindex_names_to_length_mismatch(df_multi):
    """
    Raise error if the length of names_to does not
    match the number of column levels.
    """
    with pytest.raises(ValueError):
        df_multi.pivot_longer(
            index=[("name", "a")],
            names_to=["variable_0", "variable_1", "variable_2"],
        )


def test_multiindex_incomplete_level_names(df_multi):
    """
    Raise error if not all the levels have names.
    """
    with pytest.raises(ValueError):
        df_multi.columns.names = [None, "a"]
        df_multi.pivot_longer(index=[("name", "a")])


def test_multiindex_index_level_names_intersection(df_multi):
    """
    Raise error if level names exist in index.
    """
    with pytest.raises(ValueError):
        df_multi.columns.names = [None, "a"]
        df_multi.pivot_longer(index=[("name", "a")])


def test_no_column_names(df_checks):
    """
    Test output if all the columns
    are assigned to the index parameter.
    """
    assert_frame_equal(
        df_checks.pivot_longer(df_checks.columns).rename_axis(columns=None),
        df_checks,
    )


@pytest.fixture
def test_df():
    """Fixture DataFrame"""
    return pd.DataFrame(
        {
            "off_loc": ["A", "B", "C", "D", "E", "F"],
            "pt_loc": ["G", "H", "I", "J", "K", "L"],
            "pt_lat": [
                100.07548220000001,
                75.191326,
                122.65134479999999,
                124.13553329999999,
                124.13553329999999,
                124.01028909999998,
            ],
            "off_lat": [
                121.271083,
                75.93845266,
                135.043791,
                134.51128400000002,
                134.484374,
                137.962195,
            ],
            "pt_long": [
                4.472089953,
                -144.387785,
                -40.45611048,
                -46.07156181,
                -46.07156181,
                -46.01594293,
            ],
            "off_long": [
                -7.188632000000001,
                -143.2288569,
                21.242563,
                40.937416999999996,
                40.78472,
                22.905889000000002,
            ],
        }
    )


def test_names_pattern_str(test_df):
    """Test output for names_pattern and .value."""

    result = test_df.pivot_longer(
        column_names="*_*",
        names_to=["set", ".value"],
        names_pattern="(.+)_(.+)",
        sort_by_appearance=False,
    )
    result = result.sort_values(result.columns.tolist(), ignore_index=True)

    actual = test_df.copy()
    actual.columns = actual.columns.str.split("_").str[::-1].str.join("_")
    actual = (
        pd.wide_to_long(
            actual.reset_index(),
            stubnames=["loc", "lat", "long"],
            sep="_",
            i="index",
            j="set",
            suffix=r".+",
        )
        .reset_index("set")
        .reset_index(drop=True)
    )
    actual = actual.sort_values(actual.columns.tolist(), ignore_index=True)

    assert_frame_equal(result, actual)


def test_names_sep(test_df):
    """Test output for names_sep and .value."""

    result = test_df.pivot_longer(
        names_to=["set", ".value"], names_sep="_", sort_by_appearance=True
    )

    actual = test_df.copy()
    actual.columns = actual.columns.str.split("_").str[::-1].str.join("_")
    actual = (
        pd.wide_to_long(
            actual.reset_index(),
            stubnames=["loc", "lat", "long"],
            sep="_",
            i="index",
            j="set",
            suffix=".+",
        )
        .reset_index("set")
        .reset_index(drop=True)
    )

    result = result.sort_values(result.columns.tolist(), ignore_index=True)
    actual = actual.sort_values(actual.columns.tolist(), ignore_index=True)

    assert_frame_equal(result, actual)


def test_names_pattern_list():
    """Test output for names_pattern if list/tuple."""

    df = pd.DataFrame(
        {
            "Activity": ["P1", "P2"],
            "General": ["AA", "BB"],
            "m1": ["A1", "B1"],
            "t1": ["TA1", "TB1"],
            "m2": ["A2", "B2"],
            "t2": ["TA2", "TB2"],
            "m3": ["A3", "B3"],
            "t3": ["TA3", "TB3"],
        }
    )

    result = df.pivot_longer(
        index=["Activity", "General"],
        names_pattern=["^m", "^t"],
        names_to=["M", "Task"],
        sort_by_appearance=True,
    ).loc[:, ["Activity", "General", "Task", "M"]]

    actual = (
        pd.wide_to_long(
            df, i=["Activity", "General"], stubnames=["t", "m"], j="number"
        )
        .set_axis(["Task", "M"], axis="columns")
        .droplevel(-1)
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_names_pattern_dict():
    """Test output for names_pattern if dict."""

    df = pd.DataFrame(
        {
            "Activity": ["P1", "P2"],
            "General": ["AA", "BB"],
            "m1": ["A1", "B1"],
            "t1": ["TA1", "TB1"],
            "m2": ["A2", "B2"],
            "t2": ["TA2", "TB2"],
            "m3": ["A3", "B3"],
            "t3": ["TA3", "TB3"],
        }
    )

    result = df.pivot_longer(
        index=["Activity", "General"],
        names_pattern={"M": "^m", "Task": "^t"},
        sort_by_appearance=True,
    ).loc[:, ["Activity", "General", "Task", "M"]]

    actual = (
        pd.wide_to_long(
            df, i=["Activity", "General"], stubnames=["t", "m"], j="number"
        )
        .set_axis(["Task", "M"], axis="columns")
        .droplevel(-1)
        .reset_index()
    )

    assert_frame_equal(result, actual)


@pytest.fixture
def not_dot_value():
    """Fixture DataFrame"""
    return pd.DataFrame(
        {
            "country": ["United States", "Russia", "China"],
            "vault_2012": [48.1, 46.4, 44.3],
            "floor_2012": [45.4, 41.6, 40.8],
            "vault_2016": [46.9, 45.7, 44.3],
            "floor_2016": [46.0, 42.0, 42.1],
        }
    )


def test_not_dot_value_sep(not_dot_value):
    """Test output when names_sep and no dot_value"""

    result = not_dot_value.pivot_longer(
        "country",
        names_to=("event", "year"),
        names_sep="_",
        values_to="score",
        sort_by_appearance=True,
    )
    result = result.sort_values(
        ["country", "event", "year"], ignore_index=True
    )
    actual = not_dot_value.set_index("country")
    actual.columns = actual.columns.str.split("_", expand=True)
    actual.columns.names = ["event", "year"]
    actual = (
        actual.stack(["event", "year"])
        .rename("score")
        .sort_index()
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_not_dot_value_sep2(not_dot_value):
    """Test output when names_sep and no dot_value"""

    result = not_dot_value.pivot_longer(
        "country",
        names_to="event",
        names_sep="/",
        values_to="score",
    )

    actual = not_dot_value.melt(
        "country", var_name="event", value_name="score"
    )

    assert_frame_equal(result, actual)


def test_not_dot_value_pattern(not_dot_value):
    """Test output when names_pattern is a string and no dot_value"""

    result = not_dot_value.pivot_longer(
        "country",
        names_to=("event", "year"),
        names_pattern=r"(.+)_(.+)",
        values_to="score",
        sort_by_appearance=True,
    )
    result = result.sort_values(
        ["country", "event", "year"], ignore_index=True
    )
    actual = not_dot_value.set_index("country")
    actual.columns = actual.columns.str.split("_", expand=True)
    actual.columns.names = ["event", "year"]
    actual = (
        actual.stack(["event", "year"])
        .rename("score")
        .sort_index()
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_not_dot_value_pattern_named_groups(not_dot_value):
    """
    Test output when names_pattern has named groups
    """

    result = not_dot_value.pivot_longer(
        "country",
        names_pattern=r"(?P<event>.+)_(?P<year>.+)",
        values_to="score",
        sort_by_appearance=True,
    )
    result = result.sort_values(
        ["country", "event", "year"], ignore_index=True
    )
    actual = not_dot_value.set_index("country")
    actual.columns = actual.columns.str.split("_", expand=True)
    actual.columns.names = ["event", "year"]
    actual = (
        actual.stack(["event", "year"])
        .rename("score")
        .sort_index()
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_not_dot_value_sep_single_column(not_dot_value):
    """
    Test output when names_sep and no dot_value
    for a single column.
    """

    A = not_dot_value.loc[:, ["country", "vault_2012"]]
    result = A.pivot_longer(
        "country",
        names_to=("event", "year"),
        names_sep="_",
        values_to="score",
    )
    result = result.sort_values(
        ["country", "event", "year"], ignore_index=True
    )
    actual = A.set_index("country")
    actual.columns = actual.columns.str.split("_", expand=True)
    actual.columns.names = ["event", "year"]
    actual = (
        actual.stack(["event", "year"])
        .rename("score")
        .sort_index()
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_multiple_dot_value():
    """Test output for multiple .value."""
    df = pd.DataFrame(
        {
            "x_1_mean": [1, 2, 3, 4],
            "x_2_mean": [1, 1, 0, 0],
            "x_1_sd": [0, 1, 1, 1],
            "x_2_sd": [0.739, 0.219, 1.46, 0.918],
            "y_1_mean": [1, 2, 3, 4],
            "y_2_mean": [1, 1, 0, 0],
            "y_1_sd": [0, 1, 1, 1],
            "y_2_sd": [-0.525, 0.623, -0.705, 0.662],
            "unit": [1, 2, 3, 4],
        }
    )

    result = df.pivot_longer(
        index="unit",
        names_to=(".value", "time", ".value"),
        names_pattern=r"(x|y)_([0-9])(_mean|_sd)",
    ).astype({"time": int})

    actual = df.set_index("unit")
    cols = [ent.split("_") for ent in actual.columns]
    actual.columns = [f"{start}_{end}{middle}" for start, middle, end in cols]
    actual = (
        pd.wide_to_long(
            actual.reset_index(),
            stubnames=["x_mean", "y_mean", "x_sd", "y_sd"],
            i="unit",
            j="time",
        )
        .sort_index(axis=1)
        .reset_index()
    )

    assert_frame_equal(result, actual, check_dtype=False)


def test_multiple_dot_value_named_group():
    """Test output for multiple .value."""
    df = pd.DataFrame(
        {
            "x_1_mean": [1, 2, 3, 4],
            "x_2_mean": [1, 1, 0, 0],
            "x_1_sd": [0, 1, 1, 1],
            "x_2_sd": [0.739, 0.219, 1.46, 0.918],
            "y_1_mean": [1, 2, 3, 4],
            "y_2_mean": [1, 1, 0, 0],
            "y_1_sd": [0, 1, 1, 1],
            "y_2_sd": [-0.525, 0.623, -0.705, 0.662],
            "unit": [1, 2, 3, 4],
        }
    )

    result = df.pivot_longer(
        index="unit",
        names_pattern=r"(?P<_>x|y)_(?P<time>[0-9])(?P<__>_mean|_sd)",
    ).astype({"time": int})

    actual = df.set_index("unit")
    cols = [ent.split("_") for ent in actual.columns]
    actual.columns = [f"{start}_{end}{middle}" for start, middle, end in cols]
    actual = (
        pd.wide_to_long(
            actual.reset_index(),
            stubnames=["x_mean", "y_mean", "x_sd", "y_sd"],
            i="unit",
            j="time",
        )
        .sort_index(axis=1)
        .reset_index()
    )

    assert_frame_equal(result, actual, check_dtype=False)


@pytest.fixture
def single_val():
    """fixture dataframe"""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x1": [4, 5, 6],
            "x2": [5, 6, 7],
        }
    )


def test_multiple_dot_value2(single_val):
    """Test output for multiple .value."""

    result = single_val.pivot_longer(
        index="id", names_to=(".value", ".value"), names_pattern="(.)(.)"
    )

    assert_frame_equal(result, single_val)


def test_multiple_dot_value2_named_group(single_val):
    """Test output for multiple .value."""

    result = single_val.pivot_longer(
        index="id", names_pattern="(?P<_>.)(?P<_____________>.)"
    )

    assert_frame_equal(result, single_val)


def test_names_pattern_sequence_single_unique_column(single_val):
    """
    Test output if names_pattern is a sequence of length 1.
    """

    result = single_val.pivot_longer(
        "id", names_to=["x"], names_pattern=("x",)
    )
    actual = (
        pd.wide_to_long(single_val, ["x"], i="id", j="num")
        .droplevel("num")
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_names_pattern_single_column(single_val):
    """
    Test output if names_to is only '.value'.
    """

    result = single_val.pivot_longer(
        "id", names_to=".value", names_pattern="(.)."
    )
    actual = (
        pd.wide_to_long(single_val, ["x"], i="id", j="num")
        .droplevel("num")
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_names_pattern_single_column_not_dot_value(single_val):
    """
    Test output if names_to is not '.value'.
    """
    df = single_val[["x1"]]
    result = df.pivot_longer(names_to="yA", names_pattern="(.+)")

    assert_frame_equal(result, df.melt(var_name="yA"))


def test_names_pattern_single_column_not_dot_value1(single_val):
    """
    Test output if names_to is not '.value'.
    """
    df = single_val[["id", "x1"]]
    result = df.pivot_longer(index="id", names_to="yA", names_pattern="(.+)")

    assert_frame_equal(result, df.melt("id", var_name="yA"))


def test_names_pattern_seq_single_column(single_val):
    """
    Test output if names_pattern is a list.
    """
    df = single_val[["id", "x1"]]
    result = df.pivot_longer(index="id", names_to="yA", names_pattern=[".+"])

    assert_frame_equal(result, df.rename(columns={"x1": "yA"}))


@pytest.fixture
def df_null():
    "Dataframe with nulls."
    return pd.DataFrame(
        {
            "family": [1, 2, 3, 4, 5],
            "dob_child1": [
                "1998-11-26",
                "1996-06-22",
                "2002-07-11",
                "2004-10-10",
                "2000-12-05",
            ],
            "dob_child2": [
                "2000-01-29",
                np.nan,
                "2004-04-05",
                "2009-08-27",
                "2005-02-28",
            ],
            "gender_child1": [1, 2, 2, 1, 2],
            "gender_child2": [2.0, np.nan, 2.0, 1.0, 1.0],
        }
    )


def test_names_pattern_nulls_in_data(df_null):
    """Test output if nulls are present in data."""
    result = df_null.pivot_longer(
        "family",
        names_to=[".value", "child"],
        names_pattern=r"(.+)_(.+)",
        ignore_index=True,
    )

    actual = pd.wide_to_long(
        df_null, ["dob", "gender"], i="family", j="child", sep="_", suffix=".+"
    ).reset_index()

    assert_frame_equal(result, actual)


def test_dropna_multiple_columns(df_null):
    """Test output if dropna = True."""
    result = df_null.pivot_longer(
        "family",
        names_to=[".value", "child"],
        names_pattern=r"(.+)_(.+)",
        ignore_index=True,
        dropna=True,
    )

    actual = (
        pd.wide_to_long(
            df_null,
            ["dob", "gender"],
            i="family",
            j="child",
            sep="_",
            suffix=".+",
        )
        .dropna()
        .reset_index()
    )

    assert_frame_equal(result, actual)


def test_dropna_single_column():
    """
    Test output if dropna = True,
    and a single value column is returned.
    """
    df = pd.DataFrame(
        [
            {"a": 1.0, "b": np.nan, "c": np.nan, "d": np.nan},
            {"a": np.nan, "b": 2.0, "c": np.nan, "d": np.nan},
            {"a": np.nan, "b": np.nan, "c": 3.0, "d": 2.0},
            {"a": np.nan, "b": np.nan, "c": 1.0, "d": np.nan},
        ]
    )

    expected = df.pivot_longer(dropna=True)
    actual = df.melt().dropna().reset_index(drop=True)
    assert_frame_equal(expected, actual)


@pytest.fixture
def multiple_values_to():
    """fixture for multiple values_to"""
    # https://stackoverflow.com/q/51519101/7175713
    return pd.DataFrame(
        {
            "City": ["Houston", "Austin", "Hoover"],
            "State": ["Texas", "Texas", "Alabama"],
            "Name": ["Aria", "Penelope", "Niko"],
            "Mango": [4, 10, 90],
            "Orange": [10, 8, 14],
            "Watermelon": [40, 99, 43],
            "Gin": [16, 200, 34],
            "Vodka": [20, 33, 18],
        },
        columns=[
            "City",
            "State",
            "Name",
            "Mango",
            "Orange",
            "Watermelon",
            "Gin",
            "Vodka",
        ],
    )


def test_names_pattern_dict_names_to(multiple_values_to):
    """
    Raise Error if names_pattern is a dict
    and one of the keys exists in the index
    """
    with pytest.raises(
        ValueError, match="'City' in the names_pattern dictionary.+"
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={
                "City": {"Pounds": r"M|O|W"},
                "Drink": {"Ounces": r"G|V"},
            },
        )


def test_names_pattern_dict_index(multiple_values_to):
    """
    Raise Error if names_pattern is a dict
    and the key already exists in
    """
    with pytest.raises(
        ValueError,
        match="names_to should be None when names_pattern is a dictionary",
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_to=("Fruit", "Drink"),
            names_pattern={
                "Fruit": {"Pounds": r"M|O|W"},
                "Drink": {"Ounces": r"G|V"},
            },
        )


def test_names_pattern_dict_outer_key(multiple_values_to):
    """
    Raise Error if names_pattern is a dict
    and the outer key is not a string
    """
    with pytest.raises(TypeError, match="'1' in names_pattern.+"):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={
                1: {"Pounds": r"M|O|W"},
                "Drink": {"Ounces": r"G|V"},
            },
        )


def test_names_pattern_nested_dict_length(multiple_values_to):
    """
    Raise Error if names_pattern is a nested dict
    and the inner dictionary is not of length 1
    """
    with pytest.raises(
        ValueError,
        match="The length of the dictionary paired with 'Fruit' "
        "in names_pattern should be length 1, instead got 2",
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={"Fruit": {"Pounds": r"M|O|W", "Drink": r"G|V"}},
        )


def test_names_pattern_nested_dict_inner_key(multiple_values_to):
    """
    Raise Error if names_pattern is a nested dict
    and the inner key is not a string
    """
    with pytest.raises(
        TypeError,
        match="The key in the nested dictionary for 'Fruit' "
        "in names_pattern should be a string.+",
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={
                "Fruit": {1: r"M|O|W"},
                "Drink": {"Ounces": r"G|V"},
            },
        )


def test_names_pattern_nested_dict_inner_key_outer_key_dupe(
    multiple_values_to,
):
    """
    Raise Error if names_pattern is a nested dict
    and the inner key is a dupe of the outer key
    """
    with pytest.raises(
        ValueError,
        match="'Fruit' in the nested dictionary already exists "
        "as one of the main keys in names_pattern",
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={
                "Fruit": {"Pounds": r"M|O|W"},
                "Drink": {"Fruit": r"G|V"},
            },
        )


def test_names_pattern_nested_dict_inner_key_index(multiple_values_to):
    """
    Raise Error if names_pattern is a nested dict
    and the inner key already exists in the index
    """
    with pytest.raises(
        ValueError,
        match="'State' in the nested dictionary already exists "
        "as a column label assigned to the index parameter.+",
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={
                "Fruit": {"Pounds": r"M|O|W"},
                "Drink": {"State": r"G|V"},
            },
        )


def test_names_pattern_nested_dict_inner_value(multiple_values_to):
    """
    Raise Error if names_pattern is a nested dict
    and the inner value is not a string/regex
    """
    with pytest.raises(
        TypeError,
        match="The value paired with 'Pounds' in the nested dictionary "
        "in names_pattern.+",
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={
                "Fruit": {"Pounds": [r"M|O|W"]},
                "Drink": {"Ounces": r"G|V"},
            },
        )


def test_names_pattern_dict_value(multiple_values_to):
    """
    Raise Error if names_pattern is a dict
    and the value is not a string/regex
    """
    with pytest.raises(
        TypeError,
        match="The value paired with 'Fruit' "
        "in the names_pattern dictionary.+",
    ):
        multiple_values_to.pivot_longer(
            index=["City", "State"],
            column_names=slice("Mango", "Vodka"),
            names_pattern={"Fruit": [r"M|O|W"], "Drink": r"G|V"},
        )


def test_output_values_to_seq(multiple_values_to):
    """Test output when values_to is a list/tuple."""

    actual = multiple_values_to.melt(
        ["City", "State"],
        value_vars=["Mango", "Orange", "Watermelon"],
        var_name="Fruit",
        value_name="Pounds",
    )

    expected = multiple_values_to.pivot_longer(
        index=["City", "State"],
        column_names=slice("Mango", "Watermelon"),
        names_to=("Fruit"),
        values_to=("Pounds",),
        names_pattern=[r"M|O|W"],
    )

    assert_frame_equal(expected, actual)


def test_output_values_to_seq1(multiple_values_to):
    """Test output when values_to is a list/tuple."""
    # https://stackoverflow.com/a/51520155/7175713
    df1 = multiple_values_to.melt(
        id_vars=["City", "State"],
        value_vars=["Mango", "Orange", "Watermelon"],
        var_name="Fruit",
        value_name="Pounds",
    )
    df2 = multiple_values_to.melt(
        id_vars=["City", "State"],
        value_vars=["Gin", "Vodka"],
        var_name="Drink",
        value_name="Ounces",
    )

    df1 = df1.set_index(
        ["City", "State", df1.groupby(["City", "State"]).cumcount()]
    )
    df2 = df2.set_index(
        ["City", "State", df2.groupby(["City", "State"]).cumcount()]
    )

    actual = (
        pd.concat([df1, df2], axis=1)
        .sort_index(level=2)
        .reset_index(level=2, drop=True)
        .reset_index()
        .astype({"Fruit": "category", "Drink": "category"})
    )

    expected = multiple_values_to.pivot_longer(
        index=["City", "State"],
        column_names=slice("Mango", "Vodka"),
        names_to=("Fruit", "Drink"),
        values_to=("Pounds", "Ounces"),
        names_pattern=[r"M|O|W", r"G|V"],
        names_transform={"Fruit": "category", "Drink": "category"},
    ).sort_values(["Fruit", "City", "State"], ignore_index=True)

    assert_frame_equal(expected, actual)


def test_output_names_pattern_nested_dictionary(multiple_values_to):
    """Test output when names_pattern is a nested dictionary."""
    # https://stackoverflow.com/a/51520155/7175713
    df1 = multiple_values_to.melt(
        id_vars=["City", "State"],
        value_vars=["Mango", "Orange", "Watermelon"],
        var_name="Fruit",
        value_name="Pounds",
    )
    df2 = multiple_values_to.melt(
        id_vars=["City", "State"],
        value_vars=["Gin", "Vodka"],
        var_name="Drink",
        value_name="Ounces",
    )

    df1 = df1.set_index(
        ["City", "State", df1.groupby(["City", "State"]).cumcount()]
    )
    df2 = df2.set_index(
        ["City", "State", df2.groupby(["City", "State"]).cumcount()]
    )

    actual = (
        pd.concat([df1, df2], axis=1)
        .sort_index(level=2)
        .reset_index(level=2, drop=True)
        .reset_index()
        .astype({"Fruit": "category", "Drink": "category"})
    )

    expected = multiple_values_to.pivot_longer(
        index=["City", "State"],
        column_names=slice("Mango", "Vodka"),
        names_pattern={
            "Fruit": {"Pounds": r"M|O|W"},
            "Drink": {"Ounces": r"G|V"},
        },
        names_transform={"Fruit": "category", "Drink": "category"},
    ).sort_values(["Fruit", "City", "State"], ignore_index=True)

    assert_frame_equal(expected, actual)


def test_categorical(df_checks):
    """Test category output for names_to."""

    actual = df_checks.melt(["famid", "birth"]).astype(
        {"variable": "category"}
    )
    expected = df_checks.pivot_longer(
        ["famid", "birth"], names_transform="category"
    )

    assert_frame_equal(actual, expected, check_categorical=False)


def test_names_transform_numeric():
    """
    Test output for names_transform on numeric sub columns
    """

    df = pd.DataFrame(
        {
            "treatment_1.1": [1.0, 2.0],
            "treatment_2.1": [3.0, 4.0],
            "result_1.2": [5.0, 6.0],
            "result_1": [0, 9],
            "A": ["X1", "X2"],
        }
    )

    result = df.pivot_longer(
        index="A",
        names_to=(".value", "colname"),
        names_sep="_",
        names_transform=float,
    ).loc[:, ["A", "colname", "result", "treatment"]]

    actual = pd.wide_to_long(
        df,
        ["result", "treatment"],
        i="A",
        j="colname",
        suffix="[0-9.]+",
        sep="_",
    ).reset_index()

    result = result.sort_values(result.columns.tolist(), ignore_index=True)
    actual = actual.sort_values(actual.columns.tolist(), ignore_index=True)

    assert_frame_equal(actual, result)


def test_duplicated_columns():
    """Test output for duplicated columns."""
    rows = [["credit", 1, 1, 2, 3]]
    columns = ["Type", "amount", "active", "amount", "active"]

    df = pd.DataFrame(rows, columns=columns)
    df = df.set_index("Type")

    actual = pd.DataFrame(
        {"amount": [1, 2], "active": [1, 3]},
        index=pd.Index(["credit", "credit"], name="Type"),
    )
    expected = df.pivot_longer(
        names_to=".value", names_pattern="(.+)", ignore_index=False
    )

    assert_frame_equal(actual, expected)


def test_dot_value_duplicated_sub_columns():
    """Test output when the column extracts are not unique."""
    # https://stackoverflow.com/q/64061588/7175713
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "M_start_date_1": [201709, 201709, 201709],
            "M_end_date_1": [201905, 201905, 201905],
            "M_start_date_2": [202004, 202004, 202004],
            "M_end_date_2": [202005, 202005, 202005],
            "F_start_date_1": [201803, 201803, 201803],
            "F_end_date_1": [201904, 201904, 201904],
            "F_start_date_2": [201912, 201912, 201912],
            "F_end_date_2": [202007, 202007, 202007],
        }
    )

    expected = df.set_index("id")
    expected.columns = expected.columns.str.split("_", expand=True)
    expected = (
        expected.stack(level=[0, 2, 3])
        .sort_index(level=[0, 1], ascending=[True, False])
        .reset_index(level=[2, 3], drop=True)
        .sort_index(axis=1, ascending=False)
        .rename_axis(["id", "cod"])
        .reset_index()
    )

    actual = df.pivot_longer(
        "id",
        names_to=("cod", ".value"),
        names_pattern="(.)_(start|end).+",
        sort_by_appearance=True,
    )

    assert_frame_equal(actual, expected)


def test_preserve_extension_types():
    """Preserve extension types where possible."""
    cats = pd.DataFrame(
        [
            {"Cat": "A", "L_1": 1, "L_2": 2, "L_3": 3},
            {"Cat": "B", "L_1": 4, "L_2": 5, "L_3": 6},
            {"Cat": "C", "L_1": 7, "L_2": 8, "L_3": 9},
        ]
    )
    cats = cats.astype("category")

    actual = cats.pivot_longer("Cat", sort_by_appearance=True)
    expected = (
        cats.set_index("Cat")
        .rename_axis(columns="variable")
        .stack()
        .rename("value")
        .reset_index()
    )

    assert_frame_equal(expected, actual)


def test_dropna_sort_by_appearance():
    """
    Test output when `dropna=True` and
    `sort_by_appearance=True`
    """
    # GH PR #1169, Issue #1168

    treatments = dict(
        id=range(1, 6),
        A=("A", NA, "A", NA, NA),
        A_date=(1, NA, 2, NA, NA),
        B=(NA, "B", "B", NA, NA),
        B_date=(NA, 3, 2, NA, NA),
        other=(NA, NA, NA, "C", "D"),
        other_date=(NA, NA, NA, 1, 5),
    )
    treatments = pd.DataFrame(treatments)
    actual = treatments.pivot_longer(
        index="id",
        names_to=["date", "treatment"],
        names_pattern=[".+date$", ".+"],
        dropna=True,
        sort_by_appearance=True,
    )

    expected = pd.lreshape(
        treatments,
        {
            "treatment": ["A", "B", "other"],
            "date": ["A_date", "B_date", "other_date"],
        },
    ).sort_values(["id", "treatment", "date"], ignore_index=True)

    assert_frame_equal(actual, expected)
