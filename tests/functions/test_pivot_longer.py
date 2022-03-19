import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal


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
    with pytest.raises(TypeError, match="entry in names_to.+"):
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
    with pytest.raises(ValueError):
        df_checks.pivot_longer(names_to=None, names_pattern="(.+)(.)")


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
    with pytest.raises(TypeError, match="entry in names_pattern.+"):
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
        ValueError, match="salvo in values_to already exists in names_to."
    ):
        df_checks.pivot_longer(
            values_to=["salvo"], names_pattern=["ht"], names_to="salvo"
        )


def test_sub_values_to(df_checks):
    """Raise error if values_to is a sequence, and contains non strings."""
    with pytest.raises(TypeError, match="entry in values_to.+"):
        df_checks.pivot_longer(
            names_to=["x", "y"],
            names_pattern=[r"ht", r"\d"],
            values_to=[1, "salvo"],
        )


def test_duplicate_values_to(df_checks):
    """Raise error if values_to is a sequence, and contains duplicates."""
    with pytest.raises(ValueError, match="salvo already exists in values_to."):
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


def test_values_to_seq_index_intersect(df_checks):
    """
    Raise ValueError if values_to is a sequence,
    and intersects with the index
    """
    with pytest.raises(
        ValueError,
        match="famid in values_to already exists as a column label.+",
    ):
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
    with pytest.raises(ValueError):
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
        column_names="ht*", names_to="dim", values_to="num", ignore_index=False
    )

    actual = df_checks.melt(
        ["famid", "birth"],
        var_name="dim",
        value_name="num",
        ignore_index=False,
    )

    assert_frame_equal(result, actual)


def test_pivot_sort_by_appearance(df_checks):
    """Test output if only sort_by_appearance is True."""
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

    assert_frame_equal(result, actual)


def test_multiindex_column_level(df_multi):
    """Test output from MultiIndex column"""
    result = df_multi.pivot_longer(
        index="name", column_names="names", column_level=0
    )
    expected_output = df_multi.melt(
        id_vars="name", value_vars="names", col_level=0
    )
    assert_frame_equal(result, expected_output)


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
        sort_by_appearance=True,
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
            suffix=r".+",
        )
        .reset_index("set")
        .reset_index(drop=True)
    )

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

    assert_frame_equal(result, actual)


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
    result = df.pivot_longer(index="id", names_to="yA", names_pattern=["(.+)"])

    assert_frame_equal(result, df.rename(columns={"x1": "yA"}))


def test_names_pattern_nulls_in_data():
    """Test output if nulls are present in data."""
    df = pd.DataFrame(
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

    result = df.pivot_longer(
        "family",
        names_to=[".value", "child"],
        names_pattern=r"(.+)_(.+)\d",
        ignore_index=False,
    )
    result.index = range(len(result))

    actual = (
        pd.wide_to_long(
            df, ["dob", "gender"], i="family", j="child", sep="_", suffix=".+"
        )
        .reset_index()
        .assign(child=lambda df: df.child.str[:-1])
    )

    assert_frame_equal(result, actual)


def test_output_values_to_seq():
    """Test output when values_to is a list/tuple."""
    df = pd.DataFrame(
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

    actual = df.melt(
        ["City", "State"],
        value_vars=["Mango", "Orange", "Watermelon"],
        var_name="Fruit",
        value_name="Pounds",
    )

    expected = df.pivot_longer(
        index=["City", "State"],
        column_names=slice("Mango", "Watermelon"),
        names_to=("Fruit"),
        values_to=("Pounds",),
        names_pattern=[r"M|O|W"],
    )

    assert_frame_equal(expected, actual)
