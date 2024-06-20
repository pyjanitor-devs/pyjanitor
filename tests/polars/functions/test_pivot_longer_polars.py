import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal

import janitor.polars  # noqa: F401


@pytest.fixture
def df_checks():
    """fixture dataframe"""
    return pl.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
        }
    )


def test_type_names_to(df_checks):
    """Raise TypeError if wrong type is provided for names_to."""
    msg = "names_to should be a string, list, or tuple.+"
    with pytest.raises(TypeError, match=msg):
        df_checks.pivot_longer(names_to=2007, names_sep="_")


def test_subtype_names_to(df_checks):
    """
    Raise TypeError if names_to is a sequence
    and the wrong type is provided for entries
    in names_to.
    """
    with pytest.raises(TypeError, match="'1' in names_to.+"):
        df_checks.pivot_longer(names_to=[1], names_sep="_")


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
    """Raise TypeError if the wrong type is provided for names_pattern."""
    with pytest.raises(TypeError, match="names_pattern should be one of.+"):
        df_checks.pivot_longer(names_to=["rar", "bar"], names_pattern=2007)


def test_name_sep_wrong_type(df_checks):
    """Raise TypeError if the wrong type is provided for names_sep."""
    with pytest.raises(TypeError, match="names_sep should be one of.+"):
        df_checks.pivot_longer(names_to=[".value", "num"], names_sep=["_"])


def test_values_to_wrong_type(df_checks):
    """Raise TypeError if the wrong type is provided for `values_to`."""
    with pytest.raises(TypeError, match="values_to should be one of.+"):
        df_checks.pivot_longer(values_to={"salvo"}, names_sep="_")


def test_names_to_names_pattern_len(df_checks):
    """ "
    Raise ValueError
    if length of names_to
    does not match the number of fields extracted
    """
    msg = "The length of names_to does "
    msg += "not match the number of fields extracted.+"
    with pytest.raises(ValueError, match=msg):

        df_checks.pivot_longer(
            column_names=cs.starts_with("ht"),
            names_to=(".value"),
            names_pattern=r"(\d+)(.)",
        )


def test_names_to_names_pattern_mismatch(df_checks):
    """ "
    Raise ValueError if
    there is no match for the provided names_pattern
    """
    msg = "Column label 'ht1' could not be matched with "
    msg += "any of the groups in the provided regex.+"

    with pytest.raises(ValueError, match=msg):

        df_checks.pivot_longer(
            column_names=cs.starts_with("ht"),
            names_to=(".value", "age"),
            names_pattern=r"(\d+)(.)",
        )


def test_names_pat_str(df_checks):
    """
    Test output when names_pattern is a string,
    and .value is present.
    """
    result = (
        df_checks.pivot_longer(
            index=["famid", "birth"],
            names_to=(".value", "age"),
            names_pattern="(.+)(.)",
            names_transform=pl.col("age").cast(pl.Int64),
        )
        .select("famid", "birth", "age", "ht")
        .sort(by=pl.all())
    )

    actual = [
        {"famid": 1, "birth": 1, "age": 1, "ht": 2.8},
        {"famid": 1, "birth": 1, "age": 2, "ht": 3.4},
        {"famid": 1, "birth": 2, "age": 1, "ht": 2.9},
        {"famid": 1, "birth": 2, "age": 2, "ht": 3.8},
        {"famid": 1, "birth": 3, "age": 1, "ht": 2.2},
        {"famid": 1, "birth": 3, "age": 2, "ht": 2.9},
        {"famid": 2, "birth": 1, "age": 1, "ht": 2.0},
        {"famid": 2, "birth": 1, "age": 2, "ht": 3.2},
        {"famid": 2, "birth": 2, "age": 1, "ht": 1.8},
        {"famid": 2, "birth": 2, "age": 2, "ht": 2.8},
        {"famid": 2, "birth": 3, "age": 1, "ht": 1.9},
        {"famid": 2, "birth": 3, "age": 2, "ht": 2.4},
        {"famid": 3, "birth": 1, "age": 1, "ht": 2.2},
        {"famid": 3, "birth": 1, "age": 2, "ht": 3.3},
        {"famid": 3, "birth": 2, "age": 1, "ht": 2.3},
        {"famid": 3, "birth": 2, "age": 2, "ht": 3.4},
        {"famid": 3, "birth": 3, "age": 1, "ht": 2.1},
        {"famid": 3, "birth": 3, "age": 2, "ht": 2.9},
    ]
    actual = pl.DataFrame(actual).sort(by=pl.all())

    assert_frame_equal(result, actual)


@pytest.fixture
def test_df():
    """Fixture DataFrame"""
    return pl.DataFrame(
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


actual = [
    {
        "set": "off",
        "loc": "A",
        "lat": 121.271083,
        "long": -7.188632000000001,
    },
    {"set": "off", "loc": "B", "lat": 75.93845266, "long": -143.2288569},
    {"set": "off", "loc": "C", "lat": 135.043791, "long": 21.242563},
    {
        "set": "off",
        "loc": "D",
        "lat": 134.51128400000002,
        "long": 40.937416999999996,
    },
    {"set": "off", "loc": "E", "lat": 134.484374, "long": 40.78472},
    {
        "set": "off",
        "loc": "F",
        "lat": 137.962195,
        "long": 22.905889000000002,
    },
    {
        "set": "pt",
        "loc": "G",
        "lat": 100.07548220000001,
        "long": 4.472089953,
    },
    {"set": "pt", "loc": "H", "lat": 75.191326, "long": -144.387785},
    {
        "set": "pt",
        "loc": "I",
        "lat": 122.65134479999999,
        "long": -40.45611048,
    },
    {
        "set": "pt",
        "loc": "J",
        "lat": 124.13553329999999,
        "long": -46.07156181,
    },
    {
        "set": "pt",
        "loc": "K",
        "lat": 124.13553329999999,
        "long": -46.07156181,
    },
    {
        "set": "pt",
        "loc": "L",
        "lat": 124.01028909999998,
        "long": -46.01594293,
    },
]

actual = pl.DataFrame(actual).sort(by=pl.all())


def test_names_pattern_dot_value(test_df):
    """Test output for names_pattern and .value."""

    result = (
        test_df.pivot_longer(
            column_names=cs.all(),
            names_to=["set", ".value"],
            names_pattern="(.+)_(.+)",
        )
        .sort(by=["loc", "lat", "long"])
        .select("set", "loc", "lat", "long")
    )
    assert_frame_equal(result, actual)


def test_names_sep_dot_value(test_df):
    """Test output for names_pattern and .value."""

    result = (
        test_df.pivot_longer(
            column_names=cs.all(),
            names_to=["set", ".value"],
            names_sep="_",
        )
        .sort(by=["loc", "lat", "long"])
        .select("set", "loc", "lat", "long")
    )
    assert_frame_equal(result, actual)


@pytest.fixture
def not_dot_value():
    """Fixture DataFrame"""
    return pl.DataFrame(
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

    result = (
        not_dot_value.pivot_longer(
            "country",
            names_to=("event", "year"),
            names_sep="_",
            values_to="score",
        )
        .select("country", "event", "year", "score")
        .sort(by=pl.all())
    )

    actual = (
        not_dot_value.melt(id_vars="country", value_name="score")
        .select(
            "country",
            "score",
            pl.col("variable")
            .str.split("_")
            .list.to_struct(fields=["event", "year"]),
        )
        .unnest("variable")
        .select("country", "event", "year", "score")
        .sort(by=pl.all())
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
        "country", variable_name="event", value_name="score"
    )

    assert_frame_equal(result, actual)


def test_not_dot_value_pattern(not_dot_value):
    """Test output when names_pattern is a string and no dot_value"""

    result = (
        not_dot_value.pivot_longer(
            index="country",
            names_to=("event", "year"),
            names_pattern=r"(.+)_(.+)",
            values_to="score",
        )
        .select("country", "event", "year", "score")
        .sort(by=pl.all())
    )

    actual = (
        not_dot_value.melt(id_vars="country", value_name="score")
        .select(
            "country",
            "score",
            pl.col("variable")
            .str.split("_")
            .list.to_struct(fields=["event", "year"]),
        )
        .unnest("variable")
        .select("country", "event", "year", "score")
        .sort(by=pl.all())
    )

    assert_frame_equal(result, actual)


def test_multiple_dot_value():
    """Test output for multiple .value."""
    df = pl.DataFrame(
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

    result = (
        df.pivot_longer(
            index="unit",
            names_to=(".value", "time", ".value"),
            names_pattern=r"(x|y)_([0-9])(_mean|_sd)",
            names_transform=pl.col("time").cast(pl.Int64),
        )
        .select("unit", "time", "x_mean", "x_sd", "y_mean", "y_sd")
        .sort(by=pl.all())
    )

    actual = {
        "unit": [1, 2, 3, 4, 1, 2, 3, 4],
        "time": [1, 1, 1, 1, 2, 2, 2, 2],
        "x_mean": [1, 2, 3, 4, 1, 1, 0, 0],
        "x_sd": [0.0, 1.0, 1.0, 1.0, 0.739, 0.219, 1.46, 0.918],
        "y_mean": [1, 2, 3, 4, 1, 1, 0, 0],
        "y_sd": [0.0, 1.0, 1.0, 1.0, -0.525, 0.623, -0.705, 0.662],
    }

    actual = pl.DataFrame(actual).sort(by=pl.all())

    assert_frame_equal(result, actual)


@pytest.fixture
def single_val():
    """fixture dataframe"""
    return pl.DataFrame(
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


actual3 = [
    {"id": 1, "x": 4},
    {"id": 2, "x": 5},
    {"id": 3, "x": 6},
    {"id": 1, "x": 5},
    {"id": 2, "x": 6},
    {"id": 3, "x": 7},
]

actual3 = pl.DataFrame(actual3)


def test_names_pattern_single_column(single_val):
    """
    Test output if names_to is only '.value'.
    """

    result = single_val.pivot_longer(
        "id", names_to=".value", names_pattern="(.)."
    )

    assert_frame_equal(result.sort(by=pl.all()), actual3.sort(by=pl.all()))


def test_names_pattern_single_column_not_dot_value(single_val):
    """
    Test output if names_to is not '.value'.
    """
    result = single_val.pivot_longer(
        index="id", column_names="x1", names_to="yA", names_pattern="(.+)"
    ).select("id", "yA", "value")

    assert_frame_equal(
        result,
        single_val.melt(id_vars="id", value_vars="x1", variable_name="yA"),
    )


def test_names_pattern_single_column_not_dot_value1(single_val):
    """
    Test output if names_to is not '.value'.
    """
    result = (
        single_val.select("x1")
        .pivot_longer(names_to="yA", names_pattern="(.+)")
        .select("yA", "value")
    )

    assert_frame_equal(
        result,
        single_val.select("x1").melt(variable_name="yA"),
    )


@pytest.fixture
def df_null():
    "Dataframe with nulls."
    return pl.DataFrame(
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
                None,
                "2004-04-05",
                "2009-08-27",
                "2005-02-28",
            ],
            "gender_child1": [1, 2, 2, 1, 2],
            "gender_child2": [2.0, None, 2.0, 1.0, 1.0],
        }
    )


def test_names_pattern_nulls_in_data(df_null):
    """Test output if nulls are present in data."""
    result = (
        df_null.pivot_longer(
            index="family",
            names_to=[".value", "child"],
            names_pattern=r"(.+)_(.+)",
        )
        .select("family", "child", "dob", "gender")
        .sort(by=pl.all())
    )

    actual = [
        {"family": 1, "child": "child1", "dob": "1998-11-26", "gender": 1.0},
        {"family": 2, "child": "child1", "dob": "1996-06-22", "gender": 2.0},
        {"family": 3, "child": "child1", "dob": "2002-07-11", "gender": 2.0},
        {"family": 4, "child": "child1", "dob": "2004-10-10", "gender": 1.0},
        {"family": 5, "child": "child1", "dob": "2000-12-05", "gender": 2.0},
        {"family": 1, "child": "child2", "dob": "2000-01-29", "gender": 2.0},
        {"family": 2, "child": "child2", "dob": None, "gender": None},
        {"family": 3, "child": "child2", "dob": "2004-04-05", "gender": 2.0},
        {"family": 4, "child": "child2", "dob": "2009-08-27", "gender": 1.0},
        {"family": 5, "child": "child2", "dob": "2005-02-28", "gender": 1.0},
    ]

    actual = pl.DataFrame(actual).sort(by=pl.all())

    assert_frame_equal(result, actual)
