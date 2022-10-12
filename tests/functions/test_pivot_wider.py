import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df_checks_output():
    """pytest fixture"""
    return pd.DataFrame(
        {
            "geoid": [1, 1, 13, 13],
            "name": ["Alabama", "Alabama", "Georgia", "Georgia"],
            "variable": [
                "pop_renter",
                "median_rent",
                "pop_renter",
                "median_rent",
            ],
            "estimate": [1434765, 747, 3592422, 927],
            "error": [16736, 3, 33385, 3],
        }
    )


@pytest.mark.xfail(reason="list-like is converted to list.")
def test_type_index(df_checks_output):
    """Raise TypeError if wrong type is provided for the `index`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index={"geoid"}, names_from="variable")

    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index=("geoid", "name"), names_from="variable"
        )


@pytest.mark.xfail(reason="list-like is converted to list.")
def test_type_names_from(df_checks_output):
    """Raise TypeError if wrong type is provided for `names_from`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index="geoid", names_from={"variable"})

    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index="geoid", names_from=("variable",))


def test_names_from_none(df_checks_output):
    """Raise ValueError if no value is provided for `names_from`."""
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index="geoid", names_from=None)


def test_presence_index1(df_checks_output):
    """Raise KeyError if labels in `index` do not exist."""
    with pytest.raises(KeyError):
        df_checks_output.pivot_wider(index="geo", names_from="variable")


def test_presence_index2(df_checks_output):
    """Raise KeyError if labels in `index` do not exist."""
    with pytest.raises(KeyError):
        df_checks_output.pivot_wider(
            index=["geoid", "Name"], names_from="variable"
        )


def test_presence_names_from1(df_checks_output):
    """Raise KeyError if labels in `names_from` do not exist."""
    with pytest.raises(KeyError):
        df_checks_output.pivot_wider(index="geoid", names_from="estmt")


def test_presence_names_from2(df_checks_output):
    """Raise KeyError if labels in `names_from` do not exist."""
    with pytest.raises(KeyError):
        df_checks_output.pivot_wider(index="geoid", names_from=["estimat"])


def test_flatten_levels_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `flatten_levels`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=["estimate", "variable"],
            flatten_levels=2,
        )


def test_names_glue_wrong_label(df_checks_output):
    """Raise KeyError if the wrong column label is provided in `names_glue`."""
    with pytest.raises(
        KeyError, match="'variabl' is not a column label in names_from."
    ):
        df_checks_output.pivot_wider(
            index=["geoid", "name"],
            names_from="variable",
            values_from=["estimate", "error"],
            names_glue="{variabl}_{_value}",
        )


def test_names_glue_wrong_label1(df_checks_output):
    """
    Raise KeyError if the wrong column label is provided in `names_glue`,
    And the columns is a single Index.
    """
    with pytest.raises(
        KeyError, match="'variabl' is not a column label in names_from."
    ):
        df_checks_output.pivot_wider(
            ["geoid", "name"],
            "variable",
            "estimate",
            names_glue="{variabl}_estimate",
        )


def test_names_glue_wrong_label2(df_checks_output):
    """Raise Warning if _value is in `names_glue`."""
    with pytest.warns(UserWarning):
        df_checks_output.rename(columns={"variable": "_value"}).pivot_wider(
            index=["geoid", "name"],
            names_from="_value",
            values_from=["estimate", "error"],
            names_glue="{_value}_{_value}",
        )


def test_name_sep_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `names_sep`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], names_sep=1
        )


def test_name_expand_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `names_expand`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], names_expand=1
        )


def test_id_expand_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `id_expand`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], id_expand=1
        )


def test_reset_index_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `reset_index`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], reset_index=1
        )


def test_name_glue_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `names_glue`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], names_glue=1
        )


def test_non_unique_index_names_from_combination():
    """
    Raise ValueError for non-unique combination of
    `index` and `names_from`.
    """
    df = pd.DataFrame(
        {"A": ["A", "A", "A"], "L": ["L", "L", "L"], "numbers": [30, 54, 25]}
    )
    with pytest.raises(ValueError):
        df.pivot_wider(index="A", names_from="L")


def test_pivot_long_wide_long():
    """
    Test transformation from long to wide and back to long form.
    """

    df_in = pd.DataFrame(
        [
            {
                "a": 1,
                "b": 2,
                "name": "ben",
                "points": 22,
                "marks": 5,
                "sets": 13,
            },
            {
                "a": 1,
                "b": 2,
                "name": "dave",
                "points": 23,
                "marks": 4,
                "sets": 11,
            },
        ]
    )

    result = df_in.pivot_wider(
        index=["a", "b"], names_from="name", names_sep=None
    )

    result = result.pivot_longer(
        index=["a", "b"],
        names_to=(".value", "name"),
        names_sep="_",
    )
    assert_frame_equal(result, df_in)


@pytest.mark.xfail(reason="doesnt match, since pivot implicitly sorts")
def test_pivot_wide_long_wide():
    """
    Test that transformation from pivot_longer to wider and
    back to longer returns the same source dataframe.
    """
    df = pd.DataFrame(
        {
            "name": ["Wilbur", "Petunia", "Gregory"],
            "a": [67, 80, 64],
            "b": [56, 90, 50],
        }
    )

    result = df.pivot_longer(
        column_names=["a", "b"], names_to="drug", values_to="heartrate"
    )

    result = result.pivot_wider(
        index="name", names_from="drug", values_from="heartrate"
    )

    assert_frame_equal(result, df)


def test_flatten_levels_false():
    """Test output if `flatten_levels` is False."""

    df_collapse = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )

    result = df_collapse.pivot_wider(
        index="foo",
        names_from="bar",
        values_from=["baz", "zoo"],
        flatten_levels=False,
    )

    expected_output = df_collapse.pivot(  # noqa: PD010
        index="foo", columns="bar", values=["baz", "zoo"]
    )

    assert_frame_equal(
        result,
        expected_output,
    )


def test_no_index():
    """Test output if no `index` is supplied."""
    df_in = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Female", "Male", "Male"],
            "contVar": [22379, 24523, 23421, 23831, 29234],
        },
        index=pd.Int64Index([0, 0, 1, 1, 2], dtype="int64"),
    )

    expected_output = pd.DataFrame(
        {
            "contVar_Female": [24523.0, 23421.0, np.nan],
            "contVar_Male": [22379.0, 23831.0, 29234.0],
        }
    )

    result = df_in.pivot_wider(names_from="gender")

    assert_frame_equal(result, expected_output)


def test_no_index_names_from_order():
    """Test output if no `index` is supplied and column order is maintained."""
    df_in = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Female", "Male", "Male"],
            "contVar": [22379, 24523, 23421, 23831, 29234],
        },
        index=pd.Int64Index([0, 0, 1, 1, 2], dtype="int64"),
    )

    expected_output = pd.DataFrame(
        {
            "contVar_Male": [22379.0, 23831.0, 29234.0],
            "contVar_Female": [24523.0, 23421.0, np.nan],
        }
    )

    result = df_in.encode_categorical(gender="appearance").pivot_wider(
        names_from="gender"
    )

    assert_frame_equal(result, expected_output)


def test_index_names():
    """Test output if index is supplied."""
    df = pd.DataFrame(
        [
            {"stat": "mean", "score": 4, "var": "var1"},
            {"stat": "sd", "score": 7, "var": "var1"},
            {"stat": "mean", "score": 1, "var": "var2"},
            {"stat": "sd", "score": 2, "var": "var2"},
            {"stat": "mean", "score": 11, "var": "var3"},
            {"stat": "sd", "score": 14, "var": "var3"},
        ]
    )

    expected_output = pd.DataFrame(
        {"var": ["var1", "var2", "var3"], "mean": [4, 1, 11], "sd": [7, 2, 14]}
    )

    result = df.pivot_wider(
        index="var", names_from="stat", values_from="score"
    )

    assert_frame_equal(result, expected_output)


def test_categorical():
    """Test output for categorical column"""
    df_in = pd.DataFrame(
        {
            "family": ["Kelly", "Kelly", "Quin", "Quin"],
            "name": ["Mark", "Scott", "Tegan", "Sara"],
            "n": pd.Categorical([1, 2, 1, 2]),
        }
    )
    df_out = pd.DataFrame(
        {
            "family": ["Kelly", "Quin"],
            1: ["Mark", "Tegan"],
            2: ["Scott", "Sara"],
        }
    )

    result = df_in.pivot_wider(
        index="family",
        names_from="n",
        values_from="name",
    )
    assert_frame_equal(result, df_out)


def test_names_glue():
    """Test output with `names_glue`"""
    df_in = pd.DataFrame(
        {
            "family": ["Kelly", "Kelly", "Quin", "Quin"],
            "name": ["Mark", "Scott", "Tegan", "Sara"],
            "n": ["1", "2", "1", "2"],
        }
    )
    df_out = pd.DataFrame(
        {
            "family": ["Kelly", "Quin"],
            "name1": ["Mark", "Tegan"],
            "name2": ["Scott", "Sara"],
        }
    )

    result = df_in.pivot_wider(
        index="family",
        names_from="n",
        values_from="name",
        names_glue="name{n}",
    )
    assert_frame_equal(result, df_out)


def test_names_glue_multiple_levels(df_checks_output):
    """
    Test output with names_glue for multiple levels.
    """

    df_out = pd.DataFrame(
        {
            "geoid": [1, 13],
            "name": ["Alabama", "Georgia"],
            "pop_renter_estimate": [1434765, 3592422],
            "median_rent_estimate": [747, 927],
            "pop_renter_error": [16736, 33385],
            "median_rent_error": [3, 3],
        }
    )

    result = df_checks_output.encode_categorical(
        variable="appearance"
    ).pivot_wider(
        index=["geoid", "name"],
        names_from="variable",
        values_from=["estimate", "error"],
        names_glue="{variable}_{_value}",
        reset_index=False,
    )
    assert_frame_equal(result, df_out.set_index(["geoid", "name"]))


def test_names_glue_single_column(df_checks_output):
    """
    Test names_glue for single column.
    """

    df_out = (
        df_checks_output.pivot(["geoid", "name"], "variable", "estimate")
        .add_suffix("_estimate")
        .rename_axis(columns=None)
        .reset_index()
    )

    result = df_checks_output.pivot_wider(
        slice("geoid", "name"),
        "variable",
        "estimate",
        names_glue="{variable}_estimate",
    )
    assert_frame_equal(result, df_out)


def test_int_columns():
    """Test output when names_from is not a string dtype."""
    df_in = pd.DataFrame(
        [
            {"name": 1, "n": 10, "pct": 0.1},
            {"name": 2, "n": 20, "pct": 0.2},
            {"name": 3, "n": 30, "pct": 0.3},
        ]
    )

    df_out = pd.DataFrame(
        [
            {
                "num": 0,
                "n_1": 10.0,
                "n_2": 20.0,
                "n_3": 30.0,
                "pct_1": 0.1,
                "pct_2": 0.2,
                "pct_3": 0.3,
            }
        ]
    )

    result = df_in.assign(num=0).pivot_wider(
        index="num", names_from="name", values_from=["n", "pct"], names_sep="_"
    )

    assert_frame_equal(result, df_out)


@pytest.fixture
def df_expand():
    """pytest fixture"""
    # adapted from
    # https://github.com/tidyverse/tidyr/issues/770#issuecomment-993872495
    return pd.DataFrame(
        dict(
            id=pd.Categorical(
                values=(2, 1, 1, 2, 1), categories=(1, 2, 3), ordered=True
            ),
            year=(2018, 2018, 2019, 2020, 2020),
            gender=pd.Categorical(
                ("female", "male", "male", "female", "male")
            ),
            percentage=range(30, 80, 10),
        ),
        index=np.repeat([0], 5),
    )


def test_names_expand(df_expand):
    """Test output if `names_expand`"""
    actual = df_expand.pivot("year", "id", "percentage").reindex(
        columns=pd.Categorical([1, 2, 3], ordered=True)
    )
    expected = df_expand.pivot_wider(
        "year", "id", "percentage", names_expand=True, flatten_levels=False
    )
    assert_frame_equal(actual, expected)


def test_names_expand_flatten_levels(df_expand):
    """Test output if `names_expand`"""
    actual = (
        df_expand.pivot("year", "id", "percentage")
        .reindex(columns=[1, 2, 3])
        .rename_axis(columns=None)
        .reset_index()
    )
    expected = df_expand.pivot_wider(
        "year", "id", "percentage", names_expand=True, flatten_levels=True
    )
    assert_frame_equal(actual, expected)


def test_index_expand(df_expand):
    """Test output if `index_expand`"""
    actual = df_expand.pivot("id", "year", "percentage").reindex(
        pd.Categorical([1, 2, 3], ordered=True)
    )
    expected = df_expand.pivot_wider(
        "id", "year", "percentage", index_expand=True, flatten_levels=False
    )
    assert_frame_equal(actual, expected)


def test_index_expand_flatten_levels(df_expand):
    """Test output if `index_expand`"""
    actual = (
        df_expand.pivot("id", "year", "percentage")
        .reindex(pd.Categorical([1, 2, 3], ordered=True))
        .rename_axis(columns=None)
        .reset_index()
    )
    expected = df_expand.pivot_wider(
        "id", "year", "percentage", index_expand=True
    )
    assert_frame_equal(actual, expected)


def test_expand_multiple_levels(df_expand):
    """Test output for names_expand for multiple names_from."""
    expected = df_expand.pivot_wider(
        "id",
        ("year", "gender"),
        "percentage",
        names_expand=True,
        flatten_levels=False,
    )
    actual = df_expand.complete("year", "gender", "id").pivot(
        "id", ("year", "gender"), "percentage"
    )
    assert_frame_equal(actual, expected)


def test_expand_multiple_levels_flatten_levels(df_expand):
    """Test output for names_expand for multiple names_from."""
    expected = df_expand.pivot_wider(
        "id",
        ("year", "gender"),
        "percentage",
        names_expand=True,
        flatten_levels=True,
    )
    actual = (
        df_expand.complete("year", "gender", "id")
        .pivot("id", ("year", "gender"), "percentage")
        .collapse_levels()
        .reset_index()
    )
    assert_frame_equal(actual, expected)


@pytest.fixture
def multi():
    """fixture for MultiIndex column"""
    columns = pd.MultiIndex.from_tuples(
        [("first", "extra"), ("second", "extra"), ("A", "cat")],
        names=["exp", "animal"],
    )

    data = np.array(
        [
            ["bar", "one", 0.10771469563752678],
            ["bar", "two", -0.6453410828562166],
            ["baz", "one", 0.3210232406192864],
            ["baz", "two", 2.010694653300755],
        ],
        dtype=object,
    )

    return pd.DataFrame(data, columns=columns)


errors = [
    ["multi", ("first", "extra"), [("second", "extra")], None],
    ["multi", [("first", "extra")], ("second", "extra"), None],
    ("multi", None, [("second", "extra")], ("A", "cat")),
]


@pytest.mark.parametrize(
    "multi,index,names_from,values_from", errors, indirect=["multi"]
)
def test_multiindex(multi, index, names_from, values_from):
    """
    Raise if df.columns is a MultiIndex
    and index/names_from/values_from
    is not a list of tuples
    """
    with pytest.raises(TypeError):
        multi.pivot_wider(
            index=index, names_from=names_from, values_from=values_from
        )


def test_multiindex_values_from(multi):
    """
    Raise if df.columns is a MultiIndex,
    values_from is a list of tuples,
    and not all entries are tuples
    """
    with pytest.raises(TypeError):
        multi.pivot_wider(
            names_from=[("second", "extra")], values_from=[("A", "cat"), "A"]
        )


def test_multiindex_index(multi):
    """
    Raise if df.columns is a MultiIndex,
    index is a list of tuples,
    and not all entries are tuples
    """
    with pytest.raises(TypeError):
        multi.pivot_wider(
            names_from=[("second", "extra")],
            index=[("first", "extra"), "first"],
        )


def test_multi_index_values_from(multi):
    """
    Raise if df.columns is a MultiIndex,
    values_from is a list of tuples,
    and not all entries are tuples
    """
    with pytest.raises(TypeError):
        multi.pivot_wider(
            names_from=[("second", "extra"), "first"],
            values_from=[("A", "cat"), "A"],
        )


def test_multiindex_values_from_missing(multi):
    """
    Raise if df.columns is a MultiIndex,
    values_from is a list of tuples,
    and a tuple is missing
    """
    with pytest.raises(KeyError):
        multi.pivot_wider(
            names_from=[("second", "extra")], values_from=[("A", "ct")]
        )


def test_multiindex_index_missing(multi):
    """
    Raise if df.columns is a MultiIndex,
    index is a list of tuples,
    and a tuple is missing
    """
    with pytest.raises(KeyError):
        multi.pivot_wider(
            names_from=[("second", "extra")], index=[("first", "ext")]
        )


def test_multi_index_values_from_missing(multi):
    """
    Raise if df.columns is a MultiIndex,
    values_from is a list of tuples,
    and a tuple is missing
    """
    with pytest.raises(KeyError):
        multi.pivot_wider(
            names_from=[("sec", "extra")], values_from=[("A", "cat")]
        )
