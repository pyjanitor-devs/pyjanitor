import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df_checks_output():
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


def test_names_from_None(df_checks_output):
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


def test_names_sort_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `names_sort`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=["estimate", "variable"],
            names_sort=2,
        )


def test_flatten_levels_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `flatten_levels`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=["estimate", "variable"],
            flatten_levels=2,
        )


@pytest.mark.xfail(reason="parameter is deprecated.")
def test_names_from_position_wrong_type(df_checks_output):
    """
    Raise TypeError if the wrong type
    is provided for `names_from_position`.
    """
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=["estimate", "variable"],
            names_from_position=2,
        )


@pytest.mark.xfail(reason="parameter is deprecated.")
def test_names_from_position_wrong_value(df_checks_output):
    """
    Raise ValueError if `names_from_position`
    is not "first" or "last".
    """
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=["estimate", "variable"],
            names_from_position="1st",
        )


@pytest.mark.xfail(reason="parameter is deprecated.")
def test_name_prefix_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `names_prefix`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], names_prefix=1
        )


def test_name_sep_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `names_sep`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], names_sep=1
        )


def test_name_glue_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `names_glue`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], names_glue=1
        )


def test_levels_order_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `levels_order`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], levels_order=1
        )


@pytest.mark.xfail(reason="parameter is deprecated.")
def test_fill_value_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `fill_value`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], fill_value={2}
        )


@pytest.mark.xfail(reason="parameter is deprecated.")
def test_aggfunc_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `aggfunc`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], aggfunc={2}
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

    result = df_in.pivot_wider(index=["a", "b"], names_from="name")

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
        # check_dtype=False,
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

    result = df_in.encode_categorical(gender=(None, "appearance")).pivot_wider(
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


def test_names_glue():
    """Test output with `names_glue`"""
    df_in = pd.DataFrame(
        {
            "family": ["Kelly", "Kelly", "Quin", "Quin"],
            "name": ["Mark", "Scott", "Tegan", "Sara"],
            "n": [1, 2, 1, 2],
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
        names_glue=lambda col: f"name{col}",
    )
    assert_frame_equal(result, df_out)


def test_change_level_order():
    """
    Test output with `levels_order`,
    while maintaining order from `names_from`.
    """
    df_in = pd.DataFrame(
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

    result = df_in.encode_categorical(
        variable=(None, "appearance")
    ).pivot_wider(
        index=["geoid", "name"],
        names_from="variable",
        values_from=["estimate", "error"],
        levels_order=["variable", None],
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
