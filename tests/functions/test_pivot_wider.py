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


combinations = [
    (
        pd.DataFrame(
            {
                "geoid": [1, 1, 1, 1, 13, 13, 13, 13],
                "name": [
                    "Alabama",
                    "Alabama",
                    "Alabama",
                    "Alabama",
                    "Georgia",
                    "Georgia",
                    "Georgia",
                    "Georgia",
                ],
                "variable": [
                    "pop_renter",
                    "pop_renter",
                    "median_rent",
                    "median_rent",
                    "pop_renter",
                    "pop_renter",
                    "median_rent",
                    "median_rent",
                ],
                "measure": [
                    "estimate",
                    "error",
                    "estimate",
                    "error",
                    "estimate",
                    "error",
                    "estimate",
                    "error",
                ],
                "value": [1434765, 16736, 747, 3, 3592422, 33385, 927, 3],
            }
        ),
        pd.DataFrame(
            {
                "geoid": [1, 13],
                "name": ["Alabama", "Georgia"],
                "pop_renter_estimate": [1434765, 3592422],
                "pop_renter_error": [16736, 33385],
                "median_rent_estimate": [747, 927],
                "median_rent_error": [3, 3],
            }
        ),
        ["geoid", "name"],
        ["variable", "measure"],
        "value",
        None,
        True,
        "first",
    ),
    (
        pd.DataFrame(
            {
                "family": ["Kelly", "Kelly", "Quin", "Quin"],
                "name": ["Mark", "Scott", "Tegan", "Sara"],
                "n": [1, 2, 1, 2],
            }
        ),
        pd.DataFrame(
            {
                "family": ["Kelly", "Quin"],
                1: ["Mark", "Tegan"],
                2: ["Scott", "Sara"],
            }
        ),
        "family",
        "n",
        "name",
        None,
        True,
        "first",
    ),
    (
        pd.DataFrame(
            {
                "family": ["Kelly", "Kelly", "Quin", "Quin"],
                "name": ["Mark", "Scott", "Tegan", "Sara"],
                "n": [1, 2, 1, 2],
            }
        ),
        pd.DataFrame(
            {
                "family": ["Kelly", "Quin"],
                "name1": ["Mark", "Tegan"],
                "name2": ["Scott", "Sara"],
            }
        ),
        "family",
        "n",
        "name",
        "name",
        True,
        "first",
    ),
    (
        pd.DataFrame(
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
        ),
        pd.DataFrame(
            {
                "geoid": [1, 13],
                "name": ["Alabama", "Georgia"],
                "pop_renter_estimate": [1434765, 3592422],
                "median_rent_estimate": [747, 927],
                "pop_renter_error": [16736, 33385],
                "median_rent_error": [3, 3],
            }
        ),
        ["geoid", "name"],
        "variable",
        ["estimate", "error"],
        None,
        False,
        "first",
    ),
    (
        pd.DataFrame(
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
        ),
        pd.DataFrame(
            {
                "geoid": {0: 1, 1: 13},
                "name": {0: "Alabama", 1: "Georgia"},
                "estimate_pop_renter": {0: 1434765, 1: 3592422},
                "estimate_median_rent": {0: 747, 1: 927},
                "error_pop_renter": {0: 16736, 1: 33385},
                "error_median_rent": {0: 3, 1: 3},
            }
        ),
        ["geoid", "name"],
        "variable",
        ["estimate", "error"],
        None,
        False,
        "last",
    ),
]


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
    """Raise ValueError if no value is provided for ``names_from``."""
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


def test_fill_value_wrong_type(df_checks_output):
    """Raise TypeError if the wrong type is provided for `fill_value`."""
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=["estimate", "variable"], fill_value={2}
        )


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
        names_to=("name", ".value"),
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


@pytest.mark.parametrize(
    """
    df_in,df_out,index,names_from,
    values_from, names_prefix,
    names_sort,names_from_position
    """,
    combinations,
)
def test_pivot_wider_various(
    df_in,
    df_out,
    index,
    names_from,
    values_from,
    names_prefix,
    names_sort,
    names_from_position,
):
    """
    Test `pivot_wider` function with various combinations.
    """
    result = df_in.pivot_wider(
        index=index,
        names_from=names_from,
        values_from=values_from,
        names_prefix=names_prefix,
        names_sort=names_sort,
        names_from_position=names_from_position,
    )
    assert_frame_equal(result, df_out)


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
        names_sort=True,
        names_from_position="last",
    )

    expected_output = df_collapse.pivot(  # noqa: PD010
        index="foo", columns="bar", values=["baz", "zoo"]
    )

    assert_frame_equal(
        result,
        expected_output,
        check_dtype=False,
    )


def test_fill_values():
    """Test output if `fill_value` is provided."""

    df_fill_value = pd.DataFrame(
        {
            "lev1": [1, 1, 1, 2, 2, 2],
            "lev2": [1, 1, 2, 1, 1, 2],
            "lev3": [1, 2, 1, 2, 1, 2],
            "lev4": [1, 2, 3, 4, 5, 6],
            "values": [0, 1, 2, 3, 4, 5],
        }
    )

    result = df_fill_value.pivot_wider(
        index=["lev1", "lev2"],
        names_from=["lev3"],
        values_from="values",
        flatten_levels=False,
        fill_value=0,
    )

    expected_output = pd.DataFrame(
        {1: [0, 2.0, 4, 0], 2: [1, 0, 3.0, 5]},
        index=pd.MultiIndex.from_tuples(
            [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
        ),
        columns=pd.Int64Index([1, 2], dtype="int64", name="lev3"),
    )
    assert_frame_equal(result, expected_output)


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

    result = df_in.pivot_wider(names_from="gender", names_prefix="contVar_")

    assert_frame_equal(result, expected_output)


def test_no_index_names_sort_True():
    """Test output if no `index` is supplied and `names_sort` is True."""
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

    result = df_in.pivot_wider(
        names_from="gender", names_sort=True, names_prefix="contVar_"
    )

    assert_frame_equal(result, expected_output)


def test_index_names_sort_True():
    """Test output if index is supplied and `names_sort ` is True."""
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
        index="var", names_from="stat", values_from="score", names_sort=True
    )

    assert_frame_equal(result, expected_output)


