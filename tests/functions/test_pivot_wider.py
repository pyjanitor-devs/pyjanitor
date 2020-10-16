from itertools import product

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


def test_type_index1(df_checks_output, index={"geoid"}):
    "Raise TypeError if wrong type is provided for the `index`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_type_index2(df_checks_output, index=("geoid", "name")):
    "Raise TypeError if wrong type is provided for the `index`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_type_names_from1(df_checks_output, names_from={"variable"}):
    "Raise TypeError if wrong type is provided for `names_from`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_type_names_from2(df_checks_output, names_from=("variable",)):
    "Raise TypeError if wrong type is provided for `names_from`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_names_from_None(df_checks_output, names_from=None):
    "Raise ValueError if no value is provided for ``names_from``."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_presence_index1(df_checks_output, index="geo"):
    "Raise ValueError if labels in `index` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_presence_index2(df_checks_output, index=["geoid", "Name"]):
    "Raise ValueError if labels in `index` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_presence_names_from1(df_checks_output, names_from="estmt"):
    "Raise ValueError if labels in `names_from` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_presence_names_from2(df_checks_output, names_from=["estimat"]):
    "Raise ValueError if labels in `names_from` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_values_from_first_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], values_from_first=2
):
    "Raise TypeError if the wrong type is provided for `values_from_first`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=names_from,
            values_from_first=values_from_first,
        )


def test_name_prefix_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], names_prefix=1
):
    "Raise TypeError if the wrong type is provided for `names_prefix`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, names_prefix=names_prefix
        )


def test_name_sep_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], names_sep=1
):
    "Raise TypeError if the wrong type is provided for `names_sep`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, names_sep=names_sep
        )


def test_values_from_len_equal_1(
    df_checks_output,
    names_from=["estimate", "variable"],
    values_from_first=False,
):
    """
    Raise ValueError if the length of `values_from` is 1 and
    `values_from_first` is False.
    """
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=names_from,
            values_from="error",
            values_from_first=False,
        )


def test_fill_value_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], fill_value={2}
):
    "Raise TypeError if the wrong type is provided for `fill_value`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, fill_value=fill_value
        )


def test_aggfunc_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], aggfunc=0
):
    "Raise TypeError if the wrong type is provided for `aggfunc`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, aggfunc=aggfunc
        )


def test_dropna_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], dropna=0
):
    "Raise TypeError if the wrong type is provided for `dropna`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, dropna=dropna
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


def pivot_longer_wider_longer():
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
    ).pivot_wider(index="name", names_from="drug", values_from="heartrate")

    assert_frame_equal(result, df)
