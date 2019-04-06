import pandas as pd
import pytest

from hypothesis import given
from janitor.errors import JanitorError
from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
def test_clean_names_method_chain(df):
    df = df.clean_names()
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
@given(df=df_strategy())
def test_clean_names_special_characters(df):
    df = df.clean_names(remove_special=True)
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals",
        "cities",
    ]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
@given(df=df_strategy())
def test_clean_names_uppercase(df):
    df = df.clean_names(case_type="upper", remove_special=True)
    expected_columns = [
        "A",
        "BELL_CHART",
        "DECORATED_ELEPHANT",
        "ANIMALS",
        "CITIES",
    ]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
@given(df=df_strategy())
def test_clean_names_original_columns(df):
    df = df.clean_names(preserve_original_columns=True)
    expected_columns = [
        "a",
        "Bell__Chart",
        "decorated-elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(df.original_columns) == set(expected_columns)


@pytest.mark.functions
def test_multiindex_clean_names(multiindex_dataframe):
    df = multiindex_dataframe.clean_names()

    levels = [
        ["a", "bell_chart", "decorated_elephant"],
        ["b", "normal_distribution", "r_i_p_rhino"],
    ]

    codes = [[0, 1, 2], [0, 1, 2]]

    expected_columns = pd.MultiIndex(levels=levels, codes=codes)
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
@pytest.mark.WIP
@pytest.mark.parametrize(
    "strip_underscores", ["both", True, "right", "r", "left", "l"]
)
def test_clean_names_strip_underscores(
    multiindex_dataframe, strip_underscores
):
    if strip_underscores in ["right", "r"]:
        df = multiindex_dataframe.rename(columns=lambda x: x + "_")
    elif strip_underscores in ["left", "l"]:
        df = multiindex_dataframe.rename(columns=lambda x: "_" + x)
    else:
        df = multiindex_dataframe
    df = df.clean_names(strip_underscores=strip_underscores)

    levels = [
        ["a", "bell_chart", "decorated_elephant"],
        ["b", "normal_distribution", "r_i_p_rhino"],
    ]

    codes = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, codes=codes)
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
def test_incorrect_strip_underscores(multiindex_dataframe):
    with pytest.raises(JanitorError):
        multiindex_dataframe.clean_names(strip_underscores="hello")


@pytest.mark.functions
def test_clean_names_preserve_case_true(multiindex_dataframe):
    df = multiindex_dataframe.clean_names(case_type="preserve")

    levels = [
        ["a", "Bell_Chart", "decorated_elephant"],
        ["b", "Normal_Distribution", "r_i_p_rhino"],
    ]

    codes = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, codes=codes)
    assert set(df.columns) == set(expected_columns)
