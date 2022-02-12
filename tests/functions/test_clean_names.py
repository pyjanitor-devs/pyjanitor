import pandas as pd
import pytest

from janitor.errors import JanitorError


@pytest.mark.functions
def test_clean_names_method_chain(dataframe):
    """Tests clean_names default args in a method chain."""
    df = dataframe.clean_names()
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
def test_clean_names_special_characters(dataframe):
    """Tests clean_names `remove_special` parameter."""
    df = dataframe.clean_names(remove_special=True)
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals",
        "cities",
    ]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
def test_clean_names_uppercase(dataframe):
    """Tests clean_names `case_type` parameter = upper."""
    df = dataframe.clean_names(case_type="upper", remove_special=True)
    expected_columns = [
        "A",
        "BELL_CHART",
        "DECORATED_ELEPHANT",
        "ANIMALS",
        "CITIES",
    ]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
def test_clean_names_original_columns(dataframe):
    """Tests clean_names `preserve_original_columns` parameter."""
    df = dataframe.clean_names(preserve_original_columns=True)
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
    """Tests clean_names default args on a multi-index dataframe."""
    df = multiindex_dataframe.clean_names()

    levels = [
        ["a", "bell_chart", "decorated_elephant"],
        ["b", "normal_distribution", "r_i_p_rhino"],
    ]

    codes = [[0, 1, 2], [0, 1, 2]]

    expected_columns = pd.MultiIndex(levels=levels, codes=codes)
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
@pytest.mark.parametrize(
    "strip_underscores", ["both", True, "right", "r", "left", "l"]
)
def test_clean_names_strip_underscores(
    multiindex_dataframe,
    strip_underscores,
):
    """Tests clean_names `strip_underscores` param on a multi-index
    dataframe.
    """
    if strip_underscores in ["right", "r"]:
        df = multiindex_dataframe.rename(columns=lambda x: x + "_")
    elif strip_underscores in ["left", "l"]:
        df = multiindex_dataframe.rename(columns=lambda x: "_" + x)
    elif strip_underscores in ["both", None]:
        df = multiindex_dataframe.rename(columns=lambda x: "_" + x + "_")
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
def test_clean_names_strip_accents():
    """Tests clean_names `strip_accents` parameter."""
    df = pd.DataFrame({"João": [1, 2], "Лука́ся": [1, 2], "Käfer": [1, 2]})
    df = df.clean_names(strip_accents=True)
    expected_columns = ["joao", "лукася", "kafer"]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
def test_incorrect_strip_underscores(multiindex_dataframe):
    """Checks error is thrown when `strip_underscores` is given an
    invalid argument.
    """
    with pytest.raises(JanitorError):
        multiindex_dataframe.clean_names(strip_underscores="hello")


@pytest.mark.functions
def test_clean_names_preserve_case_true(multiindex_dataframe):
    """Tests clean_names `case_type` parameter = preserve."""
    df = multiindex_dataframe.clean_names(case_type="preserve")

    levels = [
        ["a", "Bell_Chart", "decorated_elephant"],
        ["b", "Normal_Distribution", "r_i_p_rhino"],
    ]

    codes = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, codes=codes)
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
def test_clean_names_camelcase_to_snake(dataframe):
    """Tests clean_names `case_type` parameter = snake."""
    df = (
        dataframe.select_columns(["a"])
        .rename_column("a", "AColumnName")
        .clean_names(case_type="snake")
    )
    assert list(df.columns) == ["a_column_name"]


@pytest.mark.functions
def test_clean_names_camelcase_to_snake_multi(dataframe):
    """Tests clean_names `case_type` parameter = snake on a multi-index
    dataframe.
    """
    df = (
        dataframe.select_columns(["a", "Bell__Chart", "decorated-elephant"])
        .rename_column("a", "snakesOnAPlane")
        .rename_column("Bell__Chart", "SnakesOnAPlane2")
        .rename_column("decorated-elephant", "snakes_on_a_plane3")
        .clean_names(
            case_type="snake", strip_underscores=True, remove_special=True
        )
    )
    assert list(df.columns) == [
        "snakes_on_a_plane",
        "snakes_on_a_plane2",
        "snakes_on_a_plane3",
    ]


@pytest.mark.functions
def test_clean_names_enforce_string(dataframe):
    """Tests clean_names `enforce_string` parameter."""
    df = dataframe.rename(columns={"a": 1}).clean_names(enforce_string=True)
    for c in df.columns:
        assert isinstance(c, str)


@pytest.mark.functions
def test_clean_names_truncate_limit(dataframe):
    """Tests clean_names `truncate_limit` parameter."""
    df = dataframe.clean_names(truncate_limit=7)
    expected_columns = ["a", "bell_ch", "decorat", "animals", "cities"]
    assert set(df.columns) == set(expected_columns)


@pytest.mark.functions
def test_charac():
    """Ensure non standard characters and spaces have been cleaned up."""

    df = pd.DataFrame(
        {
            r"Current accountbalance(in % of GDP)": range(5),
        }
    )
    df = df.clean_names(strip_underscores=True, case_type="lower")

    assert "current_accountbalance_in_%_of_gdp" in df.columns
