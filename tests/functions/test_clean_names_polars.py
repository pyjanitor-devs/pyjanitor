import subprocess

subprocess.run(["pip", "install", "polars"])
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from janitor import make_clean_names  # noqa: E402


@pytest.mark.functions
def test_clean_names_method_chain(dataframe):
    """Tests clean_names default args in a method chain."""
    df = pl.from_pandas(dataframe)
    df = df.rename(lambda col: make_clean_names(col, object_type="string"))
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]
    assert df.columns == expected_columns


@pytest.mark.functions
def test_clean_names_special_characters(dataframe):
    """Tests clean_names `remove_special` parameter."""
    df = pl.from_pandas(dataframe)
    df = df.rename(
        lambda col: make_clean_names(
            col, object_type="string", remove_special=True
        )
    )
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals",
        "cities",
    ]
    assert df.columns == expected_columns


@pytest.mark.functions
def test_clean_names_uppercase(dataframe):
    """Tests clean_names `case_type` parameter = upper."""
    df = pl.from_pandas(dataframe)
    df = df.rename(
        lambda col: make_clean_names(
            col, object_type="string", remove_special=True, case_type="upper"
        )
    )
    expected_columns = [
        "A",
        "BELL_CHART",
        "DECORATED_ELEPHANT",
        "ANIMALS",
        "CITIES",
    ]
    assert df.columns == expected_columns


@pytest.mark.functions
def test_clean_names_strip_accents():
    """Tests clean_names `strip_accents` parameter."""
    df = pl.DataFrame({"João": [1, 2], "Лука́ся": [1, 2], "Käfer": [1, 2]})
    df = df.rename(
        lambda col: make_clean_names(
            col, object_type="string", strip_accents=True
        )
    )
    expected_columns = ["joao", "лукася", "kafer"]
    assert df.columns == expected_columns


@pytest.mark.functions
def test_clean_names_camelcase_to_snake(dataframe):
    """Tests clean_names `case_type` parameter = snake."""
    df = pl.from_pandas(dataframe)
    df = (
        df.select("a")
        .rename({"a": "AColumnName"})
        .rename(
            lambda col: make_clean_names(
                col,
                object_type="string",
                remove_special=True,
                case_type="snake",
            )
        )
    )
    assert df.columns == ["a_column_name"]


@pytest.mark.functions
def test_clean_names_truncate_limit(dataframe):
    """Tests clean_names `truncate_limit` parameter."""
    df = pl.from_pandas(dataframe)
    df = df.rename(
        lambda col: make_clean_names(
            col, object_type="string", truncate_limit=7
        )
    )
    # df = dataframe.clean_names(truncate_limit=7)
    expected_columns = ["a", "bell_ch", "decorat", "animals", "cities"]
    assert df.columns == expected_columns


@pytest.mark.functions
def test_charac():
    """Ensure non standard characters and spaces have been cleaned up."""

    df = pl.DataFrame(
        {
            r"Current accountbalance(in % of GDP)": range(5),
        }
    )
    df = df.rename(
        lambda col: make_clean_names(
            col,
            object_type="string",
            strip_underscores=True,
            case_type="lower",
        )
    )

    assert "current_accountbalance_in_%_of_gdp" in df.columns


def test_clean_column_values():
    """Clean column values"""
    raw = pl.DataFrame({"raw": ["Abçdê fgí j"]})
    outcome = raw.with_columns(
        pl.col("raw").pipe(
            make_clean_names, object_type="polars", strip_accents=True
        )
    )
    assert list(outcome)[0][0] == "abcde_fgi_j"
