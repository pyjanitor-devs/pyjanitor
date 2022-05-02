import pytest
from helpers import running_on_ci

from janitor.errors import JanitorError

if running_on_ci():
    import pyspark
else:
    pyspark = pytest.importorskip("pyspark")
import janitor.spark  # noqa: F401 isort:skip


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
@pytest.mark.turtle
def test_clean_names_method_chain(spark_df):
    """Tests clean names function in a method chain call."""
    spark_df = spark_df.clean_names()
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(spark_df.columns) == set(expected_columns)


@pytest.mark.turtle
@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_clean_names_special_characters(spark_df):
    """Tests that remove_special parameter takes effect."""
    spark_df = spark_df.clean_names(remove_special=True)
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals",
        "cities",
    ]
    assert set(spark_df.columns) == set(expected_columns)


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_clean_names_case_type_uppercase(spark_df):
    """Tests case_type parameter 'upper' takes effect."""
    spark_df = spark_df.clean_names(case_type="upper")
    expected_columns = [
        "A",
        "BELL_CHART",
        "DECORATED_ELEPHANT",
        "ANIMALS@#$%^",
        "CITIES",
    ]
    assert set(spark_df.columns) == set(expected_columns)


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_clean_names_case_type_preserve(spark_df):
    """Tests case_type parameter 'preserve' takes effect."""
    spark_df = spark_df.clean_names(case_type="preserve")
    expected_columns = [
        "a",
        "Bell_Chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(spark_df.columns) == set(expected_columns)


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_clean_names_case_type_invalid(spark_df):
    """Tests case_type parameter when it is invalid."""
    with pytest.raises(JanitorError, match=r"case_type must be one of:"):
        spark_df = spark_df.clean_names(case_type="foo")


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_clean_names_camelcase_to_snake(spark_df):
    """Tests case_type parameter 'snake' takes effect."""
    spark_df = spark_df.selectExpr("a AS AColumnName").clean_names(
        case_type="snake"
    )
    assert list(spark_df.columns) == ["a_column_name"]


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
@pytest.mark.parametrize(
    "strip_underscores", ["both", True, "right", "r", "left", "l"]
)
def test_clean_names_strip_underscores(spark_df, strip_underscores):
    """Tests strip_underscores parameter takes effect."""
    if strip_underscores in ["right", "r"]:
        spark_df = spark_df.selectExpr(
            *[f"`{col}` AS `{col}_`" for col in spark_df.columns]
        )
    elif strip_underscores in ["left", "l"]:
        spark_df = spark_df.selectExpr(
            *[f"`{col}` AS `_{col}`" for col in spark_df.columns]
        )
    elif strip_underscores in ["both", True]:
        spark_df = spark_df.selectExpr(
            *[f"`{col}` AS `_{col}_`" for col in spark_df.columns]
        )

    spark_df = spark_df.clean_names(strip_underscores=strip_underscores)

    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]

    assert set(spark_df.columns) == set(expected_columns)
