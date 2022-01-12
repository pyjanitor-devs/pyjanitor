import pytest
from helpers import running_on_ci
from pandas.testing import assert_frame_equal

if running_on_ci():
    import pyspark
else:
    pyspark = pytest.importorskip("pyspark")
import janitor.spark  # noqa: F401 isort:skip


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_update_where_string(dataframe, spark_dataframe):
    """Test update_where and update with a string."""
    assert_frame_equal(
        spark_dataframe.update_where(
            conditions="""
            `decorated-elephant` = 1 AND `animals@#$%^` = 'rabbit'
            """,
            target_column_name="cities",
            target_val="Durham",
        ).toPandas(),
        dataframe.update_where(
            (dataframe["decorated-elephant"] == 1)
            & (dataframe["animals@#$%^"] == "rabbit"),
            "cities",
            "Durham",
        ),
    )


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_update_where_float(dataframe, spark_dataframe):
    """Test update_where and update with a float."""
    assert_frame_equal(
        spark_dataframe.update_where(
            conditions="""
            `decorated-elephant` = 1 AND `animals@#$%^` = 'rabbit'
            """,
            target_column_name="Bell__Chart",
            target_val=3.234789,
        ).toPandas(),
        dataframe.update_where(
            (dataframe["decorated-elephant"] == 1)
            & (dataframe["animals@#$%^"] == "rabbit"),
            "Bell__Chart",
            3.234789,
        ),
    )


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_update_where_int(dataframe, spark_dataframe):
    """Test update_where and update with a int."""
    assert_frame_equal(
        spark_dataframe.update_where(
            conditions="""
            `decorated-elephant` = 1 AND `animals@#$%^` = 'rabbit'
            """,
            target_column_name="a",
            target_val=10,
        ).toPandas(),
        dataframe.update_where(
            (dataframe["decorated-elephant"] == 1)
            & (dataframe["animals@#$%^"] == "rabbit"),
            "a",
            10,
        ),
    )


@pytest.mark.xfail(reason="causing issues in CI, to be fixed later")
@pytest.mark.spark_functions
def test_update_where_column_dne(dataframe, spark_dataframe):
    """Test update_where. Target column name does not exists."""
    assert_frame_equal(
        spark_dataframe.update_where(
            conditions="""
            `decorated-elephant` = 1 AND `animals@#$%^` = 'rabbit'
            """,
            target_column_name="c",
            target_val=10,
        ).toPandas(),
        dataframe.update_where(
            (dataframe["decorated-elephant"] == 1)
            & (dataframe["animals@#$%^"] == "rabbit"),
            "c",
            10,
        ),
    )
