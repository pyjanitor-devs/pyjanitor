"""Spark fixtures."""

import pytest

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        FloatType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )
except ImportError:
    pass


@pytest.fixture  # (scope="session")
def spark():
    """Create spark session."""
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def spark_df(spark):
    """Create spark dataframe."""
    schema = StructType(
        [
            StructField("a", IntegerType(), True),
            StructField("Bell__Chart", FloatType(), True),
            StructField("decorated-elephant", IntegerType(), True),
            StructField("animals@#$%^", StringType(), True),
            StructField("cities", StringType(), True),
        ]
    )
    return spark.createDataFrame([], schema)


@pytest.fixture
def spark_dataframe(spark, dataframe):
    """Another function to create spark dataframe."""
    return spark.createDataFrame(dataframe)
