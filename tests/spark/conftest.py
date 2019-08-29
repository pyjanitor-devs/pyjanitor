import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def spark_df(spark):
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
