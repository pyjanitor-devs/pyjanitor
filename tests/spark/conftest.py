import pytest

try:
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        FloatType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )
except ImportError:
    pyspark = None


@pytest.fixture(scope="session")
@pytest.mark.skipif(
    pyspark is None, reason="pyspark tests only required for CI"
)
def spark():
    spark = SparkSession.builder.getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
@pytest.mark.skipif(
    pyspark is None, reason="pyspark tests only required for CI"
)
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
