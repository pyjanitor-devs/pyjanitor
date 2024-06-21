import polars as pl

import janitor.polars  # noqa: F401


def test_convert_matlab_date():
    df = pl.DataFrame(
        {
            "dates": [
                733_301.0,
                729_159.0,
                734_471.0,
                737_299.563_296_356_5,
                737_300.000_000_000_0,
            ]
        }
    )
    expression = pl.col("dates").convert_matlab_date().alias("dd")
    expression = df.with_columns(expression).get_column("dd")
    assert expression.dtype.is_temporal() is True
