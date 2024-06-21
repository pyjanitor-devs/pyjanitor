import polars as pl

import janitor.polars  # noqa: F401


def test_convert_excel_date():
    df = pl.DataFrame({"dates": [42580.3333333333]})

    expression = pl.col("dates").convert_excel_date().alias("dd")
    expression = df.with_columns(expression).get_column("dd")
    assert expression.dtype.is_temporal() is True
