from __future__ import annotations

from janitor.utils import import_message

from .polars_flavor import register_expr_method

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


@register_expr_method
def convert_excel_date(expr: pl.Expr) -> pl.Expr:
    """
    Convert Excel's serial date format into Python datetime format.

    Inspiration is from
    [Stack Overflow](https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas).

    Examples:
        >>> import polars as pl
        >>> import janitor.polars
        >>> df = pl.DataFrame({"date": [39690, 39690, 37118]})
        >>> df
        shape: (3, 1)
        ┌───────┐
        │ date  │
        │ ---   │
        │ i64   │
        ╞═══════╡
        │ 39690 │
        │ 39690 │
        │ 37118 │
        └───────┘
        >>> expression = pl.col('date').convert_excel_date().alias('date_')
        >>> df.with_columns(expression)
        shape: (3, 2)
        ┌───────┬────────────┐
        │ date  ┆ date_      │
        │ ---   ┆ ---        │
        │ i64   ┆ date       │
        ╞═══════╪════════════╡
        │ 39690 ┆ 2008-08-30 │
        │ 39690 ┆ 2008-08-30 │
        │ 37118 ┆ 2001-08-15 │
        └───────┴────────────┘

    !!! info "New in version 0.28.0"

    Returns:
        A polars Expression.
    """  # noqa: E501
    expression = pl.duration(days=expr)
    expression += pl.date(year=1899, month=12, day=30)
    return expression


@register_expr_method
def convert_matlab_date(expr: pl.Expr) -> pl.Expr:
    """
    Convert Matlab's serial date number into Python datetime format.

    Implementation is from
    [Stack Overflow](https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python).


    Examples:
        >>> import polars as pl
        >>> import janitor.polars
        >>> df = pl.DataFrame({"date": [737125.0, 737124.815863, 737124.4985, 737124]})
        >>> df
        shape: (4, 1)
        ┌───────────────┐
        │ date          │
        │ ---           │
        │ f64           │
        ╞═══════════════╡
        │ 737125.0      │
        │ 737124.815863 │
        │ 737124.4985   │
        │ 737124.0      │
        └───────────────┘
        >>> expression = pl.col('date').convert_matlab_date().alias('date_')
        >>> df.with_columns(expression)
        shape: (4, 2)
        ┌───────────────┬─────────────────────────┐
        │ date          ┆ date_                   │
        │ ---           ┆ ---                     │
        │ f64           ┆ datetime[μs]            │
        ╞═══════════════╪═════════════════════════╡
        │ 737125.0      ┆ 2018-03-06 00:00:00     │
        │ 737124.815863 ┆ 2018-03-05 19:34:50.563 │
        │ 737124.4985   ┆ 2018-03-05 11:57:50.399 │
        │ 737124.0      ┆ 2018-03-05 00:00:00     │
        └───────────────┴─────────────────────────┘

    !!! info "New in version 0.28.0"

    Returns:
        A polars Expression.
    """  # noqa: E501
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    expression = expr.sub(719529).mul(86_400_000)
    expression = pl.duration(milliseconds=expression)
    expression += pl.datetime(year=1970, month=1, day=1)
    return expression
