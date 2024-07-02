"""complete implementation for polars."""

from __future__ import annotations

from janitor.utils import check, import_message

from .polars_flavor import register_dataframe_method, register_lazyframe_method

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


@register_lazyframe_method
@register_dataframe_method
def single_inequality_join(
    df: pl.DataFrame | pl.LazyFrame,
    other: pl.DataFrame | pl.LazyFrame,
    left_on: str,
    right_on: str,
    strategy: str,
    how: str = "inner",
) -> pl.DataFrame | pl.LazyFrame:
    """
    Join two DataFrames on a single inequality join condition.

    Only numeric and temporal columns are supported.

    !!! info "New in version 0.28.0"

    Args:
        df: A polars DataFrame/LazyFrame.
        other: A polars DataFrame/LazyFrame.
        left_on: Name of the left join column.
        right_on: Name of the right join column.
        strategy: type of inequality join. It should be one of
        `less_than`, `less_than_or_equal`, `greater_than`,
        `greater_than_or_equal`.

    Returns:
        A polars DataFrame/LazyFrame.
    """
    check("other", other, [pl.LazyFrame, pl.DataFrame])
    check("left_on", left_on, [str])
    check("right_on", right_on, [str])
    check("strategy", strategy, [str])
    if strategy not in {
        "less_than",
        "less_than_or_equal",
        "greater_than",
        "greater_than_or_equal",
    }:
        raise ValueError(
            '"strategy" should be one of '
            '"less_than", "less_than_or_equal" ',
            '"greater_than", "greater_than_or_equal"',
        )
    left_dtype = df.collect_schema()[left_on]
    if not left_dtype.is_temporal() and not left_dtype.is_numeric():
        raise TypeError(
            "Only numeric and temporal columns are supported "
            "in the single_inequality_join - "
            f"{left_on} column has a {left_dtype} dtype"
        )
    right_dtype = other.collect_schema()[right_on]
    if not right_dtype.is_temporal() and not left_dtype.is_numeric():
        raise TypeError(
            "Only numeric and temporal columns are supported "
            "in the single_inequality_join - "
            f"{right_on} column has a {right_dtype} dtype"
        )
    if left_dtype != right_dtype:
        raise TypeError(
            "Both columns should have the same type - "
            f"'{left_on}' has {left_dtype} type;"
            f"'{right_on}' has {right_dtype} type."
        )
    if strategy in {"less_than", "less_than_or_equal"}:
        # expression =
        pass
    return df, other
