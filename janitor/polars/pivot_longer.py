"""pivot_longer_spec implementation for polars."""

from collections import defaultdict

from janitor.utils import import_message

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def _pivot_longer_dot_value(
    df: pl.DataFrame, spec: pl.DataFrame
) -> pl.DataFrame:
    """
    Reshape DataFrame to long form based on metadata in `spec`.
    """
    index = [column for column in df.columns if column not in spec[".name"]]
    not_dot_value = [
        column for column in spec.columns if column not in {".name", ".value"}
    ]
    idx = "".join(spec.columns)
    if len(spec.columns) == 2:
        # use a cumulative count to properly pair the columns
        spec = spec.with_columns(
            pl.cum_count(".value").over(".value").alias(idx)
        )
    else:
        # assign a number to each group
        ngroup_expr = pl.first(idx).over(not_dot_value).rank("dense").sub(1)
        spec = spec.with_row_index(name=idx).with_columns(ngroup_expr)
    mapping = defaultdict(list)
    for position, column_name, replacement_name in zip(
        spec.get_column(name=idx),
        spec.get_column(name=".name"),
        spec.get_column(name=".value"),
    ):
        expression = pl.col(column_name).alias(replacement_name)
        mapping[position].append(expression)
    mapping = (
        [
            pl.col(index),
            *columns_to_select,
            pl.lit(position, dtype=pl.UInt32).alias(idx),
        ]
        for position, columns_to_select in mapping.items()
    )

    df_is_a_lazyframe = isinstance(df, pl.LazyFrame)
    df = map(df.select, mapping)
    df = pl.concat(df, how="diagonal_relaxed")
    spec = spec.select(idx, *not_dot_value).unique()
    if df_is_a_lazyframe:
        spec = spec.lazy()
    df = df.join(spec, on=idx, how="inner")
    df = df.select(pl.exclude(idx))
    return df
