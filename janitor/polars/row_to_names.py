"""clean_names implementation for polars."""

from __future__ import annotations

from janitor.utils import check, import_message

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def _row_to_names(
    df: pl.DataFrame | pl.LazyFrame,
    row_numbers: int | list,
    remove_rows: bool,
    remove_rows_above: bool,
    separator: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to convert rows in the DataFrame to column names.
    """
    check("separator", separator, [str])
    check("row_numbers", row_numbers, [int, list])
    row_numbers_is_a_list = False
    if isinstance(row_numbers, list):
        row_numbers_is_a_list = True
        for entry in row_numbers:
            check("entry in the row_numbers argument", entry, [int])
        expression = (
            pl.all()
            .gather(row_numbers)
            .cast(pl.String)
            .implode()
            .list.join(separator=separator)
        )
        expression = pl.struct(expression)
    else:
        expression = pl.all().gather(row_numbers).cast(pl.String)
        expression = pl.struct(expression)
    mapping = df.select(expression)
    if isinstance(mapping, pl.LazyFrame):
        mapping = mapping.collect()
    mapping = mapping.to_series(0)[0]
    df = df.rename(mapping=mapping)
    if remove_rows_above:
        if row_numbers_is_a_list:
            if not pl.Series(row_numbers).diff().drop_nulls().eq(1).all():
                raise ValueError(
                    "The remove_rows_above argument is applicable "
                    "only if the row_numbers argument is an integer, "
                    "or the integers in a list are consecutive increasing, "
                    "with a difference of 1."
                )
        if remove_rows:
            tail = row_numbers[-1] if row_numbers_is_a_list else row_numbers
            tail += 1
        else:
            tail = row_numbers[0] if row_numbers_is_a_list else row_numbers
        df = df.slice(offset=tail)
    elif remove_rows:
        idx = "".join(df.columns)
        df = df.with_row_index(name=idx)
        if row_numbers_is_a_list:
            df = df.filter(~pl.col(idx).is_in(row_numbers))
        else:
            df = df.filter(pl.col(idx) != row_numbers)
        df = df.drop(idx)
    return df
