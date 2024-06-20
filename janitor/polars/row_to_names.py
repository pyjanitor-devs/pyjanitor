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
    row_numbers: int | list | slice,
    remove_rows: bool,
    remove_rows_above: bool,
    separator: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to convert rows in the DataFrame to column names.
    """
    check("separator", separator, [str])
    if isinstance(row_numbers, int):
        row_numbers = slice(row_numbers, row_numbers + 1)
    elif isinstance(row_numbers, slice):
        if row_numbers.step is not None:
            raise ValueError(
                "The step argument for slice is not supported in row_to_names."
            )
    elif isinstance(row_numbers, list):
        for entry in row_numbers:
            check("entry in the row_numbers argument", entry, [int])
    else:
        raise TypeError(
            "row_numbers should be either an integer, "
            "a slice or a list; "
            f"instead got type {type(row_numbers).__name__}"
        )
    is_a_slice = isinstance(row_numbers, slice)
    if is_a_slice:
        expression = pl.all().str.concat(delimiter=separator)
        expression = pl.struct(expression)
        offset = row_numbers.start
        length = row_numbers.stop - row_numbers.start
        mapping = df.slice(
            offset=offset,
            length=length,
        )
        mapping = mapping.select(expression)
    else:
        expression = pl.all().gather(row_numbers)
        expression = expression.str.concat(delimiter=separator)
        expression = pl.struct(expression)
        mapping = df.select(expression)
    if isinstance(df, pl.LazyFrame):
        mapping = mapping.collect()
    mapping = mapping.to_series(0)[0]
    df = df.rename(mapping=mapping)
    if remove_rows_above:
        if not is_a_slice:
            raise ValueError(
                "The remove_rows_above argument is applicable "
                "only if the row_numbers argument is an integer "
                "or a slice."
            )
        if remove_rows:
            return df.slice(offset=row_numbers.stop)
        return df.slice(offset=row_numbers.start)

    if remove_rows:
        if is_a_slice:
            df = [
                df.slice(offset=0, length=row_numbers.start),
                df.slice(offset=row_numbers.stop),
            ]
            return pl.concat(df, rechunk=True)
        name = "".join(df.columns)
        name = f"{name}_"
        df = (
            df.with_row_index(name=name)
            .filter(pl.col(name=name).is_in(row_numbers).not_())
            .select(pl.exclude(name))
        )
        return df

    return df
