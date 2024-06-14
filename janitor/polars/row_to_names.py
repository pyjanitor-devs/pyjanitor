"""clean_names implementation for polars."""

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
def row_to_names(
    df,
    row_numbers: int | list = 0,
    remove_rows: bool = False,
    remove_rows_above: bool = False,
    separator: str = "_",
) -> pl.DataFrame:
    """
    Elevates a row, or rows, to be the column names of a DataFrame.

    `row_to_names` can also be applied to a LazyFrame.

    Examples:
        Replace column names with the first row.

        >>> import polars as pl
        >>> import janitor.polars
        >>> df = pl.DataFrame({
        ...     "a": ["nums", '6', '9'],
        ...     "b": ["chars", "x", "y"],
        ... })
        >>> df
        shape: (3, 2)
        ┌──────┬───────┐
        │ a    ┆ b     │
        │ ---  ┆ ---   │
        │ str  ┆ str   │
        ╞══════╪═══════╡
        │ nums ┆ chars │
        │ 6    ┆ x     │
        │ 9    ┆ y     │
        └──────┴───────┘
        >>> df.row_to_names(0, remove_rows=True)
        shape: (2, 2)
        ┌──────┬───────┐
        │ nums ┆ chars │
        │ ---  ┆ ---   │
        │ str  ┆ str   │
        ╞══════╪═══════╡
        │ 6    ┆ x     │
        │ 9    ┆ y     │
        └──────┴───────┘
        >>> df.row_to_names(row_numbers=[0,1], remove_rows=True)
        shape: (1, 2)
        ┌────────┬─────────┐
        │ nums_6 ┆ chars_x │
        │ ---    ┆ ---     │
        │ str    ┆ str     │
        ╞════════╪═════════╡
        │ 9      ┆ y       │
        └────────┴─────────┘

        Remove rows above the elevated row and the elevated row itself.

        >>> df = pl.DataFrame({
        ...     "a": ["bla1", "nums", '6', '9'],
        ...     "b": ["bla2", "chars", "x", "y"],
        ... })
        >>> df
        shape: (4, 2)
        ┌──────┬───────┐
        │ a    ┆ b     │
        │ ---  ┆ ---   │
        │ str  ┆ str   │
        ╞══════╪═══════╡
        │ bla1 ┆ bla2  │
        │ nums ┆ chars │
        │ 6    ┆ x     │
        │ 9    ┆ y     │
        └──────┴───────┘
        >>> df.row_to_names(1, remove_rows=True, remove_rows_above=True)
        shape: (2, 2)
        ┌──────┬───────┐
        │ nums ┆ chars │
        │ ---  ┆ ---   │
        │ str  ┆ str   │
        ╞══════╪═══════╡
        │ 6    ┆ x     │
        │ 9    ┆ y     │
        └──────┴───────┘

    !!! info "New in version 0.28.0"

    Args:
        row_numbers: Position of the row(s) containing the variable names.
            Note that indexing starts from 0. It can also be a list.
            Defaults to 0 (first row).
        remove_rows: Whether the row(s) should be removed from the DataFrame.
        remove_rows_above: Whether the row(s) above the selected row should
            be removed from the DataFrame.
        separator: Combines the labels into a single string,
            if row_numbers is a list of integers. Default is '_'.

    Returns:
        A polars DataFrame.
    """  # noqa: E501
    return _row_to_names(
        df=df,
        row_numbers=row_numbers,
        remove_rows=remove_rows,
        remove_rows_above=remove_rows_above,
        separator=separator,
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
