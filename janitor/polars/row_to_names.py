"""row_to_names implementation for polars."""

from __future__ import annotations

from functools import singledispatch

from janitor.utils import check, import_message

from .polars_flavor import register_dataframe_method

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


@register_dataframe_method
def row_to_names(
    df: pl.DataFrame,
    row_numbers: int | list | slice = 0,
    remove_rows: bool = False,
    remove_rows_above: bool = False,
    separator: str = "_",
) -> pl.DataFrame:
    """
    Elevates a row, or rows, to be the column names of a DataFrame.

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
            It can be an integer, list or a slice.
        remove_rows: Whether the row(s) should be removed from the DataFrame.
        remove_rows_above: Whether the row(s) above the selected row should
            be removed from the DataFrame.
        separator: Combines the labels into a single string,
            if row_numbers is a list of integers. Default is '_'.

    Returns:
        A polars DataFrame.
    """  # noqa: E501
    return _row_to_names(
        row_numbers,
        df=df,
        remove_rows=remove_rows,
        remove_rows_above=remove_rows_above,
        separator=separator,
    )


@singledispatch
def _row_to_names(
    row_numbers, df, remove_rows, remove_rows_above, separator
) -> pl.DataFrame:
    """
    Base function for row_to_names.
    """
    raise TypeError(
        "row_numbers should be either an integer, "
        "a slice or a list; "
        f"instead got type {type(row_numbers).__name__}"
    )


@_row_to_names.register(int)  # noqa: F811
def _row_to_names_dispatch(  # noqa: F811
    row_numbers, df, remove_rows, remove_rows_above, separator
):
    headers = df.row(row_numbers, named=True)
    headers = {col: str(repl) for col, repl in headers.items()}
    df = df.rename(mapping=headers)
    if remove_rows_above and remove_rows:
        return df.slice(row_numbers + 1)
    elif remove_rows_above:
        return df.slice(row_numbers)
    elif remove_rows:
        expression = pl.int_range(pl.len()).ne(row_numbers)
        return df.filter(expression)
    return df


@_row_to_names.register(slice)  # noqa: F811
def _row_to_names_dispatch(  # noqa: F811
    row_numbers, df, remove_rows, remove_rows_above, separator
):
    if row_numbers.step is not None:
        raise ValueError(
            "The step argument for slice is not supported in row_to_names."
        )
    headers = df.slice(row_numbers.start, row_numbers.stop - row_numbers.start)
    expression = pl.all().str.concat(delimiter=separator)
    headers = headers.select(expression).row(0, named=True)
    headers = {col: str(repl) for col, repl in headers.items()}
    df = df.rename(mapping=headers)
    if remove_rows_above and remove_rows:
        return df.slice(row_numbers.stop)
    elif remove_rows_above:
        return df.slice(row_numbers.start)
    elif remove_rows:
        expression = pl.int_range(pl.len()).is_between(
            row_numbers.start, row_numbers.stop, closed="left"
        )
        return df.filter(~expression)
    return df


@_row_to_names.register(list)  # noqa: F811
def _row_to_names_dispatch(  # noqa: F811
    row_numbers, df, remove_rows, remove_rows_above, separator
):
    if remove_rows_above:
        raise ValueError(
            "The remove_rows_above argument is applicable "
            "only if the row_numbers argument is an integer "
            "or a slice."
        )

    for entry in row_numbers:
        check("entry in the row_numbers argument", entry, [int])

    expression = pl.all().gather(row_numbers)
    expression = expression.str.concat(delimiter=separator)
    headers = df.select(expression).row(0, named=True)
    headers = {col: str(repl) for col, repl in headers.items()}
    df = df.rename(mapping=headers)
    if remove_rows:
        expression = pl.int_range(pl.len()).is_in(row_numbers)
        return df.filter(~expression)
    return df
