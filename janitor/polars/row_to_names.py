"""row_to_names implementation for polars."""

from __future__ import annotations

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

    For a LazyFrame, the user should materialize into a DataFrame before using `row_to_names`..

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
            Note that indexing starts from 0. It can also be a list/slice.
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
    df: pl.DataFrame,
    row_numbers: int | list | slice,
    remove_rows: bool,
    remove_rows_above: bool,
    separator: str,
) -> pl.DataFrame:
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
