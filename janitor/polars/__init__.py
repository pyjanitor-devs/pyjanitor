from __future__ import annotations

from janitor.utils import import_message

from .row_to_names import _row_to_names
from .clean_names import _clean_column_names, _clean_expr_names

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


@pl.api.register_lazyframe_namespace("janitor")
@pl.api.register_dataframe_namespace("janitor")
class PolarsFrame:
    def __init__(self, df: pl.DataFrame) -> pl.DataFrame:
        self._df = df

    def row_to_names(
        self,
        row_numbers: int | list = 0,
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
            >>> df.janitor.row_to_names(0, remove_rows=True)
            shape: (2, 2)
            ┌──────┬───────┐
            │ nums ┆ chars │
            │ ---  ┆ ---   │
            │ str  ┆ str   │
            ╞══════╪═══════╡
            │ 6    ┆ x     │
            │ 9    ┆ y     │
            └──────┴───────┘
            >>> df.janitor.row_to_names(row_numbers=[0,1], remove_rows=True)
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
            >>> df.janitor.row_to_names(1, remove_rows=True, remove_rows_above=True)
            shape: (2, 2)
            ┌──────┬───────┐
            │ nums ┆ chars │
            │ ---  ┆ ---   │
            │ str  ┆ str   │
            ╞══════╪═══════╡
            │ 6    ┆ x     │
            │ 9    ┆ y     │
            └──────┴───────┘
