from __future__ import annotations

from polars.type_aliases import ColumnNameOrSelector

from janitor.utils import import_message

from .complete import _complete

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


@pl.api.register_dataframe_namespace("janitor")
class PolarsFrame:
    def __init__(self, df: pl.DataFrame) -> pl.DataFrame:
        self._df = df

    def complete(
        self,
        *columns: ColumnNameOrSelector,
        fill_value: dict | float | int | str = None,
        explicit: bool = True,
        sort: bool = False,
    ) -> pl.DataFrame:
        """
        Complete a data frame with missing combinations of data.

        It is modeled after tidyr's `complete` function.
        In a way, it is the inverse of `pl.drop_nulls`,
        as it exposes implicitly missing rows.

        Combinations of column names or a list/tuple of column names, or even a
        dictionary of column names and new values are possible.
        If a dictionary is passed,
        the user is required to ensure that the values are unique 1-D arrays.
        The keys in a dictionary must be present in the DataFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame(
            ...     {
            ...         "Aloha": range(3),
            ...         "Bell Chart": range(3),
            ...         "Animals@#$%^": range(3)
            ...     }
            ... )
            >>> df
            shape: (3, 3)
            ┌───────┬────────────┬──────────────┐
            │ Aloha ┆ Bell Chart ┆ Animals@#$%^ │
            │ ---   ┆ ---        ┆ ---          │
            │ i64   ┆ i64        ┆ i64          │
            ╞═══════╪════════════╪══════════════╡
            │ 0     ┆ 0          ┆ 0            │
            │ 1     ┆ 1          ┆ 1            │
            │ 2     ┆ 2          ┆ 2            │
            └───────┴────────────┴──────────────┘
            >>> df.janitor.clean_names(remove_special=True)
            shape: (3, 3)
            ┌───────┬────────────┬─────────┐
            │ aloha ┆ bell_chart ┆ animals │
            │ ---   ┆ ---        ┆ ---     │
            │ i64   ┆ i64        ┆ i64     │
            ╞═══════╪════════════╪═════════╡
            │ 0     ┆ 0          ┆ 0       │
            │ 1     ┆ 1          ┆ 1       │
            │ 2     ┆ 2          ┆ 2       │
            └───────┴────────────┴─────────┘

        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It could be column labels (string type),
                a list/tuple of column labels, or a dictionary that pairs
                column labels with new values.
            fill_value: Scalar value to use instead of nulls
                for missing combinations. A dictionary, mapping columns names
                to a scalar value is also accepted.
            explicit: Determines if only implicitly missing values
                should be filled (`False`), or all nulls existing in the dataframe
                (`True`). `explicit` is applicable only
                if `fill_value` is not `None`.
            sort: Sort DataFrame based on *columns.
        Returns:
            A polars DataFrame.
        """  # noqa: E501
        return _complete(
            df=self._df,
            columns=columns,
            fill_value=fill_value,
            explicit=explicit,
            sort=sort,
        )


@pl.api.register_lazyframe_namespace("janitor")
class PolarsLazyFrame:
    def __init__(self, df: pl.LazyFrame) -> pl.LazyFrame:
        self._df = df

    def complete(
        self,
        *columns: ColumnNameOrSelector,
        fill_value: dict | float | int | str = None,
        explicit: bool = True,
        sort: bool = False,
    ) -> pl.DataFrame:
        """
        Complete a data frame with missing combinations of data.

        It is modeled after tidyr's `complete` function.
        In a way, it is the inverse of `pl.drop_nulls`,
        as it exposes implicitly missing rows.

        Combinations of column names or a list/tuple of column names, or even a
        dictionary of column names and new values are possible.
        If a dictionary is passed,
        the user is required to ensure that the values are unique.
        The keys in a dictionary must be present in the DataFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame(
            ...     {
            ...         "Aloha": range(3),
            ...         "Bell Chart": range(3),
            ...         "Animals@#$%^": range(3)
            ...     }
            ... )
            >>> df
            shape: (3, 3)
            ┌───────┬────────────┬──────────────┐
            │ Aloha ┆ Bell Chart ┆ Animals@#$%^ │
            │ ---   ┆ ---        ┆ ---          │
            │ i64   ┆ i64        ┆ i64          │
            ╞═══════╪════════════╪══════════════╡
            │ 0     ┆ 0          ┆ 0            │
            │ 1     ┆ 1          ┆ 1            │
            │ 2     ┆ 2          ┆ 2            │
            └───────┴────────────┴──────────────┘
            >>> df.janitor.clean_names(remove_special=True)
            shape: (3, 3)
            ┌───────┬────────────┬─────────┐
            │ aloha ┆ bell_chart ┆ animals │
            │ ---   ┆ ---        ┆ ---     │
            │ i64   ┆ i64        ┆ i64     │
            ╞═══════╪════════════╪═════════╡
            │ 0     ┆ 0          ┆ 0       │
            │ 1     ┆ 1          ┆ 1       │
            │ 2     ┆ 2          ┆ 2       │
            └───────┴────────────┴─────────┘

        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It could be column labels (string type),
                a list/tuple of column labels, or a dictionary that pairs
                column labels with new values.
            fill_value: Scalar value to use instead of nulls
                for missing combinations. A dictionary, mapping columns names
                to a scalar value is also accepted.
            explicit: Determines if only implicitly missing values
                should be filled (`False`), or all nulls existing in the dataframe
                (`True`). `explicit` is applicable only
                if `fill_value` is not `None`.
            sort: Sort DataFrame based on *columns.
        Returns:
            A polars DataFrame.
        """  # noqa: E501
        return _complete(
            df=self._df,
            columns=columns,
            fill_value=fill_value,
            explicit=explicit,
            sort=sort,
        )
