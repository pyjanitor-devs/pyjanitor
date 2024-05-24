from __future__ import annotations

from typing import Any

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
        fill_value: dict | Any | pl.Expr = None,
        explicit: bool = True,
        sort: bool = False,
        by: ColumnNameOrSelector = None,
    ) -> pl.DataFrame:
        """
        Turns implicit missing values into explicit missing values

        It is modeled after tidyr's `complete` function.
        In a way, it is the inverse of `pl.drop_nulls`,
        as it exposes implicitly missing rows.

        If the combination involves multiple columns, pass it as a struct.
        If new values need to be introduced, a polars Expression
        with the new values can be passed, as long as the polars Expression
        has a name that already exists in the DataFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame(
            ...     dict(
            ...         group=(1, 2, 1, 2),
            ...         item_id=(1, 2, 2, 3),
            ...         item_name=("a", "a", "b", "b"),
            ...         value1=(1, None, 3, 4),
            ...         value2=range(4, 8),
            ...     )
            ... )
            >>> df
            shape: (4, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Generate all possible combinations of
            `group`, `item_id`, and `item_name`
            (whether or not they appear in the data)
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete("group", "item_id", "item_name", sort=True)
            shape: (12, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 1       ┆ b         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 3       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Cross all possible `group` values with the unique pairs of
            `(item_id, item_name)` that already exist in the data.
            For such situations, where there is a group of columns,
            pass it in as a struct:
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete("group", pl.struct("item_id", "item_name"), sort=True)
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Fill in nulls:
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name"),
            ...         fill_value={"value1": 0, "value2": 99},
            ...         explicit=True,
            ...         sort=True,
            ...     )
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ 0      ┆ 99     │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 1       ┆ a         ┆ 0      ┆ 99     │
            │ 2     ┆ 2       ┆ a         ┆ 0      ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Limit the fill to only the newly created
            missing values with `explicit = FALSE`
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name"),
            ...         fill_value={"value1": 0, "value2": 99},
            ...         explicit=False,
            ...         sort=True,
            ...     )
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ 0      ┆ 99     │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 1       ┆ a         ┆ 0      ┆ 99     │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            >>> df = pl.DataFrame(
            ...     {
            ...         "Year": [1999, 2000, 2004, 1999, 2004],
            ...         "Taxon": [
            ...             "Saccharina",
            ...             "Saccharina",
            ...             "Saccharina",
            ...             "Agarum",
            ...             "Agarum",
            ...         ],
            ...         "Abundance": [4, 5, 2, 1, 8],
            ...     }
            ... )
            >>> df
            shape: (5, 3)
            ┌──────┬────────────┬───────────┐
            │ Year ┆ Taxon      ┆ Abundance │
            │ ---  ┆ ---        ┆ ---       │
            │ i64  ┆ str        ┆ i64       │
            ╞══════╪════════════╪═══════════╡
            │ 1999 ┆ Saccharina ┆ 4         │
            │ 2000 ┆ Saccharina ┆ 5         │
            │ 2004 ┆ Saccharina ┆ 2         │
            │ 1999 ┆ Agarum     ┆ 1         │
            │ 2004 ┆ Agarum     ┆ 8         │
            └──────┴────────────┴───────────┘

            Expose missing years from 1999 to 2004 -
            pass a polars expression with the new dates,
            and ensure the expression's name already exists
            in the DataFrame:
            >>> expression = pl.int_range(1999,2005).alias('Year')
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(expression,'Taxon',sort=True)
            shape: (12, 3)
            ┌──────┬────────────┬───────────┐
            │ Year ┆ Taxon      ┆ Abundance │
            │ ---  ┆ ---        ┆ ---       │
            │ i64  ┆ str        ┆ i64       │
            ╞══════╪════════════╪═══════════╡
            │ 1999 ┆ Agarum     ┆ 1         │
            │ 1999 ┆ Saccharina ┆ 4         │
            │ 2000 ┆ Agarum     ┆ null      │
            │ 2000 ┆ Saccharina ┆ 5         │
            │ 2001 ┆ Agarum     ┆ null      │
            │ 2001 ┆ Saccharina ┆ null      │
            │ 2002 ┆ Agarum     ┆ null      │
            │ 2002 ┆ Saccharina ┆ null      │
            │ 2003 ┆ Agarum     ┆ null      │
            │ 2003 ┆ Saccharina ┆ null      │
            │ 2004 ┆ Agarum     ┆ 8         │
            │ 2004 ┆ Saccharina ┆ 2         │
            └──────┴────────────┴───────────┘

        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It can be a string or a column selector or a polars expression.
                A polars expression can be used to introduced new values,
                as long as the polars expression has a name that already exists
                in the DataFrame.
            fill_value: Scalar value or polars expression to use instead of nulls
                for missing combinations. A dictionary, mapping columns names
                to a scalar value is also accepted.
            explicit: Determines if only implicitly missing values
                should be filled (`False`), or all nulls existing in the LazyFrame
                (`True`). `explicit` is applicable only
                if `fill_value` is not `None`.
            sort: Sort the DataFrame based on *columns.
            by: Column(s) to group by.
                The explicit missing rows are returned per group.
        Returns:
            A polars DataFrame.
        """  # noqa: E501
        return _complete(
            df=self._df,
            columns=columns,
            fill_value=fill_value,
            explicit=explicit,
            sort=sort,
            by=by,
        )


@pl.api.register_lazyframe_namespace("janitor")
class PolarsLazyFrame:
    def __init__(self, df: pl.LazyFrame) -> pl.LazyFrame:
        self._df = df

    def complete(
        self,
        *columns: ColumnNameOrSelector,
        fill_value: dict | Any | pl.Expr = None,
        explicit: bool = True,
        sort: bool = False,
        by: ColumnNameOrSelector = None,
    ) -> pl.DataFrame:
        """
        Turns implicit missing values into explicit missing values

        It is modeled after tidyr's `complete` function.
        In a way, it is the inverse of `pl.drop_nulls`,
        as it exposes implicitly missing rows.

        If the combination involves multiple columns, pass it as a struct.
        If new values need to be introduced, a polars Expression
        with the new values can be passed, as long as the polars Expression
        has a name that already exists in the LazyFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.LazyFrame(
            ...     dict(
            ...         group=(1, 2, 1, 2),
            ...         item_id=(1, 2, 2, 3),
            ...         item_name=("a", "a", "b", "b"),
            ...         value1=(1, None, 3, 4),
            ...         value2=range(4, 8),
            ...     )
            ... )
            >>> df.collect()
            shape: (4, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Generate all possible combinations of
            `group`, `item_id`, and `item_name`
            (whether or not they appear in the data)
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete("group", "item_id", "item_name", sort=True).collect()
            shape: (12, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 1       ┆ b         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 3       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Cross all possible `group` values with the unique pairs of
            `(item_id, item_name)` that already exist in the data.
            For such situations, where there is a group of columns,
            pass it in as a struct:
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete("group", pl.struct("item_id", "item_name"), sort=True).collect()
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ null   ┆ null   │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 1       ┆ a         ┆ null   ┆ null   │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ null   ┆ null   │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Fill in nulls:
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name"),
            ...         fill_value={"value1": 0, "value2": 99},
            ...         explicit=True,
            ...         sort=True,
            ...     ).collect()
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ 0      ┆ 99     │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 1       ┆ a         ┆ 0      ┆ 99     │
            │ 2     ┆ 2       ┆ a         ┆ 0      ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

            Limit the fill to only the newly created
            missing values with `explicit = FALSE`
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name"),
            ...         fill_value={"value1": 0, "value2": 99},
            ...         explicit=False,
            ...         sort=True,
            ...     ).collect()
            shape: (8, 5)
            ┌───────┬─────────┬───────────┬────────┬────────┐
            │ group ┆ item_id ┆ item_name ┆ value1 ┆ value2 │
            │ ---   ┆ ---     ┆ ---       ┆ ---    ┆ ---    │
            │ i64   ┆ i64     ┆ str       ┆ i64    ┆ i64    │
            ╞═══════╪═════════╪═══════════╪════════╪════════╡
            │ 1     ┆ 1       ┆ a         ┆ 1      ┆ 4      │
            │ 1     ┆ 2       ┆ a         ┆ 0      ┆ 99     │
            │ 1     ┆ 2       ┆ b         ┆ 3      ┆ 6      │
            │ 1     ┆ 3       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 1       ┆ a         ┆ 0      ┆ 99     │
            │ 2     ┆ 2       ┆ a         ┆ null   ┆ 5      │
            │ 2     ┆ 2       ┆ b         ┆ 0      ┆ 99     │
            │ 2     ┆ 3       ┆ b         ┆ 4      ┆ 7      │
            └───────┴─────────┴───────────┴────────┴────────┘

        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It can be a string or a column selector or a polars expression.
                A polars expression can be used to introduced new values,
                as long as the polars expression has a name that already exists
                in the LazyFrame.
            fill_value: Scalar value or polars expression to use instead of nulls
                for missing combinations. A dictionary, mapping columns names
                to a scalar value is also accepted.
            explicit: Determines if only implicitly missing values
                should be filled (`False`), or all nulls existing in the LazyFrame
                (`True`). `explicit` is applicable only
                if `fill_value` is not `None`.
            sort: Sort the LazyFrame based on *columns.
            by: Column(s) to group by.
                The explicit missing rows are returned per group.
        Returns:
            A polars LazyFrame.
        """  # noqa: E501
        return _complete(
            df=self._df,
            columns=columns,
            fill_value=fill_value,
            explicit=explicit,
            sort=sort,
            by=by,
        )
