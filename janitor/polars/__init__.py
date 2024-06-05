from __future__ import annotations

from typing import Any

from polars.type_aliases import ColumnNameOrSelector

from janitor.utils import import_message

from .clean_names import _clean_column_names, _clean_expr_names
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

    def clean_names(
        self,
        strip_underscores: str | bool = None,
        case_type: str = "lower",
        remove_special: bool = False,
        strip_accents: bool = False,
        truncate_limit: int = None,
    ) -> pl.DataFrame:
        """
        Clean the column names in a polars DataFrame.

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
            strip_underscores: Removes the outer underscores from all
                column names. Default None keeps outer underscores. Values can be
                either 'left', 'right' or 'both' or the respective shorthand 'l',
                'r' and True.
            case_type: Whether to make the column names lower or uppercase.
                Current case may be preserved with 'preserve',
                while snake case conversion (from CamelCase or camelCase only)
                can be turned on using "snake".
                Default 'lower' makes all characters lowercase.
            remove_special: Remove special characters from the column names.
                Only letters, numbers and underscores are preserved.
            strip_accents: Whether or not to remove accents from
                the labels.
            truncate_limit: Truncates formatted column names to
                the specified length. Default None does not truncate.

        Returns:
            A polars DataFrame.
        """  # noqa: E501
        return self._df.rename(
            lambda col: _clean_column_names(
                obj=col,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                case_type=case_type,
                remove_special=remove_special,
                truncate_limit=truncate_limit,
            )
        )

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

        If the combination involves multiple columns, pass it as a struct,
        with an alias - the name of the struct should not exist in the DataFrame.

        If new values need to be introduced, a polars Expression
        with the new values can be passed, as long as the polars Expression
        has a name that already exists in the DataFrame.

        It is up to the user to ensure that the polars expression returns
        unique values and/or sorted values.

        Note that if the polars expression evaluates to a struct,
        then the fields, not the name, should already exist in the DataFrame.

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
            ...     df.janitor.complete(
            ...         "group",
            ...         pl.struct("item_id", "item_name").unique().sort().alias("rar"),
            ...         sort=True
            ...     )
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
            ...         pl.struct("item_id", "item_name").unique().sort().alias('rar'),
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
            ...         pl.struct("item_id", "item_name").unique().sort().alias('rar'),
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

            Expose missing rows per group:
            >>> df = pl.DataFrame(
            ...     {
            ...         "state": ["CA", "CA", "HI", "HI", "HI", "NY", "NY"],
            ...         "year": [2010, 2013, 2010, 2012, 2016, 2009, 2013],
            ...         "value": [1, 3, 1, 2, 3, 2, 5],
            ...     }
            ... )
            >>> df
            shape: (7, 3)
            ┌───────┬──────┬───────┐
            │ state ┆ year ┆ value │
            │ ---   ┆ ---  ┆ ---   │
            │ str   ┆ i64  ┆ i64   │
            ╞═══════╪══════╪═══════╡
            │ CA    ┆ 2010 ┆ 1     │
            │ CA    ┆ 2013 ┆ 3     │
            │ HI    ┆ 2010 ┆ 1     │
            │ HI    ┆ 2012 ┆ 2     │
            │ HI    ┆ 2016 ┆ 3     │
            │ NY    ┆ 2009 ┆ 2     │
            │ NY    ┆ 2013 ┆ 5     │
            └───────┴──────┴───────┘
            >>> low = pl.col('year').min()
            >>> high = pl.col('year').max().add(1)
            >>> new_year_values=pl.int_range(low,high).alias('year')
            >>> with pl.Config(tbl_rows=-1):
            ...     df.janitor.complete(new_year_values,by='state',sort=True)
            shape: (16, 3)
            ┌───────┬──────┬───────┐
            │ state ┆ year ┆ value │
            │ ---   ┆ ---  ┆ ---   │
            │ str   ┆ i64  ┆ i64   │
            ╞═══════╪══════╪═══════╡
            │ CA    ┆ 2010 ┆ 1     │
            │ CA    ┆ 2011 ┆ null  │
            │ CA    ┆ 2012 ┆ null  │
            │ CA    ┆ 2013 ┆ 3     │
            │ HI    ┆ 2010 ┆ 1     │
            │ HI    ┆ 2011 ┆ null  │
            │ HI    ┆ 2012 ┆ 2     │
            │ HI    ┆ 2013 ┆ null  │
            │ HI    ┆ 2014 ┆ null  │
            │ HI    ┆ 2015 ┆ null  │
            │ HI    ┆ 2016 ┆ 3     │
            │ NY    ┆ 2009 ┆ 2     │
            │ NY    ┆ 2010 ┆ null  │
            │ NY    ┆ 2011 ┆ null  │
            │ NY    ┆ 2012 ┆ null  │
            │ NY    ┆ 2013 ┆ 5     │
            └───────┴──────┴───────┘


        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It can be a string or a column selector or a polars expression.
                A polars expression can be used to introduced new values,
                as long as the polars expression has a name that already exists
                in the DataFrame.
                It is up to the user to ensure that the polars expression returns
                unique values.
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

    def clean_names(
        self,
        strip_underscores: str | bool = None,
        case_type: str = "lower",
        remove_special: bool = False,
        strip_accents: bool = False,
        truncate_limit: int = None,
    ) -> pl.LazyFrame:
        """
        Clean the column names in a polars LazyFrame.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.LazyFrame(
            ...     {
            ...         "Aloha": range(3),
            ...         "Bell Chart": range(3),
            ...         "Animals@#$%^": range(3)
            ...     }
            ... )
            >>> df.collect()
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
            >>> df.janitor.clean_names(remove_special=True).collect()
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
            strip_underscores: Removes the outer underscores from all
                column names. Default None keeps outer underscores. Values can be
                either 'left', 'right' or 'both' or the respective shorthand 'l',
                'r' and True.
            case_type: Whether to make the column names lower or uppercase.
                Current case may be preserved with 'preserve',
                while snake case conversion (from CamelCase or camelCase only)
                can be turned on using "snake".
                Default 'lower' makes all characters lowercase.
            remove_special: Remove special characters from the column names.
                Only letters, numbers and underscores are preserved.
            strip_accents: Whether or not to remove accents from
                the labels.
            truncate_limit: Truncates formatted column names to
                the specified length. Default None does not truncate.

        Returns:
            A polars LazyFrame.
        """  # noqa: E501
        return self._df.rename(
            lambda col: _clean_column_names(
                obj=col,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                case_type=case_type,
                remove_special=remove_special,
                truncate_limit=truncate_limit,
            )
        )

    def complete(
        self,
        *columns: ColumnNameOrSelector,
        fill_value: dict | Any | pl.Expr = None,
        explicit: bool = True,
        sort: bool = False,
        by: ColumnNameOrSelector = None,
    ) -> pl.LazyFrame:
        """
        Turns implicit missing values into explicit missing values.

        It is modeled after tidyr's `complete` function.
        In a way, it is the inverse of `pl.drop_nulls`,
        as it exposes implicitly missing rows.

        If the combination involves multiple columns, pass it as a struct,
        with an alias - the name of the struct should not exist in the LazyFrame.

        If new values need to be introduced, a polars Expression
        with the new values can be passed, as long as the polars Expression
        has a name that already exists in the LazyFrame.

        It is up to the user to ensure that the polars expression returns
        unique values.

        Note that if the polars expression evaluates to a struct,
        then the fields, not the name, should already exist in the LazyFrame.

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

        !!! info "New in version 0.28.0"

        Args:
            *columns: This refers to the columns to be completed.
                It can be a string or a column selector or a polars expression.
                A polars expression can be used to introduced new values,
                as long as the polars expression has a name that already exists
                in the LazyFrame.
                It is up to the user to ensure that the polars expression returns
                unique values.
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


@pl.api.register_expr_namespace("janitor")
class PolarsExpr:
    def __init__(self, expr: pl.Expr) -> pl.Expr:
        self._expr = expr

    def clean_names(
        self,
        strip_underscores: str | bool = None,
        case_type: str = "lower",
        remove_special: bool = False,
        strip_accents: bool = False,
        enforce_string: bool = False,
        truncate_limit: int = None,
    ) -> pl.Expr:
        """
        Clean the labels in a polars Expression.

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame({"raw": ["Abçdê fgí j"]})
            >>> df
            shape: (1, 1)
            ┌─────────────┐
            │ raw         │
            │ ---         │
            │ str         │
            ╞═════════════╡
            │ Abçdê fgí j │
            └─────────────┘

            Clean the column values:
            >>> df.with_columns(pl.col("raw").janitor.clean_names(strip_accents=True))
            shape: (1, 1)
            ┌─────────────┐
            │ raw         │
            │ ---         │
            │ str         │
            ╞═════════════╡
            │ abcde_fgi_j │
            └─────────────┘

        !!! info "New in version 0.28.0"

        Args:
            strip_underscores: Removes the outer underscores
                from all labels in the expression.
                Default None keeps outer underscores.
                Values can be either 'left', 'right'
                or 'both' or the respective shorthand 'l',
                'r' and True.
            case_type: Whether to make the labels in the expression lower or uppercase.
                Current case may be preserved with 'preserve',
                while snake case conversion (from CamelCase or camelCase only)
                can be turned on using "snake".
                Default 'lower' makes all characters lowercase.
            remove_special: Remove special characters from the values in the expression.
                Only letters, numbers and underscores are preserved.
            strip_accents: Whether or not to remove accents from
                the expression.
            enforce_string: Whether or not to cast the expression to a string type.
            truncate_limit: Truncates formatted labels in the expression to
                the specified length. Default None does not truncate.

        Returns:
            A polars Expression.
        """
        return _clean_expr_names(
            obj=self._expr,
            strip_accents=strip_accents,
            strip_underscores=strip_underscores,
            case_type=case_type,
            remove_special=remove_special,
            enforce_string=enforce_string,
            truncate_limit=truncate_limit,
        )
