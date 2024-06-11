from __future__ import annotations

from janitor.utils import import_message

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

    def convert_excel_date(self) -> pl.Expr:
        """
        Convert Excel's serial date format into Python datetime format.

        Inspiration is from
        [Stack Overflow](https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas).

        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame({"date": [39690, 39690, 37118]})
            >>> df
            shape: (3, 1)
            ┌───────┐
            │ date  │
            │ ---   │
            │ i64   │
            ╞═══════╡
            │ 39690 │
            │ 39690 │
            │ 37118 │
            └───────┘
            >>> expression = pl.col('date').janitor.convert_excel_date().alias('date_')
            >>> df.with_columns(expression)
            shape: (3, 2)
            ┌───────┬────────────┐
            │ date  ┆ date_      │
            │ ---   ┆ ---        │
            │ i64   ┆ date       │
            ╞═══════╪════════════╡
            │ 39690 ┆ 2008-08-30 │
            │ 39690 ┆ 2008-08-30 │
            │ 37118 ┆ 2001-08-15 │
            └───────┴────────────┘

        !!! info "New in version 0.28.0"

        Returns:
            A polars Expression.
        """  # noqa: E501
        expression = pl.duration(days=self._expr)
        expression += pl.date(year=1899, month=12, day=30)
        return expression

    def convert_matlab_date(self) -> pl.Expr:
        """
        Convert Matlab's serial date number into Python datetime format.

        Implementation is from
        [Stack Overflow](https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python).


        Examples:
            >>> import polars as pl
            >>> import janitor.polars
            >>> df = pl.DataFrame({"date": [737125.0, 737124.815863, 737124.4985, 737124]})
            >>> df
            shape: (4, 1)
            ┌───────────────┐
            │ date          │
            │ ---           │
            │ f64           │
            ╞═══════════════╡
            │ 737125.0      │
            │ 737124.815863 │
            │ 737124.4985   │
            │ 737124.0      │
            └───────────────┘
            >>> expression = pl.col('date').janitor.convert_matlab_date().alias('date_')
            >>> df.with_columns(expression)
            shape: (4, 2)
            ┌───────────────┬─────────────────────────┐
            │ date          ┆ date_                   │
            │ ---           ┆ ---                     │
            │ f64           ┆ datetime[μs]            │
            ╞═══════════════╪═════════════════════════╡
            │ 737125.0      ┆ 2018-03-06 00:00:00     │
            │ 737124.815863 ┆ 2018-03-05 19:34:50.563 │
            │ 737124.4985   ┆ 2018-03-05 11:57:50.399 │
            │ 737124.0      ┆ 2018-03-05 00:00:00     │
            └───────────────┴─────────────────────────┘

        !!! info "New in version 0.28.0"

        Returns:
            A polars Expression.
        """  # noqa: E501
        # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
        expression = self._expr.sub(719529).mul(86_400_000)
        expression = pl.duration(milliseconds=expression)
        expression += pl.datetime(year=1970, month=1, day=1)
        return expression


from .dataframe import PolarsDataFrame
from .expressions import PolarsExpr
from .lazyframe import PolarsLazyFrame
from .pivot_longer import pivot_longer_spec

__all__ = [
    "pivot_longer_spec",
    "clean_names",
    "PolarsDataFrame",
    "PolarsLazyFrame",
    "PolarsExpr",
]
