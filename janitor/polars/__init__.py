from typing import Optional, Union

from janitor.utils import import_message

from .functions import _clean_names

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
class Frame:
    def __init__(self, df: pl.DataFrame) -> pl.DataFrame:
        self._df = df

    def clean_names(
        self,
        strip_underscores: Optional[Union[str, bool]] = None,
        case_type: str = "lower",
        remove_special: bool = False,
        strip_accents: bool = False,
        enforce_string: bool = False,
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
        """
        return self._df.rename(
            lambda col: _clean_names(
                obj=col,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                case_type=case_type,
                remove_special=remove_special,
                enforce_string=enforce_string,
                truncate_limit=truncate_limit,
            )
        )


@pl.api.register_expr_namespace("janitor")
class PolarsExpr:
    def __init__(self, expr: pl.Expr) -> pl.Expr:
        self._expr = expr

    def clean_names(
        self,
        strip_underscores: Optional[Union[str, bool]] = None,
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
        """
        return _clean_names(
            obj=self._expr,
            strip_accents=strip_accents,
            strip_underscores=strip_underscores,
            case_type=case_type,
            remove_special=remove_special,
            enforce_string=enforce_string,
            truncate_limit=truncate_limit,
        )
