from __future__ import annotations

from janitor.utils import import_message

from .clean_names import _clean_expr_names

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
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
