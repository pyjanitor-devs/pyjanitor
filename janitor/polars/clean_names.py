"""clean_names implementation for polars."""

from __future__ import annotations

import re
import unicodedata

from janitor.errors import JanitorError
from janitor.functions.utils import (
    _change_case,
    _normalize_1,
    _remove_special,
    _strip_accents,
    _strip_underscores_func,
)
from janitor.utils import import_message

from .polars_flavor import (
    register_dataframe_method,
    register_expr_method,
    register_lazyframe_method,
)

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
def clean_names(
    df: pl.DataFrame | pl.LazyFrame,
    strip_underscores: str | bool = None,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_accents: bool = False,
    truncate_limit: int = None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Clean the column names in a polars DataFrame.

    `clean_names` can also be applied to a LazyFrame.

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
        >>> df.clean_names(remove_special=True)
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
        A polars DataFrame/LazyFrame.
    """  # noqa: E501
    return df.rename(
        lambda col: _clean_column_names(
            obj=col,
            strip_accents=strip_accents,
            strip_underscores=strip_underscores,
            case_type=case_type,
            remove_special=remove_special,
            truncate_limit=truncate_limit,
        )
    )


@register_expr_method
def make_clean_names(
    expression,
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
        >>> df.with_columns(pl.col("raw").make_clean_names(strip_accents=True))
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
        obj=expression,
        strip_accents=strip_accents,
        strip_underscores=strip_underscores,
        case_type=case_type,
        remove_special=remove_special,
        enforce_string=enforce_string,
        truncate_limit=truncate_limit,
    )


def _change_case_expr(
    obj: pl.Expr,
    case_type: str,
) -> pl.Expr:
    """Change case of labels in obj."""
    case_types = {"preserve", "upper", "lower", "snake"}
    case_type = case_type.lower()
    if case_type not in case_types:
        raise JanitorError(f"type must be one of: {case_types}")

    if case_type == "preserve":
        return obj
    if case_type == "upper":
        return obj.str.to_uppercase()
    if case_type == "lower":
        return obj.str.to_lowercase()
    # Implementation taken from: https://gist.github.com/jaytaylor/3660565
    # by @jtaylor
    return (
        obj.str.replace_all(
            pattern=r"(.)([A-Z][a-z]+)", value=r"${1}_${2}", literal=False
        )
        .str.replace_all(
            pattern=r"([a-z0-9])([A-Z])", value=r"${1}_${2}", literal=False
        )
        .str.to_lowercase()
    )


def _normalize_expr(obj: pl.Expr) -> pl.Expr:
    """Perform normalization of labels in obj."""
    FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", ""), (r"[\xa0]", "_")]
    for search, replace in FIXES:
        obj = obj.str.replace_all(pattern=search, value=replace, literal=False)
    return obj


def _remove_special_expr(
    obj: pl.Expr,
) -> pl.Expr:
    """Remove special characters from the labels in obj."""
    return obj.str.replace_all(
        pattern="[^A-Za-z_\\d]", value="", literal=False
    ).str.strip_chars()


def _strip_accents_expr(
    obj: pl.Expr,
) -> pl.Expr:
    """Remove accents from the labels in obj.

    Inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-strin
    """  # noqa: E501
    # TODO: possible implementation in Rust
    # or use a pyarrow implementation?
    # https://github.com/pola-rs/polars/issues/11455
    return obj.map_elements(
        lambda word: [
            letter
            for letter in unicodedata.normalize("NFD", word)
            if not unicodedata.combining(letter)
        ],
        return_dtype=pl.List(pl.Utf8),
    ).list.join("")


def _strip_underscores_func_expr(
    obj: pl.Expr,
    strip_underscores: str | bool = None,
) -> pl.Expr:
    """Strip underscores from obj."""
    underscore_options = {None, "left", "right", "both", "l", "r", True}
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )
    if strip_underscores in {"left", "l"}:
        return obj.str.strip_chars_start("_")
    if strip_underscores in {"right", "r"}:
        return obj.str.strip_chars_end("_")
    if strip_underscores in {True, "both"}:
        return obj.str.strip_chars("_")
    return obj


def _clean_column_names(
    obj: str,
    strip_underscores: str | bool,
    case_type: str,
    remove_special: bool,
    strip_accents: bool,
    truncate_limit: int,
) -> str:
    """
    Function to clean the column names of a polars DataFrame.
    """
    obj = _change_case(obj=obj, case_type=case_type)
    obj = _normalize_1(obj=obj)
    if remove_special:
        obj = _remove_special(obj=obj)
    if strip_accents:
        obj = _strip_accents(obj=obj)
    obj = re.sub(pattern="_+", repl="_", string=obj)
    obj = _strip_underscores_func(
        obj,
        strip_underscores=strip_underscores,
    )
    obj = obj[:truncate_limit]
    return obj


def _clean_expr_names(
    obj: pl.Expr,
    strip_underscores: str | bool = None,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_accents: bool = False,
    enforce_string: bool = False,
    truncate_limit: int = None,
) -> pl.Expr:
    """
    Function to clean the labels of a polars Expression.
    """
    if enforce_string:
        obj = obj.cast(pl.Utf8)
    obj = _change_case_expr(obj=obj, case_type=case_type)
    obj = _normalize_expr(obj=obj)
    if remove_special:
        obj = _remove_special_expr(obj=obj)
    if strip_accents:
        obj = _strip_accents_expr(obj=obj)
    obj = obj.str.replace(pattern="_+", value="_", literal=False)
    obj = _strip_underscores_func_expr(
        obj,
        strip_underscores=strip_underscores,
    )
    if truncate_limit:
        obj = obj.str.slice(offset=0, length=truncate_limit)
    return obj
