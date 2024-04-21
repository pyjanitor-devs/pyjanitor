"""functions for polars."""

import re
import unicodedata
from typing import Optional, Union

from janitor.errors import JanitorError
from janitor.functions.utils import (
    _change_case,
    _normalize_1,
    _remove_special,
    _strip_accents,
    _strip_underscores_func,
)
from janitor.utils import import_message

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
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
    strip_underscores: Union[str, bool] = None,
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
    strip_underscores: Optional[Union[str, bool]] = None,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_accents: bool = False,
    truncate_limit: int = None,
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
    strip_underscores: Optional[Union[str, bool]] = None,
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
