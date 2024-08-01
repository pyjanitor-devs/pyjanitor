"""Functions for cleaning columns/index names and/or column values."""

from __future__ import annotations

import unicodedata

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar

from janitor.errors import JanitorError
from janitor.functions.select import get_index_labels
from janitor.functions.utils import _is_str_or_cat
from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(preserve_original_columns="preserve_original_labels")
def clean_names(
    df: pd.DataFrame,
    axis: str = "columns",
    column_names: str | list = None,
    strip_underscores: str | bool = None,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_accents: bool = True,
    preserve_original_labels: bool = True,
    enforce_string: bool = True,
    truncate_limit: int = None,
) -> pd.DataFrame:
    """Clean column/index names. It can also be applied to column values.

    Takes all column names, converts them to lowercase,
    then replaces all spaces with underscores.

    By default, column names are converted to string types.
    This can be switched off by passing in `enforce_string=False`.

    This method does not mutate the original DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame(
        ...     {
        ...         "Aloha": range(3),
        ...         "Bell Chart": range(3),
        ...         "Animals@#$%^": range(3)
        ...     }
        ... )
        >>> df
           Aloha  Bell Chart  Animals@#$%^
        0      0           0             0
        1      1           1             1
        2      2           2             2
        >>> df.clean_names()
           aloha  bell_chart  animals@#$%^
        0      0           0             0
        1      1           1             1
        2      2           2             2
        >>> df.clean_names(remove_special=True)
           aloha  bell_chart  animals
        0      0           0        0
        1      1           1        1
        2      2           2        2

    !!! summary "Version Changed"

        - 0.26.0
             - Added `axis` and `column_names` parameters.

    Args:
        df: The pandas DataFrame object.
        axis: Whether to clean the labels on the index or columns.
            If `None`, applies to a defined column
            or columns in `column_names`.
        column_names: Clean the values in a column.
            `axis` should be `None`.
            Column selection is possible using the
            [`select`][janitor.functions.select.select] syntax.
        strip_underscores: Removes the outer underscores from all
            column names/values. Default None keeps outer underscores.
            Values can be either 'left', 'right' or 'both'
            or the respective shorthand 'l',
            'r' and True.
        case_type: Whether to make columns lower or uppercase.
            Current case may be preserved with 'preserve',
            while snake case conversion (from CamelCase or camelCase only)
            can be turned on using "snake".
            Default 'lower' makes all characters lowercase.
        remove_special: Remove special characters from columns.
            Only letters, numbers and underscores are preserved.
        strip_accents: Whether or not to remove accents from
            columns names/values.
        preserve_original_labels: Preserve original names.
            This is later retrievable using `df.original_labels`.
            Applies if `axis` is not None.
        enforce_string: Whether or not to convert all
            column names/values to string type.
            Defaults to True, but can be turned off.
            Columns with >1 levels will not be converted by default.
        truncate_limit: Truncates formatted column names/values
            to the specified length.
            Default None does not truncate.

    Raises:
        ValueError: If `axis=None` and `column_names=None`.

    Returns:
        A pandas DataFrame.
    """
    if not axis and not column_names:
        raise ValueError(
            "Kindly provide an argument to `column_names`, if axis is None."
        )
    if axis is None:
        column_names = get_index_labels(
            arg=column_names, df=df, axis="columns"
        )
        if is_scalar(column_names):
            column_names = [column_names]
        df = df.copy()
        for column_name in column_names:
            df[column_name] = _clean_names(
                obj=df[column_name],
                enforce_string=enforce_string,
                case_type=case_type,
                remove_special=remove_special,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                truncate_limit=truncate_limit,
            )
        return df

    assert axis in {"index", "columns"}
    df = df[:]
    target_axis = getattr(df, axis)
    if isinstance(target_axis, pd.MultiIndex):
        target_axis = [
            target_axis.get_level_values(number)
            for number in range(target_axis.nlevels)
        ]
        target_axis = [
            _clean_names(
                obj=obj,
                enforce_string=enforce_string,
                case_type=case_type,
                remove_special=remove_special,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                truncate_limit=truncate_limit,
            )
            for obj in target_axis
        ]
    else:
        target_axis = _clean_names(
            obj=target_axis,
            enforce_string=enforce_string,
            case_type=case_type,
            remove_special=remove_special,
            strip_accents=strip_accents,
            strip_underscores=strip_underscores,
            truncate_limit=truncate_limit,
        )
    # Store the original column names, if enabled by user
    if preserve_original_labels:
        df.__dict__["original_labels"] = getattr(df, axis)
    setattr(df, axis, target_axis)
    return df


def _clean_names(
    obj: pd.Index | pd.Series,
    strip_underscores: str | bool,
    case_type: str,
    remove_special: bool,
    strip_accents: bool,
    enforce_string: bool,
    truncate_limit: int,
) -> pd.Index | pd.Series:
    """
    Generic function to clean labels in a pandas object.
    """
    if enforce_string and not _is_str_or_cat(obj):
        obj = obj.astype(str)
    obj = _change_case(obj=obj, case_type=case_type)
    obj = _normalize_1(obj=obj)
    if remove_special:
        obj = obj.str.replace(
            pat="[^A-Za-z_\\d]", repl="", regex=True
        ).str.strip()
    if strip_accents:
        obj = _strip_accents(obj=obj)
    obj = obj.str.replace(pat="_+", repl="_", regex=True)
    obj = _strip_underscores_func(
        obj,
        strip_underscores=strip_underscores,
    )
    if truncate_limit:
        obj = obj.str[:truncate_limit]
    return obj


def _change_case(
    obj: pd.Index | pd.Series,
    case_type: str,
) -> pd.Index | pd.Series:
    """Change case of labels in obj."""
    case_types = {"preserve", "upper", "lower", "snake"}
    case_type = case_type.lower()
    if case_type not in case_types:
        raise JanitorError(f"case_type must be one of: {case_types}")

    if case_type == "preserve":
        return obj
    if case_type == "upper":
        return obj.str.upper()
    if case_type == "lower":
        return obj.str.lower()
    # Implementation taken from: https://gist.github.com/jaytaylor/3660565
    # by @jtaylor
    return (
        obj.str.replace(pat=r"(.)([A-Z][a-z]+)", repl=r"\1_\2", regex=True)
        .str.replace(pat=r"([a-z0-9])([A-Z])", repl=r"\1_\2", regex=True)
        .str.lower()
    )


def _normalize_1(obj: pd.Index | pd.Series) -> pd.Index | pd.Series:
    """Perform normalization of labels in obj."""
    FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", ""), (r"[\xa0]", "_")]
    for search, replace in FIXES:
        obj = obj.str.replace(pat=search, repl=replace, regex=True)

    return obj


def _strip_accents(
    obj: pd.Index | pd.Series,
) -> pd.Index | pd.Series:
    """Remove accents from a label.

    Inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-strin
    """  # noqa: E501
    return obj.map(
        lambda f: "".join(
            [
                letter
                for letter in unicodedata.normalize("NFD", str(f))
                if not unicodedata.combining(letter)
            ]
        )
    )


def _strip_underscores_func(
    obj: pd.Index | pd.Series,
    strip_underscores: str | bool = None,
) -> pd.Index | pd.Series:
    """Strip underscores."""
    underscore_options = {None, "left", "right", "both", "l", "r", True}
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )
    if strip_underscores in {"left", "l"}:
        return obj.str.lstrip("_")
    if strip_underscores in {"right", "r"}:
        return obj.str.rstrip("_")
    if strip_underscores in {True, "both"}:
        return obj.str.strip("_")
    return obj
