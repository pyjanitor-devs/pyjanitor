"""Functions for cleaning columns names."""
import re
from typing import Hashable, Optional, Union
import pandas as pd
import pandas_flavor as pf

from janitor.errors import JanitorError
import unicodedata


@pf.register_dataframe_method
def clean_names(
    df: pd.DataFrame,
    strip_underscores: Optional[Union[str, bool]] = None,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_accents: bool = True,
    preserve_original_columns: bool = True,
    enforce_string: bool = True,
    truncate_limit: int = None,
) -> pd.DataFrame:
    """
    Clean column names.

    Takes all column names, converts them to lowercase,
    then replaces all spaces with underscores.

    By default, column names are converted to string types.
    This can be switched off by passing in `enforce_string=False`.

    This method does not mutate the original DataFrame.

    Example usage:

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

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer underscores from all
        column names. Default None keeps outer underscores. Values can be
        either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
        and True.
    :param case_type: (optional) Whether to make columns lower or uppercase.
        Current case may be preserved with 'preserve',
        while snake case conversion (from CamelCase or camelCase only)
        can be turned on using "snake".
        Default 'lower' makes all characters lowercase.
    :param remove_special: (optional) Remove special characters from columns.
        Only letters, numbers and underscores are preserved.
    :param strip_accents: Whether or not to remove accents from
        columns names.
    :param preserve_original_columns: (optional) Preserve original names.
        This is later retrievable using `df.original_columns`.
    :param enforce_string: Whether or not to convert all column names
        to string type. Defaults to True, but can be turned off.
        Columns with >1 levels will not be converted by default.
    :param truncate_limit: (optional) Truncates formatted column names to
        the specified length. Default None does not truncate.
    :returns: A pandas DataFrame.
    """
    original_column_names = list(df.columns)

    if enforce_string:
        df = df.rename(columns=str)

    df = df.rename(columns=lambda x: _change_case(x, case_type))

    df = df.rename(columns=_normalize_1)

    if remove_special:
        df = df.rename(columns=_remove_special)

    if strip_accents:
        df = df.rename(columns=_strip_accents)

    df = df.rename(columns=lambda x: re.sub("_+", "_", x))  # noqa: PD005
    df = _strip_underscores(df, strip_underscores)

    df = df.rename(columns=lambda x: x[:truncate_limit])

    # Store the original column names, if enabled by user
    if preserve_original_columns:
        df.__dict__["original_columns"] = original_column_names
    return df


def _change_case(col: str, case_type: str) -> str:
    """Change case of a column name."""
    case_types = ["preserve", "upper", "lower", "snake"]
    if case_type.lower() not in case_types:
        raise JanitorError(f"case_type must be one of: {case_types}")

    if case_type.lower() != "preserve":
        if case_type.lower() == "upper":
            col = col.upper()
        elif case_type.lower() == "lower":
            col = col.lower()
        elif case_type.lower() == "snake":
            col = _camel2snake(col)

    return col


def _remove_special(col_name: Hashable) -> str:
    """Remove special characters from column name."""
    return "".join(
        item for item in str(col_name) if item.isalnum() or "_" in item
    )


_underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
_underscorer2 = re.compile("([a-z0-9])([A-Z])")


def _camel2snake(col_name: str) -> str:
    """Convert camelcase names to snake case.

    Implementation taken from: https://gist.github.com/jaytaylor/3660565
    by @jtaylor
    """
    subbed = _underscorer1.sub(r"\1_\2", col_name)  # noqa: PD005
    return _underscorer2.sub(r"\1_\2", subbed).lower()  # noqa: PD005


FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", ""), (r"[\xa0]", "_")]


def _normalize_1(col_name: Hashable) -> str:
    """Perform normalization of column name."""
    result = str(col_name)
    for search, replace in FIXES:
        result = re.sub(search, replace, result)  # noqa: PD005
    return result


def _strip_accents(col_name: str) -> str:
    """Remove accents from a DataFrame column name.

    Inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-strin
    """  # noqa: E501

    return "".join(
        letter
        for letter in unicodedata.normalize("NFD", col_name)
        if not unicodedata.combining(letter)
    )


def _strip_underscores(
    df: pd.DataFrame, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
    """
    Strip underscores from DataFrames column names.

    Underscores can be stripped from the beginning, end or both.

    Example usage:

    ```
    df = _strip_underscores(df, strip_underscores='left')
    ```

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer underscores from all
        column names. Default `None` keeps outer underscores. Values can be
        either `'left'`, `'right'` or `'both'` or the respective shorthand
        `'l'`, `'r'` and `True`.
    :returns: A pandas DataFrame with underscores removed.
    """
    df = df.rename(
        columns=lambda x: _strip_underscores_func(x, strip_underscores)
    )
    return df


def _strip_underscores_func(
    col: str, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
    """Strip underscores from a string."""
    underscore_options = [None, "left", "right", "both", "l", "r", True]
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )

    if strip_underscores in ["left", "l"]:
        col = col.lstrip("_")
    elif strip_underscores in ["right", "r"]:
        col = col.rstrip("_")
    elif strip_underscores == "both" or strip_underscores is True:
        col = col.strip("_")
    return col
