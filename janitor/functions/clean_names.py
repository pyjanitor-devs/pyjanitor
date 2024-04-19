"""Functions for cleaning columns/index names and/or column values."""

from typing import Optional, Union

import pandas as pd
import pandas_flavor as pf
from pandas.api.types import is_scalar

from janitor.functions.utils import (
    get_index_labels,
    make_clean_names,
)
from janitor.utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(preserve_original_columns="preserve_original_labels")
def clean_names(
    df: pd.DataFrame,
    axis: Union[str, None] = "columns",
    column_names: Union[str, list] = None,
    strip_underscores: Optional[Union[str, bool]] = None,
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
            column names. Default None keeps outer underscores. Values can be
            either 'left', 'right' or 'both' or the respective shorthand 'l',
            'r' and True.
        case_type: Whether to make columns lower or uppercase.
            Current case may be preserved with 'preserve',
            while snake case conversion (from CamelCase or camelCase only)
            can be turned on using "snake".
            Default 'lower' makes all characters lowercase.
        remove_special: Remove special characters from columns.
            Only letters, numbers and underscores are preserved.
        strip_accents: Whether or not to remove accents from
            columns names.
        preserve_original_labels: Preserve original names.
            This is later retrievable using `df.original_labels`.
            Applies if `axis` is not None.
        enforce_string: Whether or not to convert all column names
            to string type. Defaults to True, but can be turned off.
            Columns with >1 levels will not be converted by default.
        truncate_limit: Truncates formatted column names to
            the specified length. Default None does not truncate.

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
            df[column_name] = make_clean_names(
                col=df[column_name],
                enforce_string=enforce_string,
                case_type=case_type,
                remove_special=remove_special,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                truncate_limit=truncate_limit,
                df_type="pandas",
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
            make_clean_names(
                col=obj,
                enforce_string=enforce_string,
                case_type=case_type,
                remove_special=remove_special,
                strip_accents=strip_accents,
                strip_underscores=strip_underscores,
                truncate_limit=truncate_limit,
                df_type="pandas",
            )
            for obj in target_axis
        ]
    else:
        target_axis = make_clean_names(
            col=target_axis,
            enforce_string=enforce_string,
            case_type=case_type,
            remove_special=remove_special,
            strip_accents=strip_accents,
            strip_underscores=strip_underscores,
            truncate_limit=truncate_limit,
            df_type="pandas",
        )
    # Store the original column names, if enabled by user
    if preserve_original_labels:
        df.__dict__["original_labels"] = getattr(df, axis)
    setattr(df, axis, target_axis)
    return df
