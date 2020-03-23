""" General purpose data cleaning functions. """

import datetime as dt
import re
import unicodedata
import warnings
from fnmatch import translate
from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pandas_flavor as pf
from natsort import index_natsorted
from pandas.api.types import union_categoricals
from pandas.errors import OutOfBoundsDatetime
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

from .errors import JanitorError
from .utils import (
    _clean_accounting_column,
    _currency_column_to_numeric,
    _replace_empty_string_with_none,
    _replace_original_empty_string_with_none,
    _strip_underscores,
    check,
    check_column,
    deprecated_alias,
)


def unionize_dataframe_categories(
    *dataframes, column_names: Iterable[pd.CategoricalDtype] = None
) -> List[pd.DataFrame]:
    """
    Given a group of dataframes which contain some categorical columns, for
    each categorical column present, find all the possible categories across
    all the dataframes which have that column.
    Update each dataframes' corresponding column with a new categorical object
    that contains the original data
    but has labels for all the possible categories from all dataframes.
    This is useful when concatenating a list of dataframes which all have the
    same categorical columns into one dataframe.

    If, for a given categorical column, all input dataframes do not have at
    least one instance of all the possible categories,
    Pandas will change the output dtype of that column from ``category`` to
    ``object``, losing out on dramatic speed gains you get from the former
    format.

    Usage example for concatentation of categorical column-containing
    dataframes:

    Instead of:

    .. code-block:: python

        concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)

    which in your case has resulted in ``category`` -> ``object`` conversion,
    use:

    .. code-block:: python

        unionized_dataframes = unionize_dataframe_categories(df1, df2, df2)
        concatenated_df = pd.concat(unionized_dataframes, ignore_index=True)

    :param dataframes: The dataframes you wish to unionize the categorical
        objects for.
    :param column_names: If supplied, only unionize this subset of columns.
    :returns: A list of the category-unioned dataframes in the same order they
        were provided.
    """

    if any(not isinstance(df, pd.DataFrame) for df in dataframes):
        raise TypeError("Inputs must all be dataframes.")

    elif column_names is None:
        # Find all columns across all dataframes that are categorical

        column_names = set()

        for df in dataframes:
            column_names = column_names.union(
                [
                    column_name
                    for column_name in df.columns
                    if isinstance(df[column_name].dtype, pd.CategoricalDtype)
                ]
            )

    else:
        column_names = [column_names]
    # For each categorical column, find all possible values across the DFs

    category_unions = {
        column_name: union_categoricals(
            [df[column_name] for df in dataframes if column_name in df.columns]
        )
        for column_name in column_names
    }

    # Make a shallow copy of all DFs and modify the categorical columns
    # such that they can encode the union of all possible categories for each.

    refactored_dfs = []

    for df in dataframes:
        df = df.copy(deep=False)

        for column_name, categorical in category_unions.items():
            if column_name in df.columns:
                df[column_name] = pd.Categorical(
                    df[column_name], categories=categorical.categories
                )

        refactored_dfs.append(df)

    return refactored_dfs


@pf.register_dataframe_method
def move(
    df: pd.DataFrame,
    source: Union[int, str],
    target: Union[int, str],
    position: str = "before",
    axis: int = 0,
) -> pd.DataFrame:
    """
    Move column or row to a position adjacent to another column or row in
    dataframe. Must have unique column names or indices.

    This operation does not reset the index of the dataframe. User must
    explicitly do so.

    Does not apply to multilevel dataframes.

    Functional usage syntax:

    .. code-block:: python

        df = move(df, source=3, target=15, position='after', axis=0)

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).move(source=3, target=15, position='after',
        axis=0)

    :param df: The pandas Dataframe object.
    :param int or str source: column or row to move
    :param str target: column or row to move adjacent to
    :param str position: Specifies whether the Series is moved to before or
        after the adjacent Series. Values can be either 'before' or 'after';
        defaults to 'before'.
    :param int axis: Axis along which the function is applied. 0 to move a
        row, 1 to move a column.
    :returns: The dataframe with the Series moved.

    """
    if axis not in [0, 1]:
        raise ValueError(f"Invalid axis '{axis}'. Can only be 0 or 1.")

    if position not in ["before", "after"]:
        raise ValueError(
            f"Invalid position '{position}'. Can only be 'before' or 'after'."
        )

    if axis == 0:
        names = list(df.index)

        if source not in names:
            raise ValueError(f"Source row '{source}' not in dataframe.")

        if target not in names:
            raise ValueError(f"Target row '{target}' not in dataframe.")

        names.remove(source)
        pos = names.index(target)

        if position == "after":
            pos += 1
        names.insert(pos, source)

        df = df.loc[names, :]
    else:
        names = list(df.columns)

        if source not in names:
            raise ValueError(f"Source column '{source}' not in dataframe.")

        if target not in names:
            raise ValueError(f"Target column '{target}' not in dataframe.")

        names.remove(source)
        pos = names.index(target)

        if position == "after":
            pos += 1
        names.insert(pos, source)

        df = df.loc[:, names]

    return df


@pf.register_dataframe_method
def clean_names(
    df: pd.DataFrame,
    strip_underscores: str = None,
    case_type: str = "lower",
    remove_special: bool = False,
    strip_accents: bool = True,
    preserve_original_columns: bool = True,
    enforce_string: bool = True,
) -> pd.DataFrame:
    """
    Clean column names.

    Takes all column names, converts them to lowercase,
    then replaces all spaces with underscores.

    By default, column names are converted to string types.
    This can be switched off by passing in ``enforce_string=False``.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = clean_names(df)

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).clean_names()

    :Example of transformation:

    .. code-block:: python

        Columns before: First Name, Last Name, Employee Status, Subject
        Columns after: first_name, last_name, employee_status, subject

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
    :param preserve_original_columns: (optional) Preserve original names.
        This is later retrievable using `df.original_columns`.
    :param enforce_string: Whether or not to convert all column names
        to string type. Defaults to True, but can be turned off.
        Columns with >1 levels will not be converted by default.
    :returns: A pandas DataFrame.
    """
    original_column_names = list(df.columns)

    if enforce_string:
        df = df.rename(columns=lambda x: str(x))

    df = df.rename(columns=lambda x: _change_case(x, case_type))

    df = df.rename(columns=_normalize_1)

    if remove_special:
        df = df.rename(columns=_remove_special)

    if strip_accents:
        df = df.rename(columns=_strip_accents)

    df = df.rename(columns=lambda x: re.sub("_+", "_", x))  # noqa: PD005
    df = _strip_underscores(df, strip_underscores)

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
    """
    Convert camelcase names to snake case.

    Implementation taken from: https://gist.github.com/jaytaylor/3660565
    by @jtaylor
    """

    subbed = _underscorer1.sub(r"\1_\2", col_name)  # noqa: PD005
    return _underscorer2.sub(r"\1_\2", subbed).lower()  # noqa: PD005


FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", "")]


def _normalize_1(col_name: Hashable) -> str:
    result = str(col_name)
    for search, replace in FIXES:
        result = re.sub(search, replace, result)  # noqa: PD005
    return result


def _strip_accents(col_name: str) -> str:
    """
    Removes accents from a DataFrame column name.
    .. _StackOverflow: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
    """  # noqa: E501
    return "".join(
        l
        for l in unicodedata.normalize("NFD", col_name)
        if not unicodedata.combining(l)
    )


@pf.register_dataframe_method
def remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows and columns that are completely null.

    This method also resets the index(by default) since it doesn't make sense
    to preserve the index of a completely empty row.

    This method mutates the original DataFrame.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/38884538/python-pandas-find-all-rows-where-all-values-are-nan

    Functional usage syntax:

    .. code-block:: python

        df = remove_empty(df)

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).remove_empty()

    :param df: The pandas DataFrame object.

    :returns: A pandas DataFrame.
    """  # noqa: E501
    nanrows = df.index[df.isna().all(axis=1)]
    df = df.drop(index=nanrows).reset_index(drop=True)

    nancols = df.columns[df.isna().all(axis=0)]
    df = df.drop(columns=nancols)

    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def get_dupes(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable] = None
) -> pd.DataFrame:
    """
    Return all duplicate rows.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = pd.DataFrame(...)
        df = get_dupes(df)

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).get_dupes()

    :param df: The pandas DataFrame object.
    :param str/iterable column_names: (optional) A column name or an iterable
        (list or tuple) of column names. Following pandas API, this only
        considers certain columns for identifying duplicates. Defaults to using
        all columns.
    :returns: The duplicate rows, as a pandas DataFrame.
    """
    dupes = df.duplicated(subset=column_names, keep=False)
    return df[dupes == True]  # noqa: E712


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def encode_categorical(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable]
) -> pd.DataFrame:
    """
    Encode the specified columns with Pandas'
    `category dtype <http://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_.

    This method mutates the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        categorical_cols = ['col1', 'col2', 'col4']
        df = encode_categorical(df, columns=categorical_cols)  # one way

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        categorical_cols = ['col1', 'col2', 'col4']
        df = pd.DataFrame(...).encode_categorical(columns=categorical_cols)

    :param df: The pandas DataFrame object.
    :param Hashable/iterable column_names: A column name or an iterable (list or
        tuple) of column names.
    :returns: A pandas DataFrame
    """  # noqa: E501
    if isinstance(column_names, list) or isinstance(column_names, tuple):
        for col in column_names:
            if col not in df.columns:
                raise JanitorError(f"{col} missing from dataframe columns!")
            df[col] = pd.Categorical(df[col])
    elif isinstance(column_names, Hashable):
        if column_names not in df.columns:
            raise JanitorError(
                f"{column_names} missing from dataframe columns!"
            )
        df[column_names] = pd.Categorical(df[column_names])
    else:
        raise JanitorError(
            "kwarg `column_names` must be hashable or iterable!"
        )
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def label_encode(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable]
) -> pd.DataFrame:
    """
    Convert labels into numerical data.

    This method will create a new column with the string "_enc" appended
    after the original column's name. Consider this to be syntactic sugar.

    This method behaves differently from `encode_categorical`. This method
    creates a new column of numeric data. `encode_categorical` replaces the
    dtype of the original column with a "categorical" dtype.

    This method mutates the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = label_encode(df, column_names="my_categorical_column")  # one way

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        categorical_cols = ['col1', 'col2', 'col4']
        df = pd.DataFrame(...).label_encode(column_names=categorical_cols)

    :param df: The pandas DataFrame object.
    :param Hashable/iterable column_names: A column name or an iterable (list
        or tuple) of column names.
    :returns: A pandas DataFrame.
    """
    le = LabelEncoder()
    if isinstance(column_names, list) or isinstance(column_names, tuple):
        for col in column_names:
            if col not in df.columns:
                raise JanitorError(f"{col} missing from column_names")
            df[f"{col}_enc"] = le.fit_transform(df[col])
    elif isinstance(column_names, Hashable):
        if column_names not in df.columns:
            raise JanitorError(f"{column_names} missing from column_names")
        df[f"{column_names}_enc"] = le.fit_transform(df[column_names])
    else:
        raise JanitorError(
            "kwarg `column_names` must be hashable or iterable!"
        )
    return df


@pf.register_dataframe_method
@deprecated_alias(old="old_column_name", new="new_column_name")
def rename_column(
    df: pd.DataFrame, old_column_name: str, new_column_name: str
) -> pd.DataFrame:
    """
    Rename a column in place.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = rename_column(df, "old_column_name", "new_column_name")

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).rename_column("old_column_name", "new_column_name")

    This is just syntactic sugar/a convenience function for renaming one column
    at a time. If you are convinced that there are multiple columns in need of
    changing, then use the :py:meth:`pandas.DataFrame.rename` method.

    :param df: The pandas DataFrame object.
    :param old_column_name: The old column name.
    :param new_column_name: The new column name.
    :returns: A pandas DataFrame with renamed columns.
    """  # noqa: E501
    check_column(df, [old_column_name])

    return df.rename(columns={old_column_name: new_column_name})


@pf.register_dataframe_method
def rename_columns(df: pd.DataFrame, new_column_names: Dict) -> pd.DataFrame:
    """
    Rename columns in place.

    Functional usage syntax:

    .. code-block:: python

        df = rename_columns(df, {"old_column_name": "new_column_name"})

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).rename_columns({"old_column_name": "new_column_name"})

    This is just syntactic sugar/a convenience function for renaming one column
    at a time. If you are convinced that there are multiple columns in need of
    changing, then use the :py:meth:`pandas.DataFrame.rename` method.

    :param df: The pandas DataFrame object.
    :param new_column_names: A dictionary of old and new column names.
    :returns: A pandas DataFrame with renamed columns.
    """  # noqa: E501
    check_column(df, list(new_column_names.keys()))

    return df.rename(columns=new_column_names)


@pf.register_dataframe_method
def reorder_columns(
    df: pd.DataFrame, column_order: Union[Iterable[str], pd.Index, Hashable]
) -> pd.DataFrame:
    """
    Reorder DataFrame columns by specifying desired order as list of col names.

    Columns not specified retain their order and follow after specified cols.

    Validates column_order to ensure columns are all present in DataFrame.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    Given `DataFrame` with column names `col1`, `col2`, `col3`:

    .. code-block:: python

        df = reorder_columns(df, ['col2', 'col3'])

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).reorder_columns(['col2', 'col3'])

    The column order of `df` is now `col2`, `col3`, `col1`.

    Internally, this function uses `DataFrame.reindex` with `copy=False`
    to avoid unnecessary data duplication.

    :param df: `DataFrame` to reorder
    :param column_order: A list of column names or Pandas `Index`
        specifying their order in the returned `DataFrame`.
    :returns: A pandas DataFrame with reordered columns.
    """
    check("column_order", column_order, [list, tuple, pd.Index])

    if any(col not in df.columns for col in column_order):
        raise IndexError(
            "A column in column_order was not found in the DataFrame."
        )

    # if column_order is a Pandas index, needs conversion to list:
    column_order = list(column_order)

    return df.reindex(
        columns=(
            column_order
            + [col for col in df.columns if col not in column_order]
        ),
        copy=False,
    )


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def coalesce(
    df: pd.DataFrame,
    column_names: Iterable[Hashable],
    new_column_name: str = None,
    delete_columns: bool = True,
) -> pd.DataFrame:
    """

    Coalesces two or more columns of data in order of column names provided.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = coalesce(df, columns=['col1', 'col2'], 'col3')

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).coalesce(['col1', 'col2'])

    The first example will create a new column called 'col3' with values from
    'col2' inserted where values from 'col1' are NaN, then delete the original
    columns. The second example will keep the name 'col1' in the new column.

    This is more syntactic diabetes! For R users, this should look familiar to
    `dplyr`'s `coalesce` function; for Python users, the interface
    should be more intuitive than the :py:meth:`pandas.Series.combine_first`
    method (which we're just using internally anyways).

    :param df: A pandas DataFrame.
    :param column_names: A list of column names.
    :param str new_column_name: The new column name after combining.
    :param bool delete_columns: Whether to delete the columns being coalesced
    :returns: A pandas DataFrame with coalesced columns.
    """
    series = [df[c] for c in column_names]

    def _coalesce(series1, series2):
        return series1.combine_first(series2)

    if delete_columns:
        df = df.drop(columns=column_names)
    if not new_column_name:
        new_column_name = column_names[0]
    df[new_column_name] = reduce(_coalesce, series)  # noqa: F821
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_excel_date(
    df: pd.DataFrame, column_name: Hashable
) -> pd.DataFrame:
    """
    Convert Excel's serial date format into Python datetime format.

    This method mutates the original DataFrame.

    Implementation is also from `Stack Overflow`.

    .. _Stack Overflow: https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas

    Functional usage syntax:

    .. code-block:: python

        df = convert_excel_date(df, column_name='date')

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).convert_excel_date('date')

    :param df: A pandas DataFrame.
    :param Hashable column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """  # noqa: E501
    df[column_name] = pd.TimedeltaIndex(
        df[column_name], unit="d"
    ) + dt.datetime(
        1899, 12, 30
    )  # noqa: W503
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_matlab_date(
    df: pd.DataFrame, column_name: Hashable
) -> pd.DataFrame:
    """
    Convert Matlab's serial date number into Python datetime format.

    Implementation is also from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python

    This method mutates the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = convert_matlab_date(df, column_name='date')

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).convert_matlab_date('date')

    :param df: A pandas DataFrame.
    :param Hashable column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """  # noqa: E501
    days = pd.Series([dt.timedelta(v % 1) for v in df[column_name]])
    df[column_name] = (
        df[column_name].astype(int).apply(dt.datetime.fromordinal)
        + days
        - dt.timedelta(days=366)
    )
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_unix_date(df: pd.DataFrame, column_name: Hashable) -> pd.DataFrame:
    """
    Convert unix epoch time into Python datetime format.

    Note that this ignores local tz and convert all timestamps to naive
    datetime based on UTC!

    This method mutates the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = convert_unix_date(df, column_name='date')

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).convert_unix_date('date')

    :param df: A pandas DataFrame.
    :param Hashable column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """

    try:
        df[column_name] = pd.to_datetime(df[column_name], unit="s")
    except OutOfBoundsDatetime:  # Indicates time is in milliseconds.
        df[column_name] = pd.to_datetime(df[column_name], unit="ms")
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def fill_empty(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable], value
) -> pd.DataFrame:
    """
    Fill `NaN` values in specified columns with a given value.

    Super sugary syntax that wraps :py:meth:`pandas.DataFrame.fillna`.

    This method mutates the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = fill_empty(df, column_names=['col1', 'col2'], value=0)

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).fill_empty(column_names='col1', value=0)

    :param df: A pandas DataFrame.
    :param column_names: column_names: A column name or an iterable (list
        or tuple) of column names If a single column name is passed in, then
        only that column will be filled; if a list or tuple is passed in, then
        those columns will all be filled with the same value.
    :param value: The value that replaces the `NaN` values.
    :returns: A pandas DataFrame with `Nan` values filled.
    """
    if isinstance(column_names, list) or isinstance(column_names, tuple):
        for col in column_names:
            if col not in df.columns:
                raise JanitorError(f"{col} missing from dataframe columns!")
            df[col] = df[col].fillna(value)
    else:
        if column_names not in df.columns:
            raise JanitorError(
                f"{column_names} missing from dataframe columns!"
            )
        df[column_names] = df[column_names].fillna(value)

    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def expand_column(
    df: pd.DataFrame, column_name: Hashable, sep: str, concat: bool = True
) -> pd.DataFrame:
    """
    Expand a categorical column with multiple labels into dummy-coded columns.

    Super sugary syntax that wraps :py:meth:`pandas.Series.str.get_dummies`.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = expand_column(df,
                           column_name='col_name',
                           sep=', ')  # note space in sep

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).expand_column(column_name='col_name',
                                             sep=', ')

    :param df: A pandas DataFrame.
    :param column_name: Which column to expand.
    :param sep: The delimiter. Example delimiters include `|`, `, `, `,` etc.
    :param bool concat: Whether to return the expanded column concatenated to
        the original dataframe (`concat=True`), or to return it standalone
        (`concat=False`).
    :returns: A pandas DataFrame with an expanded column.
    """
    expanded_df = df[column_name].str.get_dummies(sep=sep)
    if concat:
        df = df.join(expanded_df)
        return df
    else:
        return expanded_df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def concatenate_columns(
    df: pd.DataFrame,
    column_names: List[Hashable],
    new_column_name,
    sep: str = "-",
) -> pd.DataFrame:
    """
    Concatenates the set of columns into a single column.

    Used to quickly generate an index based on a group of columns.

    This method mutates the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = concatenate_columns(df,
                                 column_names=['col1', 'col2'],
                                 new_column_name='id',
                                 sep='-')

    Method chaining syntax:

    .. code-block:: python

        df = (pd.DataFrame(...).
              concatenate_columns(column_names=['col1', 'col2'],
                                  new_column_name='id',
                                  sep='-'))

    :param df: A pandas DataFrame.
    :param column_names: A list of columns to concatenate together.
    :param new_column_name: The name of the new column.
    :param sep: The separator between each column's data.
    :returns: A pandas DataFrame with concatenated columns.
    """
    if len(column_names) < 2:
        raise JanitorError("At least two columns must be specified")
    for i, col in enumerate(column_names):
        if i == 0:
            df[new_column_name] = df[col].astype(str)
        else:
            df[new_column_name] = (
                df[new_column_name] + sep + df[col].astype(str)
            )

    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def deconcatenate_column(
    df: pd.DataFrame,
    column_name: Hashable,
    sep: str = None,
    new_column_names: Union[List[str], Tuple[str]] = None,
    autoname: str = None,
    preserve_position: bool = False,
) -> pd.DataFrame:
    """
    De-concatenates a single column into multiple columns.
    The column to de-concatenate can be either a collection (list, tuple, ...)
    which can be separated out with ``pd.Series.tolist()``,
    or a string to slice based on ``sep``.
    To determine this behaviour automatically,
    the first element in the column specified is inspected.
    If it is a string, then ``sep`` must be specified.
    Else, the function assumes that it is an iterable type
    (e.g. ``list`` or ``tuple``),
    and will attempt to deconcatenate by splitting the list.

    Given a column with string values, this is the inverse of the
    ``concatenate_columns`` function.

    Used to quickly split columns out of a single column.

    The keyword argument ``preserve_position``
    takes ``True`` or ``False`` boolean
    that controls whether the ``new_column_names``
    will take the original position
    of the to-be-deconcatenated ``column_name``:

    - When `preserve_position=False` (default), `df.columns` change from
      `[..., column_name, ...]` to `[..., column_name, ..., new_column_names]`.
      In other words, the deconcatenated new columns are appended to the right
      of the original dataframe and the original `column_name` is NOT dropped.
    - When `preserve_position=True`, `df.column` change from
      `[..., column_name, ...]` to `[..., new_column_names, ...]`.
      In other words, the deconcatenated new column will REPLACE the original
      `column_name` at its original position, and `column_name` itself
      is dropped.

    The keyword argument ``autoname`` accepts a base string
    and then automatically creates numbered column names
    based off the base string.
    For example, if ``col`` is passed in
    as the argument to ``autoname``,
    and 4 columns are created,
    then the resulting columns will be named
    ``col1, col2, col3, col4``.
    Numbering is always 1-indexed, not 0-indexed,
    in order to make the column names human-friendly.

    This method does not mutate the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = deconcatenate_column(
                df, column_name='id', new_column_names=['col1', 'col2'],
                sep='-', preserve_position=True
        )

    Method chaining syntax:

    .. code-block:: python

        df = (pd.DataFrame(...).
                deconcatenate_column(
                    column_name='id', new_column_names=['col1', 'col2'],
                    sep='-', preserve_position=True
                ))

    :param df: A pandas DataFrame.
    :param column_name: The column to split.
    :param sep: The separator delimiting the column's data.
    :param new_column_names: A list of new column names post-splitting.
    :param autoname: A base name for automatically naming the new columns.
        Takes precedence over ``new_column_names`` if both are provided.
    :param preserve_position: Boolean for whether or not to preserve original
        position of the column upon de-concatenation, default to False
    :returns: A pandas DataFrame with a deconcatenated column.
    """

    if column_name not in df.columns:
        raise ValueError(f"column name {column_name} not present in dataframe")

    if isinstance(df[column_name].iloc[0], str):
        if sep is None:
            raise ValueError(
                "`sep` must be specified if the column values are " "strings."
            )
        df_deconcat = df[column_name].str.split(sep, expand=True)
    else:
        df_deconcat = pd.DataFrame(
            df[column_name].to_list(), columns=new_column_names, index=df.index
        )

    if preserve_position:
        # Keep a copy of the original dataframe
        df_original = df.copy()

    if new_column_names is None and autoname is None:
        raise ValueError(
            "One of `new_column_names` or `autoname` must be supplied."
        )

    if autoname:
        new_column_names = [
            f"{autoname}{i}" for i in range(1, df_deconcat.shape[1] + 1)
        ]

    if not len(new_column_names) == df_deconcat.shape[1]:
        raise JanitorError(
            f"you need to provide {len(df_deconcat.shape[1])} names "
            "to new_column_names"
        )

    df_deconcat.columns = new_column_names
    df = pd.concat([df, df_deconcat], axis=1)

    if preserve_position:
        cols = list(df_original.columns)
        index_original = cols.index(column_name)

        for i, col_new in enumerate(new_column_names):
            cols.insert(index_original + i, col_new)

        df = df[cols].drop(columns=column_name)

    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_string(
    df: pd.DataFrame,
    column_name: Hashable,
    search_string: str,
    complement: bool = False,
) -> pd.DataFrame:
    """
    Filter a string-based column according to whether it contains a substring.

    This is super sugary syntax that builds on top of
    `pandas.Series.str.contains`.

    Because this uses internally `pandas.Series.str.contains`, which allows a
    regex string to be passed into it, thus `search_string` can also be a regex
    pattern.

    This method does not mutate the original DataFrame.

    This function allows us to method chain filtering operations:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_string('column', search_string='pattern', complement=False)
              ...)  # chain on more data preprocessing.

    This stands in contrast to the in-place syntax that is usually used:

    .. code-block:: python

        df = pd.DataFrame(...)
        df = df[df['column'].str.contains('pattern')]]

    As can be seen here, the API design allows for a more seamless flow in
    expressing the filtering operations.

    Functional usage syntax:

    .. code-block:: python

        df = filter_string(df,
                           column_name='column',
                           search_string='pattern',
                           complement=False)

    Method chaining syntax:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_string(column_name='column',
                             search_string='pattern',
                             complement=False)
              ...)

    :param df: A pandas DataFrame.
    :param column_name: The column to filter. The column should contain strings.
    :param search_string: A regex pattern or a (sub-)string to search.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """  # noqa: E501
    criteria = df[column_name].str.contains(search_string)
    if complement:
        return df[~criteria]
    else:
        return df[criteria]


@pf.register_dataframe_method
def filter_on(
    df: pd.DataFrame, criteria: str, complement: bool = False
) -> pd.DataFrame:
    """
    Return a dataframe filtered on a particular criteria.

    This method does not mutate the original DataFrame.

    This is super-sugary syntax that wraps the pandas `.query()` API, enabling
    users to use strings to quickly specify filters for filtering their
    dataframe. The intent is that `filter_on` as a verb better matches the
    intent of a pandas user than the verb `query`.

    Let's say we wanted to filter students based on whether they failed an exam
    or not, which is defined as their score (in the "score" column) being less
    than 50.

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_on('score < 50', complement=False)
              ...)  # chain on more data preprocessing.

    This stands in contrast to the in-place syntax that is usually used:

    .. code-block:: python

        df = pd.DataFrame(...)
        df = df[df['score'] < 3]

    As with the `filter_string` function, a more seamless flow can be expressed
    in the code.

    Functional usage syntax:

    .. code-block:: python

        df = filter_on(df,
                       'score < 50',
                       complement=False)

    Method chaining syntax:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_on('score < 50', complement=False))

    Credit to Brant Peterson for the name.

    :param df: A pandas DataFrame.
    :param criteria: A filtering criteria that returns an array or Series of
        booleans, on which pandas can filter on.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """
    if complement:
        return df.query("not " + criteria)
    else:
        return df.query(criteria)


@pf.register_dataframe_method
@deprecated_alias(column="column_name", start="start_date", end="end_date")
def filter_date(
    df: pd.DataFrame,
    column_name: Hashable,
    start_date: dt.date = None,
    end_date: dt.date = None,
    years: List = None,
    months: List = None,
    days: List = None,
    column_date_options: Dict = None,
    format: str = None,
) -> pd.DataFrame:
    """
    Filter a date-based column based on certain criteria.

    This method does not mutate the original DataFrame.

    Dates may be finicky and this function builds on top of the "magic" from
    the pandas `to_datetime` function that is able to parse dates well.

    Additional options to parse the date type of your column may be found at
    the official pandas documentation:

    pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

    **Note:** This method will cast your column to a Timestamp!

    :param df: A pandas dataframe.
    :param column_name: The column which to apply the fraction transformation.
    :param start_date: The beginning date to use to filter the DataFrame.
    :param end_date: The end date to use to filter the DataFrame.
    :param years: The years to use to filter the DataFrame.
    :param months: The months to use to filter the DataFrame.
    :param days: The days to use to filter the DataFrame.
    :param column_date_options: 'Special options to use when parsing the date
        column in the original DataFrame. The options may be found at the
        official Pandas documentation.'
    :param format: 'If you're using a format for `start_date` or `end_date`
        that is not recognized natively by pandas' to_datetime function, you
        may supply the format yourself. Python date and time formats may be
        found at http://strftime.org/.'
    :returns: A filtered pandas DataFrame.

    **Note:** This only affects the format of the `start_date` and `end_date`
    parameters. If there's an issue with the format of the DataFrame being
    parsed, you would pass `{'format': your_format}` to `column_date_options`.

    """
    # TODO: need to convert this to notebook.
    #     :Setup:
    # .. code-block:: python

    #     import pandas as pd
    #     import janitor

    #     date_list = [
    #         [1, "01/28/19"], [2, "01/29/19"], [3, "01/30/19"],
    #         [4, "01/31/19"], [5, "02/01/19"], [6, "02/02/19"],
    #         [7, "02/03/19"], [8, "02/04/19"], [9, "02/05/19"],
    #         [10, "02/06/19"], [11, "02/07/20"], [12, "02/08/20"],
    #         [13, "02/09/20"], [14, "02/10/20"], [15, "02/11/20"],
    #         [16, "02/12/20"], [17, "02/07/20"], [18, "02/08/20"],
    #         [19, "02/09/20"], [20, "02/10/20"], [21, "02/11/20"],
    #         [22, "02/12/20"], [23, "03/08/20"], [24, "03/09/20"],
    #         [25, "03/10/20"], [26, "03/11/20"], [27, "03/12/20"]]

    #     example_dataframe = pd.DataFrame(date_list,
    #                                      columns = ['AMOUNT', 'DATE'])

    # :Example 1: Filter dataframe between two dates

    # .. code-block:: python

    #     start_date = "01/29/19"
    #     end_date = "01/30/19"

    #     example_dataframe.filter_date(
    #         'DATE', start_date=start_date, end_date=end_date
    #     )

    # :Output:

    # .. code-block:: python

    #        AMOUNT       DATE
    #     1       2 2019-01-29
    #     2       3 2019-01-30

    # :Example 2: Using a different date format for filtering

    # .. code-block:: python

    #     end_date = "01$$$30$$$19"
    #     format = "%m$$$%d$$$%y"

    #     example_dataframe.filter_date(
    #         'DATE', end_date=end_date, format=format
    #     )

    # :Output:

    # .. code-block:: python

    #        AMOUNT       DATE
    #     0       1 2019-01-28
    #     1       2 2019-01-29
    #     2       3 2019-01-30

    # :Example 3: Filtering by year

    # .. code-block:: python

    #     years = [2019]

    #     example_dataframe.filter_date('DATE', years=years)

    # :Output:

    # .. code-block:: python

    #        AMOUNT       DATE
    #     0       1 2019-01-28
    #     1       2 2019-01-29
    #     2       3 2019-01-30
    #     3       4 2019-01-31
    #     4       5 2019-02-01
    #     5       6 2019-02-02
    #     6       7 2019-02-03
    #     7       8 2019-02-04
    #     8       9 2019-02-05
    #     9      10 2019-02-06

    # :Example 4: Filtering by year and month

    # .. code-block:: python

    #     years = [2020]
    #     months = [3]

    #     example_dataframe.filter_date('DATE', years=years, months=months)

    # :Output:

    # .. code-block:: python

    #         AMOUNT       DATE
    #     22      23 2020-03-08
    #     23      24 2020-03-09
    #     24      25 2020-03-10
    #     25      26 2020-03-11
    #     26      27 2020-03-12

    # :Example 5: Filtering by year and day

    # .. code-block:: python

    #     years = [2020]
    #     days = range(10,12)

    #     example_dataframe.filter_date('DATE', years=years, days=days)

    # :Output:

    # .. code-block:: python

    #         AMOUNT       DATE
    #     13      14 2020-02-10
    #     14      15 2020-02-11
    #     19      20 2020-02-10
    #     20      21 2020-02-11
    #     24      25 2020-03-10
    #     25      26 2020-03-11

    def _date_filter_conditions(conditions):
        """Taken from: https://stackoverflow.com/a/13616382."""
        return reduce(np.logical_and, conditions)

    if column_date_options:
        df.loc[:, column_name] = pd.to_datetime(
            df.loc[:, column_name], **column_date_options
        )
    else:
        df.loc[:, column_name] = pd.to_datetime(df.loc[:, column_name])

    _filter_list = []

    if start_date:
        start_date = pd.to_datetime(start_date, format=format)
        _filter_list.append(df.loc[:, column_name] >= start_date)

    if end_date:
        end_date = pd.to_datetime(end_date, format=format)
        _filter_list.append(df.loc[:, column_name] <= end_date)

    if years:
        _filter_list.append(df.loc[:, column_name].dt.year.isin(years))

    if months:
        _filter_list.append(df.loc[:, column_name].dt.month.isin(months))

    if days:
        _filter_list.append(df.loc[:, column_name].dt.day.isin(days))

    if start_date and end_date:
        if start_date > end_date:
            warnings.warn(
                f"Your start date of {start_date} is after your end date of "
                f"{end_date}. Is this intended?"
            )

    return df.loc[_date_filter_conditions(_filter_list), :]


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_column_isin(
    df: pd.DataFrame,
    column_name: Hashable,
    iterable: Iterable,
    complement: bool = False,
) -> pd.DataFrame:
    """
    Filter a dataframe for values in a column that exist in another iterable.

    This method does not mutate the original DataFrame.

    Assumes exact matching; fuzzy matching not implemented.

    The below example syntax will filter the DataFrame such that we only get
    rows for which the "names" are exactly "James" and "John".

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .clean_names()
            .filter_column_isin(column_name="names", iterable=["James", "John"]
            )
        )

    This is the method chaining alternative to:

    .. code-block:: python

        df = df[df['names'].isin(['James', 'John'])]

    If "complement" is true, then we will only get rows for which the names
    are not James or John.

    :param df: A pandas DataFrame
    :param column_name: The column on which to filter.
    :param iterable: An iterable. Could be a list, tuple, another pandas
        Series.
    :param complement: Whether to return the complement of the selection or
        not.
    :returns: A filtered pandas DataFrame.
    """
    if len(iterable) == 0:
        raise ValueError(
            "`iterable` kwarg must be given an iterable of length 1 or greater"
        )
    criteria = df[column_name].isin(iterable)

    if complement:
        return df[~criteria]
    else:
        return df[criteria]


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def remove_columns(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable]
) -> pd.DataFrame:
    """
    Remove the set of columns specified in `column_names`.

    This method does not mutate the original DataFrame.

    Intended to be the method-chaining alternative to `del df[col]`.

    Method chaining syntax:

    .. code-block:: python

        df = pd.DataFrame(...).remove_columns(column_names=['col1', 'col2'])

    :param df: A pandas DataFrame
    :param column_names: The columns to remove.
    :returns: A pandas DataFrame.
    """
    return df.drop(columns=column_names)


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def change_type(
    df: pd.DataFrame,
    column_name: Hashable,
    dtype: type,
    ignore_exception: bool = False,
) -> pd.DataFrame:
    """
    Change the type of a column.

    This method mutates the original DataFrame.

    Exceptions that are raised can be ignored. For example, if one has a mixed
    dtype column that has non-integer strings and integers, and you want to
    coerce everything to integers, you can optionally ignore the non-integer
    strings and replace them with ``NaN`` or keep the original value

    Intended to be the method-chaining alternative to:

        df[col] = df[col].astype(dtype)

    Method chaining syntax:

    .. code-block:: python

        df = pd.DataFrame(...).change_type('col1', str)

    :param df: A pandas dataframe.
    :param column_name: A column in the dataframe.
    :param dtype: The datatype to convert to. Should be one of the standard
        Python types, or a numpy datatype.
    :param ignore_exception: one of ``{False, "fillna", "keep_values"}``.
    :returns: A pandas DataFrame with changed column types.
    """
    if not ignore_exception:
        df[column_name] = df[column_name].astype(dtype)
    elif ignore_exception == "keep_values":
        df[column_name] = df[column_name].astype(dtype, errors="ignore")
    elif ignore_exception == "fillna":
        # returns None when conversion
        def convert(x, dtype):
            try:
                return dtype(x)
            except ValueError:
                return None

        df[column_name] = df[column_name].apply(lambda x: convert(x, dtype))
    else:
        raise ValueError("unknown option for ignore_exception")
    return df


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name")
def add_column(
    df: pd.DataFrame,
    column_name: str,
    value: Union[List[Any], Tuple[Any], Any],
    fill_remaining: bool = False,
) -> pd.DataFrame:
    """
    Add a column to the dataframe.

    This method does not mutate the original DataFrame.

    Intended to be the method-chaining alternative to::

        df[column_name] = value

    Method chaining syntax adding a column with only a single value:

    .. code-block:: python

        # This will add a column with only one value.
        df = pd.DataFrame(...).add_column(column_name="new_column", 2)

    Method chaining syntax adding a column with more than one value:

    .. code-block:: python

        # This will add a column with an iterable of values.
        vals = [1, 2, 5, ..., 3, 4]  # of same length as the dataframe.
        df = pd.DataFrame(...).add_column(column_name="new_column", vals)

    :param df: A pandas dataframe.
    :param column_name: Name of the new column. Should be a string, in order
        for the column name to be compatible with the Feather binary
        format (this is a useful thing to have).
    :param value: Either a single value, or a list/tuple of values.
    :param fill_remaining: If value is a tuple or list that is smaller than
        the number of rows in the DataFrame, repeat the list or tuple
        (R-style) to the end of the DataFrame.
    :returns: A pandas DataFrame with an added column.
    """
    # TODO: Convert examples to notebook.
    # :Setup:

    # .. code-block:: python

    #     import pandas as pd
    #     import janitor
    #     data = {
    #         "a": [1, 2, 3] * 3,
    #         "Bell__Chart": [1, 2, 3] * 3,
    #         "decorated-elephant": [1, 2, 3] * 3,
    #         "animals": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    #     }
    #     df = pd.DataFrame(data)

    # :Example 1: Create a new column with a single value:

    # .. code-block:: python

    #     df.add_column("city_pop", 100000)

    # :Output:

    # .. code-block:: python

    #        a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    #     0  1            1                   1   rabbit  Cambridge    100000
    #     1  2            2                   2  leopard   Shanghai    100000
    #     2  3            3                   3     lion      Basel    100000
    #     3  1            1                   1   rabbit  Cambridge    100000
    #     4  2            2                   2  leopard   Shanghai    100000
    #     5  3            3                   3     lion      Basel    100000
    #     6  1            1                   1   rabbit  Cambridge    100000
    #     7  2            2                   2  leopard   Shanghai    100000
    #     8  3            3                   3     lion      Basel    100000

    # :Example 2: Create a new column with an iterator which fills to the
    # column
    # size:

    # .. code-block:: python

    #     df.add_column("city_pop", range(3), fill_remaining=True)

    # :Output:

    # .. code-block:: python

    #        a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    #     0  1            1                   1   rabbit  Cambridge         0
    #     1  2            2                   2  leopard   Shanghai         1
    #     2  3            3                   3     lion      Basel         2
    #     3  1            1                   1   rabbit  Cambridge         0
    #     4  2            2                   2  leopard   Shanghai         1
    #     5  3            3                   3     lion      Basel         2
    #     6  1            1                   1   rabbit  Cambridge         0
    #     7  2            2                   2  leopard   Shanghai         1
    #     8  3            3                   3     lion      Basel         2

    # :Example 3: Add new column based on mutation of other columns:

    # .. code-block:: python

    #     df.add_column("city_pop", df.Bell__Chart - 2 * df.a)

    # :Output:

    # .. code-block:: python

    #        a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    #     0  1            1                   1   rabbit  Cambridge        -1
    #     1  2            2                   2  leopard   Shanghai        -2
    #     2  3            3                   3     lion      Basel        -3
    #     3  1            1                   1   rabbit  Cambridge        -1
    #     4  2            2                   2  leopard   Shanghai        -2
    #     5  3            3                   3     lion      Basel        -3
    #     6  1            1                   1   rabbit  Cambridge        -1
    #     7  2            2                   2  leopard   Shanghai        -2
    #     8  3            3                   3     lion      Basel        -3

    df = df.copy()
    check("column_name", column_name, [str])

    if column_name in df.columns:
        raise ValueError(
            f"Attempted to add column that already exists: " f"{column_name}."
        )

    nrows = df.shape[0]

    if hasattr(value, "__len__") and not isinstance(
        value, (str, bytes, bytearray)
    ):
        # if `value` is a list, ndarray, etc.
        if len(value) > nrows:
            raise ValueError(
                f"`values` has more elements than number of rows "
                f"in your `DataFrame`. vals: {len(value)}, "
                f"df: {nrows}"
            )
        if len(value) != nrows and not fill_remaining:
            raise ValueError(
                f"Attempted to add iterable of values with length"
                f" not equal to number of DataFrame rows"
            )

        if len(value) == 0:
            raise ValueError(
                f"Values has to be an iterable of minimum length 1"
            )
        len_value = len(value)
    elif fill_remaining:
        # relevant if a scalar val was passed, yet fill_remaining == True
        len_value = 1
        value = [value]

    nrows = df.shape[0]

    if fill_remaining:
        times_to_loop = int(np.ceil(nrows / len_value))

        fill_values = list(value) * times_to_loop

        df[column_name] = fill_values[:nrows]
    else:
        df[column_name] = value

    return df


@pf.register_dataframe_method
def add_columns(
    df: pd.DataFrame, fill_remaining: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Add multiple columns to the dataframe.

    This method does not mutate the original DataFrame.

    Method to augment `add_column` with ability to add multiple columns in
    one go. This replaces the need for multiple `add_column` calls.

    Usage is through supplying kwargs where the key is the col name and the
    values correspond to the values of the new DataFrame column.

    Values passed can be scalar or iterable (list, ndarray, etc.)

    Usage example:

    .. code-block:: python

        x = 3
        y = np.arange(0, 10)
        df = pd.DataFrame(...).add_columns(x=x, y=y)

    :param df: A pandas dataframe.
    :param fill_remaining: If value is a tuple or list that is smaller than
        the number of rows in the DataFrame, repeat the list or tuple
        (R-style) to the end of the DataFrame. (Passed to `add_column`)
    :param kwargs: column, value pairs which are looped through in
        `add_column` calls.
    :returns: A pandas DataFrame with added columns.
    """
    # Note: error checking can pretty much be handled in `add_column`

    for col_name, values in kwargs.items():
        df = df.add_column(col_name, values, fill_remaining=fill_remaining)

    return df


@pf.register_dataframe_method
def limit_column_characters(
    df: pd.DataFrame, column_length: int, col_separator: str = "_"
) -> pd.DataFrame:
    """
    Truncate column sizes to a specific length.

    This method mutates the original DataFrame.

    Method chaining will truncate all columns to a given length and append
    a given separator character with the index of duplicate columns, except
    for the first distinct column name.

    :param df: A pandas dataframe.
    :param column_length: Character length for which to truncate all columns.
        The column separator value and number for duplicate column name does
        not contribute. Therefore, if all columns are truncated to 10
        characters, the first distinct column will be 10 characters and the
        remaining will be 12 characters (assuming a column separator of one
        character).
    :param col_separator: The separator to use for counting distinct column
        values. I think an underscore looks nicest, however a period is a
        common option as well. Supply an empty string (i.e. '') to remove the
        separator.
    :returns: A pandas DataFrame with truncated column lengths.
    """
    # :Example Setup:

    # .. code-block:: python

    #     import pandas as pd
    #     import janitor
    #     data_dict = {
    #         "really_long_name_for_a_column": range(10),
    #         "another_really_long_name_for_a_column": \
    #         [2 * item for item in range(10)],
    #         "another_really_longer_name_for_a_column": list("lllongname"),
    #         "this_is_getting_out_of_hand": list("longername"),
    #     }

    # :Example: Standard truncation:

    # .. code-block:: python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.limit_column_characters(7)

    # :Output:

    # .. code-block:: python

    #            really_  another another_1 this_is
    #     0        0        0         l       l
    #     1        1        2         l       o
    #     2        2        4         l       n
    #     3        3        6         o       g
    #     4        4        8         n       e
    #     5        5       10         g       r
    #     6        6       12         n       n
    #     7        7       14         a       a
    #     8        8       16         m       m
    #     9        9       18         e       e

    # :Example: Standard truncation with different separator character:

    # .. code-block:: python

    #     example_dataframe2 = pd.DataFrame(data_dict)
    #     example_dataframe2.limit_column_characters(7, ".")

    # .. code-block:: python

    #            really_  another another.1 this_is
    #     0        0        0         l       l
    #     1        1        2         l       o
    #     2        2        4         l       n
    #     3        3        6         o       g
    #     4        4        8         n       e
    #     5        5       10         g       r
    #     6        6       12         n       n
    #     7        7       14         a       a
    #     8        8       16         m       m
    #     9        9       18         e       e
    check("column_length", column_length, [int])
    check("col_separator", col_separator, [str])

    col_names = df.columns
    col_names = [col_name[:column_length] for col_name in col_names]

    col_name_set = set(col_names)
    col_name_count = dict()

    # If no columns are duplicates, we can skip the loops below.
    if len(col_name_set) == len(col_names):
        df.columns = col_names
        return df

    for col_name_to_check in col_name_set:
        count = 0
        for idx, col_name in enumerate(col_names):
            if col_name_to_check == col_name:
                col_name_count[idx] = count
                count += 1

    final_col_names = []
    for idx, col_name in enumerate(col_names):
        if col_name_count[idx] > 0:
            col_name_to_append = (
                col_name + col_separator + str(col_name_count[idx])
            )
            final_col_names.append(col_name_to_append)
        else:
            final_col_names.append(col_name)

    df.columns = final_col_names
    return df


@pf.register_dataframe_method
def row_to_names(
    df: pd.DataFrame,
    row_number: int = None,
    remove_row: bool = False,
    remove_rows_above: bool = False,
) -> pd.DataFrame:
    """
    Elevates a row to be the column names of a DataFrame.

    This method mutates the original DataFrame.

    Contains options to remove the elevated row from the DataFrame along with
    removing the rows above the selected row.

    Method chaining usage:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .row_to_names(
                row_number=0,
                remove_row=False,
                remove_rows_above=False,
            )
        )

    :param df: A pandas DataFrame.
    :param row_number: The row containing the variable names
    :param remove_row: Whether the row should be removed from the DataFrame.
        Defaults to False.
    :param remove_rows_above: Whether the rows above the selected row should
        be removed from the DataFrame. Defaults to False.
    :returns: A pandas DataFrame with set column names.
    """
    # :Setup:

    # .. code-block:: python

    #     import pandas as pd
    #     import janitor
    #     data_dict = {
    #         "a": [1, 2, 3] * 3,
    #         "Bell__Chart": [1, 2, 3] * 3,
    #         "decorated-elephant": [1, 2, 3] * 3,
    #         "animals": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3
    #     }

    # :Example: Move first row to column names:

    # .. code-block:: python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(0)

    # :Output:

    # .. code-block:: python

    #        1  1  1   rabbit  Cambridge
    #     0  1  1  1   rabbit  Cambridge
    #     1  2  2  2  leopard   Shanghai
    #     2  3  3  3     lion      Basel
    #     3  1  1  1   rabbit  Cambridge
    #     4  2  2  2  leopard   Shanghai
    #     5  3  3  3     lion      Basel
    #     6  1  1  1   rabbit  Cambridge
    #     7  2  2  2  leopard   Shanghai

    # :Example: Move first row to column names and remove row:

    # .. code-block:: python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(0, remove_row=True)

    # :Output:

    # .. code-block:: python

    #        1  1  1   rabbit  Cambridge
    #     1  2  2  2  leopard   Shanghai
    #     2  3  3  3     lion      Basel
    #     3  1  1  1   rabbit  Cambridge
    #     4  2  2  2  leopard   Shanghai
    #     5  3  3  3     lion      Basel
    #     6  1  1  1   rabbit  Cambridge
    #     7  2  2  2  leopard   Shanghai
    #     8  3  3  3     lion      Basel

    # :Example: Move first row to column names, remove row, \
    # and remove rows above selected row:

    # .. code-block:: python

    #     example_dataframe = pd.DataFrame(data_dict)
    #     example_dataframe.row_to_names(2, remove_row=True, \
    #         remove_rows_above=True)

    # :Output:

    # .. code-block:: python

    #        3  3  3     lion      Basel
    #     3  1  1  1   rabbit  Cambridge
    #     4  2  2  2  leopard   Shanghai
    #     5  3  3  3     lion      Basel
    #     6  1  1  1   rabbit  Cambridge
    #     7  2  2  2  leopard   Shanghai
    #     8  3  3  3     lion      Basel

    check("row_number", row_number, [int])

    df.columns = df.iloc[row_number, :]
    df.columns.name = None

    if remove_row:
        df = df.drop(df.index[row_number])

    if remove_rows_above:
        df = df.drop(df.index[range(row_number)])

    return df


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name")
def round_to_fraction(
    df: pd.DataFrame,
    column_name: Hashable = None,
    denominator: float = None,
    digits: float = np.inf,
) -> pd.DataFrame:
    """
    Round all values in a column to a fraction.

    This method mutates the original DataFrame.

    Taken from https://github.com/sfirke/janitor/issues/235.

    Also, optionally round to a specified number of digits.

    Method-chaining usage:

    .. code-block:: python

        # Round to two decimal places
        df = pd.DataFrame(...).round_to_fraction('a', 2)

    :param df: A pandas dataframe.
    :param column_name: Name of column to round to fraction.
    :param denominator: The denominator of the fraction for rounding
    :param digits: The number of digits for rounding after rounding to the
        fraction. Default is np.inf (i.e. no subsequent rounding)
    :returns: A pandas DataFrame with a column's values rounded.
    """
    # NOTE: THESE EXAMPLES SHOULD BE MOVED TO NOTEBOOKS.
    #     :Example Setup:

    # .. code-block:: python

    #     import pandas as pd
    #     import janitor
    #     data_dict = {
    #         "a": [1.23452345, 2.456234, 3.2346125] * 3,
    #         "Bell__Chart": [1/3, 2/7, 3/2] * 3,
    #         "decorated-elephant": [1/234, 2/13, 3/167] * 3,
    #         "animals": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    #     }

    # :Example: Rounding the first column to the nearest half:

    # .. code-block:: python

    # :Output:

    # .. code-block:: python

    #          a  Bell__Chart  decorated-elephant  animals     cities
    #     0  1.0     0.333333            0.004274   rabbit  Cambridge
    #     1  2.5     0.285714            0.153846  leopard   Shanghai
    #     2  3.0     1.500000            0.017964     lion      Basel
    #     3  1.0     0.333333            0.004274   rabbit  Cambridge
    #     4  2.5     0.285714            0.153846  leopard   Shanghai
    #     5  3.0     1.500000            0.017964     lion      Basel
    #     6  1.0     0.333333            0.004274   rabbit  Cambridge
    #     7  2.5     0.285714            0.153846  leopard   Shanghai
    #     8  3.0     1.500000            0.017964     lion      Basel

    # :Example: Rounding the first column to nearest third:

    # .. code-block:: python

    #     example_dataframe2 = pd.DataFrame(data_dict)
    #     example_dataframe2.round_to_fraction('a', 3)

    # :Output:

    # .. code-block:: python

    #               a  Bell__Chart  decorated-elephant  animals     cities
    #     0  1.333333     0.333333            0.004274   rabbit  Cambridge
    #     1  2.333333     0.285714            0.153846  leopard   Shanghai
    #     2  3.333333     1.500000            0.017964     lion      Basel
    #     3  1.333333     0.333333            0.004274   rabbit  Cambridge
    #     4  2.333333     0.285714            0.153846  leopard   Shanghai
    #     5  3.333333     1.500000            0.017964     lion      Basel
    #     6  1.333333     0.333333            0.004274   rabbit  Cambridge
    #     7  2.333333     0.285714            0.153846  leopard   Shanghai
    #     8  3.333333     1.500000            0.017964     lion      Basel

    # :Example 3: Rounding the first column to the nearest third and rounding \
    # each value to the 10,000th place:

    # .. code-block:: python

    #     example_dataframe2 = pd.DataFrame(data_dict)
    #     example_dataframe2.round_to_fraction('a', 3, 4)

    # :Output:

    # .. code-block:: python

    #             a  Bell__Chart  decorated-elephant  animals     cities
    #     0  1.3333     0.333333            0.004274   rabbit  Cambridge
    #     1  2.3333     0.285714            0.153846  leopard   Shanghai
    #     2  3.3333     1.500000            0.017964     lion      Basel
    #     3  1.3333     0.333333            0.004274   rabbit  Cambridge
    #     4  2.3333     0.285714            0.153846  leopard   Shanghai
    #     5  3.3333     1.500000            0.017964     lion      Basel
    #     6  1.3333     0.333333            0.004274   rabbit  Cambridge
    #     7  2.3333     0.285714            0.153846  leopard   Shanghai
    #     8  3.3333     1.500000            0.017964     lion      Basel

    if denominator:
        check("denominator", denominator, [float, int])

    if digits:
        check("digits", digits, [float, int])

    df[column_name] = round(df[column_name] * denominator, 0) / denominator
    if not np.isinf(digits):
        df[column_name] = round(df[column_name], digits)

    return df


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", dest_col_name="dest_column_name")
def transform_column(
    df: pd.DataFrame,
    column_name: Hashable,
    function: Callable,
    dest_column_name: str = None,
) -> pd.DataFrame:
    """
    Transform the given column in-place using the provided function.

    This method mutates the original DataFrame.

    Let's say we wanted to apply a log10 transform a column of data.

    Originally one would write code like this:

    .. code-block:: python

        # YOU NO LONGER NEED TO WRITE THIS!
        df[column_name] = df[column_name].apply(function)

    With the method chaining syntax, we can do the following instead:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .transform_column(column_name, function)
        )

    With the functional syntax:

    .. code-block:: python

        df = pd.DataFrame(...)
        df = transform_column(df, column_name, function)

    :param df: A pandas DataFrame.
    :param column_name: The column to transform.
    :param function: A function to apply on the column.
    :param dest_column_name: The column name to store the transformation result
        in. Defaults to None, which will result in the original column
        name being overwritten. If a name is provided here, then a new column
        with the transformed values will be created.
    :returns: A pandas DataFrame with a transformed column.
    """
    if dest_column_name is None:
        dest_column_name = column_name

    df[dest_column_name] = df[column_name].apply(function)
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_names="new_column_names")
def transform_columns(
    df: pd.DataFrame,
    column_names: Union[List[str], Tuple[str]],
    function: Callable,
    suffix: str = None,
    new_column_names: Dict[str, str] = None,
) -> pd.DataFrame:
    """
    Transform multiple columns through the same transformation.

    This method mutates the original DataFrame.

    Super syntactic sugar!

    Basically wraps `transform_column` and calls it repeatedly over all column
    names provided.

    User can optionally supply either a suffix to create a new set of columns
    with the specified suffix, or provide a dictionary mapping each original
    column name to its corresponding new column name. Note that all column
    names must be strings.

    A few examples below. Firstly, to just log10 transform a list of columns
    without creating new columns to hold the transformed values:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .transform_columns(['col1', 'col2', 'col3'], np.log10)
        )

    Secondly, to add a '_log' suffix when creating a new column, which we think
    is going to be the most common use case:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .transform_columns(
                ['col1', 'col2', 'col3'],
                np.log10,
                suffix="_log"
            )
        )

    Finally, to provide new names explicitly:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .transform_column(
                ['col1', 'col2', 'col3'],
                np.log10,
                new_column_names={
                    'col1': 'transform1',
                    'col2': 'transform2',
                    'col3': 'transform3',
                    }
                )
        )

    :param df: A pandas DataFrame.
    :param column_names: An iterable of columns to transform.
    :param function: A function to apply on each column.
    :param suffix: (optional) Suffix to use when creating new columns to hold
        the transformed values.
    :param new_column_names: (optional) An explicit mapping of old column names
        to new column names.
    :returns: A pandas DataFrame with transformed columns.
    """
    dest_column_names = dict(zip(column_names, column_names))

    check("column_names", column_names, [list, tuple])

    if suffix is not None and new_column_names is not None:
        raise ValueError(
            "only one of suffix or new_column_names should be specified"
        )

    if suffix:  # If suffix is specified...
        check("suffix", suffix, [str])
        for col in column_names:
            dest_column_names[col] = col + suffix

    if new_column_names:  # If new_column_names is specified...
        check("new_column_names", new_column_names, [dict])
        dest_column_names = new_column_names

    # Now, transform columns.
    for old_col, new_col in dest_column_names.items():
        df = transform_column(df, old_col, function, new_col)

    return df


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name")
def min_max_scale(
    df: pd.DataFrame,
    old_min=None,
    old_max=None,
    column_name=None,
    new_min=0,
    new_max=1,
) -> pd.DataFrame:
    """
    Scales data to between a minimum and maximum value.

    This method mutates the original DataFrame.

    If `minimum` and `maximum` are provided, the true min/max of the
    `DataFrame` or column is ignored in the scaling process and replaced with
    these values, instead.

    One can optionally set a new target minimum and maximum value using the
    `new_min` and `new_max` keyword arguments. This will result in the
    transformed data being bounded between `new_min` and `new_max`.

    If a particular column name is specified, then only that column of data
    are scaled. Otherwise, the entire dataframe is scaled.

    Method chaining syntax:

    .. code-block:: python

        df = pd.DataFrame(...).min_max_scale(column_name="a")

    Setting custom minimum and maximum:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .min_max_scale(
                column_name="a",
                new_min=2,
                new_max=10
            )
        )

    Setting a min and max that is not based on the data, while applying to
    entire dataframe:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .min_max_scale(
                old_min=0,
                old_max=14,
                new_min=0,
                new_max=1,
            )
        )

    The aforementioned example might be applied to something like scaling the
    isoelectric points of amino acids. While technically they range from
    approx 3-10, we can also think of them on the pH scale which ranges from
    1 to 14. Hence, 3 gets scaled not to 0 but approx. 0.15 instead, while 10
    gets scaled to approx. 0.69 instead.

    :param df: A pandas DataFrame.
    :param old_min, old_max (optional): Overrides for the current minimum and
        maximum values of the data to be transformed.
    :param new_min, new_max (optional): The minimum and maximum values of the
        data after it has been scaled.
    :param column_name (optional): The column on which to perform scaling.
    :returns: A pandas DataFrame with scaled data.
    """
    if (
        (old_min is not None)
        and (old_max is not None)
        and (old_max <= old_min)
    ):
        raise ValueError("`old_max` should be greater than `old_max`")

    if new_max <= new_min:
        raise ValueError("`new_max` should be greater than `new_min`")

    new_range = new_max - new_min

    if column_name:
        if old_min is None:
            old_min = df[column_name].min()
        if old_max is None:
            old_max = df[column_name].max()
        old_range = old_max - old_min
        df[column_name] = (
            df[column_name] - old_min
        ) * new_range / old_range + new_min
    else:
        if old_min is None:
            old_min = df.min().min()
        if old_max is None:
            old_max = df.max().max()
        old_range = old_max - old_min
        df = (df - old_min) * new_range / old_range + new_min
    return df


@pf.register_dataframe_method
def collapse_levels(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """
    Flatten multi-level column dataframe to a single level.

    This method mutates the original DataFrame.

    Given a `DataFrame` containing multi-level columns, flatten to single-
    level by string-joining the column labels in each level.

    After a `groupby` / `aggregate` operation where `.agg()` is passed a
    list of multiple aggregation functions, a multi-level `DataFrame` is
    returned with the name of the function applied in the second level.

    It is sometimes convenient for later indexing to flatten out this
    multi-level configuration back into a single level. This function does
    this through a simple string-joining of all the names across different
    levels in a single column.

    Method chaining syntax given two value columns `['max_speed', 'type']`:

    .. code-block:: python

        data = {"class": ["bird", "bird", "bird", "mammal", "mammal"],
                "max_speed": [389, 389, 24, 80, 21],
                "type": ["falcon", "falcon", "parrot", "Lion", "Monkey"]}

        df = (
            pd.DataFrame(data)
                .groupby('class')
                .agg(['mean', 'median'])
                .collapse_levels(sep='_')
        )

    Before applying ``.collapse_levels``, the ``.agg`` operation returns a
    multi-level column `DataFrame` whose columns are (level 1, level 2):

    .. code-block:: python

        [('class', ''), ('max_speed', 'mean'), ('max_speed', 'median'),
        ('type', 'mean'), ('type', 'median')]

    ``.collapse_levels`` then flattens the column names to:

    .. code-block:: python

        ['class', 'max_speed_mean', 'max_speed_median',
        'type_mean', 'type_median']

    :param df: A pandas DataFrame.
    :param sep: String separator used to join the column level names
    :returns: A flattened pandas DataFrame.
    """
    check("sep", sep, [str])

    # if already single-level, just return the DataFrame
    if not isinstance(df.columns.values[0], tuple):  # noqa: PD011
        return df

    df.columns = [
        sep.join([str(el) for el in tup if str(el) != ""])
        for tup in df.columns.values  # noqa: PD011
    ]

    return df


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", type="cleaning_style")
def currency_column_to_numeric(
    df: pd.DataFrame,
    column_name,
    cleaning_style: str = None,
    cast_non_numeric: dict = None,
    fill_all_non_numeric: float = None,
    remove_non_numeric: bool = False,
) -> pd.DataFrame:
    """
    Convert currency column to numeric.

    This method does not mutate the original DataFrame.

    This method allows one to take a column containing currency values,
    inadvertently imported as a string, and cast it as a float. This is
    usually the case when reading CSV files that were modified in Excel.
    Empty strings (i.e. `''`) are retained as `NaN` values.

    :param df: The DataFrame
    :param column_name: The column to modify
    :param cleaning_style: What style of cleaning to perform. If None, standard
        cleaning is applied. Options are:

            * 'accounting':
            Replaces numbers in parentheses with negatives, removes commas.

    :param cast_non_numeric: A dict of how to coerce certain strings. For
        example, if there are values of 'REORDER' in the DataFrame,
        {'REORDER': 0} will cast all instances of 'REORDER' to 0.
    :param fill_all_non_numeric: Similar to `cast_non_numeric`, but fills all
        strings to the same value. For example,  fill_all_non_numeric=1, will
        make everything that doesn't coerce to a currency 1.
    :param remove_non_numeric: Will remove rows of a DataFrame that contain
        non-numeric values in the `column_name` column. Defaults to `False`.
    :returns: A pandas DataFrame.
    """
    # TODO: Convert this to a notebook.
    # :Example Setup:

    # .. code-block:: python

    #     import pandas as pd
    #     import janitor
    #     data = {
    #         "a": ["-$1.00", "", "REPAY"] * 2 + ["$23.00", "",
    # "Other Account"],
    #         "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
    #         "decorated-elephant": [1, 2, 3] * 3,
    #         "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    #     }
    #     df = pd.DataFrame(data)

    # :Example 1: Coerce numeric values in column to float:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a")

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2   NaN     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5   NaN     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8   NaN     3.234612                   3         lion      Basel

    # :Example 2: Coerce numeric values in column to float, and replace a
    # string\
    # value with a specific value:

    # .. code-block:: python

    #     cast_non_numeric = {"REPAY": 22}
    #     df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  22.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  22.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8   NaN     3.234612                   3         lion      Basel

    # :Example 3: Coerce numeric values in column to float, and replace all\
    #     string value with a specific value:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", fill_all_non_numeric=35)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  35.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  35.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8  35.0     3.234612                   3         lion      Basel

    # :Example 4: Coerce numeric values in column to float, replace a string\
    #     value with a specific value, and replace remaining string values
    # with\
    #     a specific value:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
    #     fill_all_non_numeric=35)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  22.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  22.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8  35.0     3.234612                   3         lion      Basel

    # :Example 5: Coerce numeric values in column to float, and remove string\
    #     values:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", remove_non_numeric=True)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai

    # :Example 6: Coerce numeric values in column to float, replace a string\
    #     value with a specific value, and remove remaining string values:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
    #     remove_non_numeric=True)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  22.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  22.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai

    check("column_name", column_name, [str])

    column_series = df[column_name]
    if cleaning_style == "accounting":
        df.loc[:, column_name] = df[column_name].apply(
            _clean_accounting_column
        )
        return df

    if cast_non_numeric:
        check("cast_non_numeric", cast_non_numeric, [dict])

    _make_cc_patrial = partial(
        _currency_column_to_numeric, cast_non_numeric=cast_non_numeric
    )

    column_series = column_series.apply(_make_cc_patrial)

    if remove_non_numeric:
        df = df.loc[column_series != "", :]

    # _replace_empty_string_with_none is applied here after the check on
    # remove_non_numeric since "" is our indicator that a string was coerced
    # in the original column
    column_series = _replace_empty_string_with_none(column_series)

    if fill_all_non_numeric is not None:
        check("fill_all_non_numeric", fill_all_non_numeric, [int, float])
        column_series = column_series.fillna(fill_all_non_numeric)

    column_series = _replace_original_empty_string_with_none(column_series)

    df = df.assign(**{column_name: pd.to_numeric(column_series)})

    return df


@pf.register_dataframe_method
@deprecated_alias(search_cols="search_column_names")
def select_columns(
    df: pd.DataFrame, search_column_names: List[str], invert: bool = False
) -> pd.DataFrame:
    """
    Method-chainable selection of columns.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of columns available as well.

    Method-chaining example:

    .. code-block:: python

        df = pd.DataFrame(...).select_columns(['a', 'b', 'col_*'], invert=True)

    :param df: A pandas DataFrame.
    :param search_column_names: A list of column names or search strings to be
        used to select. Valid inputs include:
        1) an exact column name to look for
        2) a shell-style glob string (e.g., `*_thing_*`)
    :param invert: Whether or not to invert the selection.
        This will result in selection of the complement of the columns
        provided.
    :returns: A pandas DataFrame with the specified columns selected.
    :raises:
        TypeError: if input is not passed as a list.
    :raises:
        NameError: if one or more of the specified column names or
        search strings are not found in DataFrame columns.
    """
    if not isinstance(search_column_names, list):
        raise TypeError(
            "Column name(s) or search string(s) must be passed as list"
        )

    wildcards = {col for col in search_column_names if "*" in col}
    non_wildcards = set(search_column_names) - wildcards

    if not non_wildcards.issubset(df.columns):
        nonexistent_column_names = non_wildcards.difference(df.columns)
        raise NameError(
            f"{list(nonexistent_column_names)} missing from DataFrame"
        )

    missing_wildcards = []
    full_column_list = []

    for col in search_column_names:
        search_string = translate(col)
        columns = [col for col in df if re.match(search_string, col)]
        if len(columns) == 0:
            missing_wildcards.append(col)
        else:
            full_column_list.extend(columns)

    if len(missing_wildcards) > 0:
        raise NameError(
            f"Search string(s) {missing_wildcards} not found in DataFrame"
        )

    return (
        df.drop(columns=full_column_list) if invert else df[full_column_list]
    )


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
@deprecated_alias(statistic="statistic_column_name")
def impute(
    df: pd.DataFrame,
    column_name: Hashable,
    value: Any = None,
    statistic_column_name: str = None,
) -> pd.DataFrame:
    """
    Method-chainable imputation of values in a column.

    This method mutates the original DataFrame.

    Underneath the hood, this function calls the ``.fillna()`` method available
    to every pandas.Series object.

    Method-chaining example:

    .. code-block:: python

        import numpy as np
        import pandas as pd
        import janitor

        data = {
            "a": [1, 2, 3],
            "sales": np.nan,
            "score": [np.nan, 3, 2]}
        df = (
            pd.DataFrame(data)
            # Impute null values with 0
            .impute(column_name='sales', value=0.0)
            # Impute null values with median
            .impute(column_name='score', statistic_column_name='median')
        )

    Either one of ``value`` or ``statistic_column_name`` should be provided.

    If ``value`` is provided, then all null values in the selected column will
        take on the value provided.

    If ``statistic_column_name`` is provided, then all null values in the
    selected column will take on the summary statistic value of other non-null
    values.

    Currently supported statistics include:

    - ``mean`` (also aliased by ``average``)
    - ``median``
    - ``mode``
    - ``minimum`` (also aliased by ``min``)
    - ``maximum`` (also aliased by ``max``)

    :param df: A pandas DataFrame
    :param column_name: The name of the column on which to impute values.
    :param value: (optional) The value to impute.
    :param statistic_column_name: (optional) The column statistic to impute.
    :returns: An imputed pandas DataFrame.
    """
    # Firstly, we check that only one of `value` or `statistic` are provided.
    if value is not None and statistic_column_name is not None:
        raise ValueError(
            "Only one of `value` or `statistic` should be provided"
        )

    # If statistic is provided, then we compute the relevant summary statistic
    # from the other data.
    funcs = {
        "mean": np.mean,
        "average": np.mean,  # aliased
        "median": np.median,
        "mode": mode,
        "minimum": np.min,
        "min": np.min,  # aliased
        "maximum": np.max,
        "max": np.max,  # aliased
    }
    if statistic_column_name is not None:
        # Check that the statistic keyword argument is one of the approved.
        if statistic_column_name not in funcs.keys():
            raise KeyError(f"`statistic` must be one of {funcs.keys()}")

        value = funcs[statistic_column_name](
            df[column_name].dropna().to_numpy()
        )
        # special treatment for mode, because scipy stats mode returns a
        # moderesult object.
        if statistic_column_name == "mode":
            value = value.mode[0]

    # The code is architected this way - if `value` is not provided but
    # statistic is, we then overwrite the None value taken on by `value`, and
    # use it to set the imputation column.
    if value is not None:
        df[column_name] = df[column_name].fillna(value)
    return df


@pf.register_dataframe_method
def then(df: pd.DataFrame, func: Callable) -> pd.DataFrame:
    """
    Add an arbitrary function to run in the ``pyjanitor`` method chain.

    This method does not mutate the original DataFrame.

    :param df: A pandas dataframe.
    :param func: A function you would like to run in the method chain.
        It should take one parameter and return one parameter, each being the
        DataFrame object. After that, do whatever you want in the middle.
        Go crazy.
    :returns: A pandas DataFrame.
    """
    df = func(df)
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def dropnotnull(df: pd.DataFrame, column_name: Hashable) -> pd.DataFrame:
    """
    Drop rows that do not have null values in the given column.

    This method does not mutate the original DataFrame.

    Example usage:

    .. code-block:: python

        df = pd.DataFrame(...).dropnotnull('column3')

    :param df: A pandas DataFrame.
    :param column_name: The column name to drop rows from.
    :returns: A pandas DataFrame with dropped rows.
    """
    return df[pd.isna(df[column_name])]


@pf.register_dataframe_method
def find_replace(df, match: str = "exact", **mappings):
    """
    Perform a find-and-replace action on provided columns.

    Depending on use case, users can choose either exact, full-value matching,
    or regular-expression-based fuzzy matching
    (hence allowing substring matching in the latter case).
    For strings, the matching is always case sensitive.

    For instance, given a dataframe containing orders at a coffee shop:

    .. code-block:: python

        df = pd.DataFrame({
            'customer': ['Mary', 'Tom', 'Lila'],
            'order': ['ice coffee', 'lemonade', 'regular coffee']
        })

    Our task is to replace values `'ice coffee'` and `'regular coffee'`
    of the `'order'` column into `'latte'`.

    Example 1: Exact matching

    .. code-block:: python

        # Functional usage
        df = find_replace(
            df,
            match='exact',
            order={'ice coffee': 'latte', 'regular coffee': 'latte'},
        )

        # Method chaining usage
        df = df.find_replace(
            match='exact'
            order={'ice coffee': 'latte', 'regular coffee': 'latte'},
        )

    Example 2: Regular-expression-based matching

    .. code-block:: python

        # Functional usage
        df = find_replace(
            df,
            match='regex',
            order={'coffee$': 'latte'},
        )

        # Method chaining usage
        df = df.find_replace(
            match='regex',
            order={'coffee$': 'latte'},
        )

    To perform a find and replace on the entire dataframe,
    pandas' ``df.replace()`` function provides the appropriate functionality.
    You can find more detail on the replace_ docs.

    This function only works with column names that have no spaces
    or punctuation in them.
    For example, a column name ``item_name`` would work with ``find_replace``,
    because it is a contiguous string that can be parsed correctly,
    but ``item name`` would not be parsed correctly by the Python interpreter.

    If you have column names that might not be compatible,
    we recommend calling on ``clean_names()`` as the first method call.
    If, for whatever reason, that is not possible,
    then ``_find_replace()`` is available as a function
    that you can do a pandas pipe_ call on.

    .. _replace: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
    .. _pipe: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html

    :param df: A pandas DataFrame.
    :param match: Whether or not to perform an exact match or not.
        Valid values are "exact" or "regex".
    :param mappings: keyword arguments corresponding to column names
        that have dictionaries passed in indicating what to find (keys)
        and what to replace with (values).
    """  # noqa: E501
    for column_name, mapper in mappings.items():
        df = _find_replace(df, column_name, mapper, match=match)
    return df


def _find_replace(
    df: pd.DataFrame, column_name: str, mapper: Dict, match: str = "exact"
) -> pd.DataFrame:
    """
    Utility function for ``find_replace``.

    The code in here was the original implementation of ``find_replace``,
    but we decided to change out the front-facing API to accept
    kwargs + dictionaries for readability,
    and instead dispatch underneath to this function.
    This implementation was kept
    because it has a number of validations that are quite useful.

    :param df: A pandas DataFrame.
    :param column_name: The column on which the find/replace action is to be
        made. Must be a string.
    :param mapper: A dictionary that maps "thing to find" -> "thing to
        replace".  Note: Does not support null-value replacement.
    :param match: A string that dictates whether exact match or
        regular-expression-based fuzzy match will be used for finding patterns.
        Default to "exact". Can only be "exact" or "regex".
    :returns: A pandas DataFrame.
    :raises: ValueError
    """
    if any(map(pd.isna, mapper.keys())):
        raise ValueError(
            "find_replace() does not support null replacement. "
            "Use DataFrame.fillna() instead."
        )
    if match.lower() not in ("exact", "regex"):
        raise ValueError("`match` can only be 'exact' or 'regex'.")

    if match.lower() == "exact":
        df[column_name] = df[column_name].apply(lambda x: mapper.get(x, x))
    if match.lower() == "regex":
        for k, v in mapper.items():
            condition = df[column_name].str.contains(k, regex=True)
            df = df.update_where(condition, column_name, v)
    return df


@pf.register_dataframe_method
@deprecated_alias(target_col="target_column_name")
def update_where(
    df: pd.DataFrame,
    conditions: Any,
    target_column_name: Hashable,
    target_val: Any,
) -> pd.DataFrame:
    """
    Add multiple conditions to update a column in the dataframe.

    This method mutates the original DataFrame.

    Example usage:

    .. code-block:: python

        # The dataframe must be assigned to a variable first.
        data = {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [0, 0, 0, 0]
        }
        df = pd.DataFrame(data)
        df = (
            df
            .update_where(
                condition=(df['a'] > 2) & (df['b'] < 8),
                target_column_name='c',
                target_val=10)
            )
        # a b  c
        # 1 5  0
        # 2 6  0
        # 3 7 10
        # 4 8  0

    :param df: The pandas DataFrame object.
    :param conditions: conditions used to update a target column and target
        value
    :param target_column_name: Column to be updated. If column does not exist
        in dataframe, a new column will be created; note that entries that do
        not get set in the new column will be null.
    :param target_val: Value to be updated
    :returns: An updated pandas DataFrame.
    :raises: IndexError if **conditions** does not have the same length as
        **df**.
    """
    df.loc[conditions, target_column_name] = target_val
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def to_datetime(
    df: pd.DataFrame, column_name: Hashable, **kwargs
) -> pd.DataFrame:
    """
    Method-chainable to_datetime.

    This method mutates the original DataFrame.

    Functional usage syntax:

    .. code-block:: python

        df = to_datetime(df, 'col1', format='%Y%m%d')

    Method chaining syntax:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).to_datetime('col1', format='%Y%m%d')

    :param df: A pandas DataFrame.
    :param column_name: Column name.
    :param kwargs: provide any kwargs that pd.to_datetime can take.
    :returns: A pandas DataFrame with updated datetime data.
    """
    df[column_name] = pd.to_datetime(df[column_name], **kwargs)

    return df


@pf.register_dataframe_method
@deprecated_alias(new_column="new_column_name", agg_column="agg_column_name")
def groupby_agg(
    df: pd.DataFrame,
    by: Union[List, str],
    new_column_name: str,
    agg_column_name: str,
    agg: Union[Callable, str],
) -> pd.DataFrame:
    """
    Shortcut for assigning a groupby-transform to a new column.

    This method does not mutate the original DataFrame.

    Without this function, we would have to write a verbose line:

    .. code-block:: python

        df = df.assign(...=df.groupby(...)[...].tranform(...))

    Now, this function can be method-chained:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).groupby_agg(by='group',
                                           agg='mean',
                                           agg_column_name="col1"
                                           new_column_name='col1_mean_by_group')

    Example Link : https://pyjanitor.readthedocs.io/notebooks/groupby_agg.html

    :param df: A pandas DataFrame.
    :param by: Column(s) to groupby on, either a `str` or
               a `list` of `str`
    :param new_column_name: Name of the aggregation output column.
    :param agg_column_name: Name of the column to aggregate over.
    :param agg: How to aggregate.
    :param axis: Split along rows (0) or columns (1).
    :returns: A pandas DataFrame.
    """
    new_col = df.groupby(by)[agg_column_name].transform(agg)
    df_new = df.assign(**{new_column_name: new_col})
    return df_new


@pf.register_dataframe_accessor("data_description")
class DataDescription:
    """
    High-level description of data present in this DataFrame.

    This is a custom data accessor.
    """

    def __init__(self, data):
        """Initialize DataDescription class."""
        self._data = data
        self._desc = dict()

    def _get_data_df(self) -> pd.DataFrame:
        df = self._data

        data_dict = dict()
        data_dict["column_name"] = df.columns.tolist()
        data_dict["type"] = df.dtypes.tolist()
        data_dict["count"] = df.count().tolist()
        data_dict["pct_missing"] = (1 - (df.count() / len(df))).tolist()
        data_dict["description"] = [self._desc.get(c, "") for c in df.columns]

        return pd.DataFrame(data_dict).set_index("column_name")

    @property
    def df(self) -> pd.DataFrame:
        """Get a table of descriptive information in a DataFrame format."""
        return self._get_data_df()

    def __repr__(self):
        """Human-readable representation of the `DataDescription` object."""
        return str(self._get_data_df())

    def display(self):
        """Print the table of descriptive information about this DataFrame."""
        print(self)

    def set_description(self, desc: Union[List, Dict]):
        """
        Update the description for each of the columns in the DataFrame.

        :param desc: The structure containing the descriptions to update
        """
        if isinstance(desc, list):
            if len(desc) != len(self._data.columns):
                raise ValueError(
                    f"Length of description list "
                    f"({len(desc)}) does not match number of columns in "
                    f"DataFrame ({len(self._data.columns)})"
                )

            self._desc = dict(zip(self._data.columns, desc))

        elif isinstance(desc, dict):
            self._desc = desc


@pf.register_dataframe_method
@deprecated_alias(from_column="from_column_name", to_column="to_column_name")
def bin_numeric(
    df: pd.DataFrame,
    from_column_name: Hashable,
    to_column_name: Hashable,
    num_bins: int = 5,
    labels: str = None,
) -> pd.DataFrame:
    """
    Generate a new column that labels bins for a specified numeric column.

    This method mutates the original DataFrame.

    Makes use of pandas cut() function to bin data of one column, generating a
    new column with the results.

    .. code-block:: python

        import pandas as pd
        import janitor
        df = (
            pd.DataFrame(...)
            .bin_numeric(
                from_column_name='col1',
                to_column_name='col1_binned',
                num_bins=3,
                labels=['1-2', '3-4', '5-6']
                )
        )

    :param df: A pandas DataFrame.
    :param from_column_name: The column whose data you want binned.
    :param to_column_name: The new column to be created with the binned data.
    :param num_bins: The number of bins to be utilized.
    :param labels: Optionally rename numeric bin ranges with labels. Number of
        label names must match number of bins specified.
    :return: A pandas DataFrame.
    """
    if not labels:
        df[str(to_column_name)] = pd.cut(
            df[str(from_column_name)], bins=num_bins
        )
    else:
        if not len(labels) == num_bins:
            raise ValueError(f"Number of labels must match number of bins.")

        df[str(to_column_name)] = pd.cut(
            df[str(from_column_name)], bins=num_bins, labels=labels
        )

    return df


@pf.register_dataframe_method
def drop_duplicate_columns(
    df: pd.DataFrame, column_name: Hashable, nth_index: int = 0
) -> pd.DataFrame:
    """
    Remove a duplicated column specified by column_name, its index.

    This method does not mutate the original DataFrame.

    Column order 0 is to remove the first column,
           order 1 is to remove the second column, and etc

    The corresponding tidyverse R's library is:
    `select(-<column_name>_<nth_index + 1>)`

    Method chaining syntax:

    .. code-block:: python

        df = pd.DataFrame({
            "a": range(10),
            "b": range(10),
            "A": range(10, 20),
            "a*": range(20, 30),
        }).clean_names(remove_special=True)

        # remove a duplicated second 'a' column
        df.drop_duplicate_columns(column_name="a", nth_index=1)



    :param df: A pandas DataFrame
    :param column_name: Column to be removed
    :param nth_index: Among the duplicated columns,
      select the nth column to drop.
    :return: A pandas DataFrame
    """
    cols = df.columns.to_list()
    col_indexes = [
        col_idx
        for col_idx, col_name in enumerate(cols)
        if col_name == column_name
    ]

    # given that a column could be duplicated,
    # user could opt based on its order
    removed_col_idx = col_indexes[nth_index]
    # get the column indexes without column that is being removed
    filtered_cols = [
        c_i for c_i, c_v in enumerate(cols) if c_i != removed_col_idx
    ]

    return df.iloc[:, filtered_cols]


@pf.register_dataframe_method
def take_first(
    df: pd.DataFrame,
    subset: Union[Hashable, Iterable[Hashable]],
    by: Hashable,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Take the first row within each group specified by `subset`.

    This method does not mutate the original DataFrame.

    .. code-block:: python

        import pandas as pd
        import janitor

        data = {
            "a": ["x", "x", "y", "y"],
            "b": [0, 1, 2, 3]
        }
        df = pd.DataFrame(data)

        df.take_first(subset="a", by="b")

    :param df: A pandas DataFrame.
    :param subset: Column(s) defining the group.
    :param by: Column to sort by.
    :param ascending: Whether or not to sort in ascending order, `bool`.
    :returns: A pandas DataFrame.
    """
    result = df.sort_values(by=by, ascending=ascending).drop_duplicates(
        subset=subset, keep="first"
    )

    return result


@pf.register_dataframe_method
def shuffle(
    df: pd.DataFrame, random_state=None, reset_index=True
) -> pd.DataFrame:
    """
    Shuffle the rows of the DataFrame.

    This method does not mutate the original DataFrame.

    Super-sugary syntax! Underneath the hood, we use ``df.sample(frac=1)``,
    with the option to set the random state.

    Example usage:

    .. code-block:: python

        df = pd.DataFrame(...).shuffle()

    :param df: A pandas DataFrame
    :param random_state: (optional) A seed for the random number generator.
    :param reset_index: (optional) Resets index to default integers
    """
    result = df.sample(frac=1, random_state=random_state)
    if reset_index:
        result = result.reset_index(drop=True)
    return result


@pf.register_dataframe_method
def join_apply(df: pd.DataFrame, func: Callable, new_column_name: str):
    """
    Join the result of applying a function across dataframe rows.

    This method does not mutate the original DataFrame.

    This is a convenience function that allows us to apply arbitrary functions
    that take any combination of information from any of the columns. The only
    requirement is that the function signature takes in a row from the
    DataFrame.

    The example below shows us how to sum the result of two columns into a new
    column.

    .. code-block:: python

        df = (
            pd.DataFrame({'a':[1, 2, 3], 'b': [2, 3, 4]})
            .join_apply(lambda x: 2 * x['a'] + x['b'], new_column_name="2a+b")
        )

    This following example shows us how to use conditionals in the same
    function.

    .. code-block:: python

        def take_a_if_even(x):
            if x['a'] % 2:
                return x['a']
            else:
                return x['b']

        df = (
            pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
            .join_apply(take_a_if_even, 'a_if_even')
        )

    :param df: A pandas DataFrame
    :param func: A function that is applied elementwise across all rows of the
        DataFrame.
    :param new_column_name: New column name.
    """
    df = df.copy().join(df.apply(func, axis=1).rename(new_column_name))
    return df


@pf.register_dataframe_method
def flag_nulls(
    df: pd.DataFrame,
    column_name: Hashable = "null_flag",
    columns: Union[str, Iterable[str], Hashable] = None,
) -> pd.DataFrame:
    """
    Creates a new column to indicate whether you have null values in a given
    row. If the columns parameter is not set, looks across the entire
    DataFrame, otherwise will look only in the columns you set.

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        data = pd.DataFrame(
            {'a': [1, 2, None, 4],
             'b': [5.0, None, 7.0, 8.0]})

        df.flag_nulls()
        #  'a' | 'b'  | 'null_flag'
        #   1  | 5.0  |   0
        #   2  | None |   1
        # None | 7.0  |   1
        #   4  | 8.0  |   0

        jn.functions.flag_nulls(data)
        #  'a' | 'b'  | 'null_flag'
        #   1  | 5.0  |   0
        #   2  | None |   1
        # None | 7.0  |   1
        #   4  | 8.0  |   0

        df.flag_nulls(columns=['b'])
        #  'a' | 'b'  | 'null_flag'
        #   1  | 5.0  |   0
        #   2  | None |   1
        # None | 7.0  |   0
        #   4  | 8.0  |   0


    :param df: Input Pandas dataframe.
    :param column_name: Name for the output column. Defaults to 'null_flag'.
    :param columns: List of columns to look at for finding null values. If you
        only want to look at one column, you can simply give its name. If set
        to None (default), all DataFrame columns are used.
    :returns: Input dataframe with the null flag column.
    :raises: ValueError
    """
    # Sort out columns input
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.columns
    elif not isinstance(columns, Iterable):
        # catches other hashable types
        columns = [columns]

    # Input sanitation checks
    check_column(df, columns)
    check_column(df, [column_name], present=False)

    # This algorithm works best for n_rows >> n_cols. See issue #501
    null_array = np.zeros(len(df))
    for col in columns:
        null_array = np.logical_or(null_array, pd.isna(df[col]))

    df = df.copy()
    df[column_name] = null_array.astype(int)
    return df


@pf.register_dataframe_method
def count_cumulative_unique(
    df: pd.DataFrame,
    column_name: Hashable,
    dest_column_name: str,
    case_sensitive: bool = True,
) -> pd.DataFrame:
    """
    Generates a running total of cumulative unique values in a given column.

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.functions.count_cumulative_unique(
            df=df,
            column_name='animals',
            dest_column_name='animals_unique_count',
            case_sensitive=True
        )

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor

        df = pd.DataFrame(...)

        df = df.count_cumulative_unique(
            column_name='animals',
            dest_column_name='animals_unique_count',
            case_sensitive=True
        )

    A new column will be created containing a running
    count of unique values in the specified column.
    If `case_sensitive` is `True`, then the case of
    any letters will matter (i.e., 'a' != 'A');
    othewise, the case of any letters will not matter.

    This method mutates the original DataFrame.

    :param df: A pandas dataframe.
    :param column_name: Name of the column containing
        values from which a running count of unique values
        will be created.
    :param dest_column_name: The name of the new column containing the
        cumulative count of unique values that will be created.
    :param case_sensitive: Whether or not uppercase and lowercase letters
        will be considered equal (e.g., 'A' != 'a' if `True`).

    :returns: A pandas DataFrame with a new column containing a cumulative
        count of unique values from another column.
    """

    if not case_sensitive:
        # Make it so that the the same uppercase and lowercase
        # letter are treated as one unique value
        df[column_name] = df[column_name].astype(str).map(str.lower)

    df[dest_column_name] = (
        (
            df[[column_name]]
            .drop_duplicates()
            .assign(dummyabcxyz=1)
            .dummyabcxyz.cumsum()
        )
        .reindex(df.index)
        .ffill()
        .astype(int)
    )

    return df


@pf.register_series_method
def toset(series: pd.Series) -> Set:
    """
    Return a set of the values.

    These are each a scalar type, which is a Python scalar
    (for str, int, float) or a pandas scalar
    (for Timestamp/Timedelta/Interval/Period)

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        series = pd.Series(...)
        s = jn.functions.toset(series=series)

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor

        series = pd.Series(...)
        s = series.toset()

    :param series: A pandas series.
    :returns: A set of values.
    """

    return set(series.tolist())


@pf.register_dataframe_method
def jitter(
    df: pd.DataFrame,
    column_name: Hashable,
    dest_column_name: str,
    scale: np.number,
    clip: Optional[Iterable[np.number]] = None,
    random_state: Optional[np.number] = None,
) -> pd.DataFrame:
    """
    Adds Gaussian noise (jitter) to the values of a column.

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.functions.jitter(
            df=df,
            column_name='values',
            dest_column_name='values_jitter',
            scale=1.0,
            clip=None,
            random_state=None,
        )

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor

        df = pd.DataFrame(...)

        df = df.jitter(
            column_name='values',
            dest_column_name='values_jitter',
            scale=1.0,
            clip=None,
            random_state=None,
        )

    A new column will be created containing the values of the original column
    with Gaussian noise added.
    For each value in the column, a Gaussian distribution is created
    having a location (mean) equal to the value
    and a scale (standard deviation) equal to `scale`.
    A random value is then sampled from this distribution,
    which is the jittered value.
    If a tuple is supplied for `clip`,
    then any values of the new column less than `clip[0]`
    will be set to `clip[0]`,
    and any values greater than `clip[1]` will be set to `clip[1]`.
    Additionally, if a numeric value is supplied for `random_state`,
    this value will be used to set the random seed used for sampling.
    NaN values are ignored in this method.

    This method mutates the original DataFrame.

    :param df: A pandas dataframe.
    :param column_name: Name of the column containing
        values to add Gaussian jitter to.
    :param dest_column_name: The name of the new column containing the
        jittered values that will be created.
    :param scale: A positive value multiplied by the original
        column value to determine the scale (standard deviation) of the
        Gaussian distribution to sample from. (A value of zero results in
        no jittering.)
    :param clip: An iterable of two values (minimum and maximum) to clip
        the jittered values to, default to None.
    :param random_state: A interger or 1-d array value used to set the random
        seed, default to None.

    :returns: A pandas DataFrame with a new column containing Gaussian-
        jittered values from another column.
    """

    # Check types
    check("scale", scale, [int, float])

    # Check that `column_name` is a numeric column
    if not np.issubdtype(df[column_name].dtype, np.number):
        raise TypeError(f"{column_name} must be a numeric column.")

    if scale <= 0:
        raise ValueError("`scale` must be a numeric value greater than 0.")
    values = df[column_name]
    if random_state is not None:
        np.random.seed(random_state)
    result = np.random.normal(loc=values, scale=scale)
    if clip:
        # Ensure `clip` has length 2
        if len(clip) != 2:
            raise ValueError("`clip` must be an iterable of length 2.")
        # Ensure the values in `clip` are ordered as min, max
        if clip[1] < clip[0]:
            raise ValueError("`clip[0]` must be less than `clip[1]`.")
        result = np.clip(result, *clip)
    df[dest_column_name] = result

    return df


@pf.register_dataframe_method
def sort_naturally(
    df: pd.DataFrame, column_name: str, **natsorted_kwargs
) -> pd.DataFrame:
    """
    Sort an DataFrame by a column using "natural" sorting.

    Natural sorting is distinct from
    the default lexiographical sorting provided by ``pandas``.
    For example, given the following list of items:

        ["A1", "A11", "A3", "A2", "A10"]

    lexicographical sorting would give us:


        ["A1", "A10", "A11", "A2", "A3"]

    By contrast, "natural" sorting would give us:

        ["A1", "A2", "A3", "A10", "A11"]

    This function thus provides "natural" sorting
    on a single column of a dataframe.

    To accomplish this, we do a natural sort
    on the unique values that are present in the dataframe.
    Then, we reconstitute the entire dataframe
    in the naturally sorted order.

    Natural sorting is provided by the Python package natsort_.

    .. _natsort: https://natsort.readthedocs.io/en/master/index.html

    All keyword arguments to ``natsort`` should be provided
    after the column name to sort by is provided.
    They are passed through to the ``natsorted`` function.

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.sort_naturally(
            df=df,
            column_name='alphanumeric_column',
        )

    Method chaining usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor

        df = pd.DataFrame(...)

        df = df.sort_naturally(
            column_name='alphanumeric_column',
        )

    :param df: A pandas DataFrame.
    :param column_name: The column on which natural sorting should take place.
    :param natsorted_kwargs: Keyword arguments to be passed
        to natsort's ``natsorted`` function.
    """
    new_order = index_natsorted(df[column_name], **natsorted_kwargs)
    return df.iloc[new_order, :]
