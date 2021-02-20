""" General purpose data cleaning functions. """

import collections
import datetime as dt
import inspect
import re
import unicodedata
import warnings
from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pandas_flavor as pf
from multipledispatch import dispatch
from natsort import index_natsorted
from pandas.api.types import union_categoricals, is_list_like
from pandas.errors import OutOfBoundsDatetime
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

from .errors import JanitorError
from .utils import (
    _computations_expand_grid,
    _clean_accounting_column,
    _computations_as_categorical,
    _computations_complete,
    _computations_pivot_longer,
    _computations_pivot_wider,
    _currency_column_to_numeric,
    _data_checks_pivot_longer,
    _data_checks_pivot_wider,
    _process_text,
    _replace_empty_string_with_none,
    _replace_original_empty_string_with_none,
    _select_columns,
    _strip_underscores,
    check,
    check_column,
    deprecated_alias,
)


def unionize_dataframe_categories(
    *dataframes, column_names: Optional[Iterable[pd.CategoricalDtype]] = None
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

    Usage example for concatenation of categorical column-containing
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
    :raises TypeError: if any inputs are not pandas DataFrames.
    """

    if any(not isinstance(df, pd.DataFrame) for df in dataframes):
        raise TypeError("Inputs must all be dataframes.")

    if column_names is None:
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
    :param source: column or row to move
    :param target: column or row to move adjacent to
    :param position: Specifies whether the Series is moved to before or
        after the adjacent Series. Values can be either 'before' or 'after';
        defaults to 'before'.
    :param axis: Axis along which the function is applied. 0 to move a
        row, 1 to move a column.
    :returns: The dataframe with the Series moved.
    :raises ValueError: if ``axis`` is not ``0`` or ``1``.
    :raises ValueError: if ``position`` is not ``before`` or ``after``.
    :raises ValueError: if  ``source`` row or column is not in dataframe.
    :raises ValueError: if ``target`` row or column is not in dataframe.
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


FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", "")]


def _normalize_1(col_name: Hashable) -> str:
    """Perform normalization of column name."""
    result = str(col_name)
    for search, replace in FIXES:
        result = re.sub(search, replace, result)  # noqa: PD005
    return result


def _strip_accents(col_name: str) -> str:
    """Remove accents from a DataFrame column name.
    .. _StackOverflow: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-strin
    """  # noqa: E501

    return "".join(
        letter
        for letter in unicodedata.normalize("NFD", col_name)
        if not unicodedata.combining(letter)
    )


@pf.register_dataframe_method
def remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows and columns that are completely null.

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
    df: pd.DataFrame,
    column_names: Optional[Union[str, Iterable[str], Hashable]] = None,
) -> pd.DataFrame:
    """Return all duplicate rows.

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
    :param column_names: (optional) A column name or an iterable
        (list or tuple) of column names. Following pandas API, this only
        considers certain columns for identifying duplicates. Defaults to using
        all columns.
    :returns: The duplicate rows, as a pandas DataFrame.
    """
    dupes = df.duplicated(subset=column_names, keep=False)
    return df[dupes == True]  # noqa: E712


def As_Categorical(
    categories: Optional[
        Union[List, Set, Tuple, pd.Series, np.ndarray]
    ] = None,
    order: Optional[str] = None,
) -> NamedTuple:
    """
    Helper function for `encode_categorical`. It makes creating the
    `categories` and `order` more explicit. Inspired by pd.NamedAgg.

    :param categories: list-like object to create new categorical column.
    :param order: string object that can be either "sort" or "appearance".
        If "sort", the `categories` argument will be sorted with np.sort;
        if "apperance", the `categories` argument will be used as is.
    :returns: A namedtuple of (`categories`, `order`).
    """
    AsCategorical = collections.namedtuple(
        "AsCategorical", ["categories", "order"], defaults=(None, None)
    )

    return AsCategorical._make((categories, order))


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def encode_categorical(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable] = None,
    **kwargs,
) -> pd.DataFrame:
    """Encode the specified columns with Pandas'
    `category dtype <http://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_.

    Categories and order can be explicitly specified via the `kwargs` option, which is a
    pairing of column name and a tuple of (categories, order).

    The `janitor.As_Categorical` function is provided to make it clearer what the arguments
    to the function are.

    It is syntactic sugar around `pd.Categorical`.

    This method does not mutate the original DataFrame.

    .. note:: In versions < 0.20.11, this method mutates the original DataFrame.

    **Examples:**


    .. code-block:: python

               col1	col2	col3
        0	2.0	a	2020-01-01
        1	1.0	b	2020-01-02
        2	3.0	c	2020-01-03
        3	1.0	d	2020-01-04
        4	NaN	a	2020-01-05

        df.dtypes

        col1           float64
        col2            object
        col3    datetime64[ns]
        dtype: object

    Specific columns can be converted to category type:

    .. code-block:: python

        df = (pd.DataFrame(...)
                .encode_categorical(
                    column_names=['col1', 'col2', 'col3']
                    )
            )

        df.dtypes

        col1    category
        col2    category
        col3    category
        dtype: object

    Note that for the code above, the categories were inferred from
    the columns, and is unordered::

        df['col3']
        0   2020-01-01
        1   2020-01-02
        2   2020-01-03
        3   2020-01-04
        4   2020-01-05
        Name: col3, dtype: category
        Categories (5, datetime64[ns]):
        [2020-01-01, 2020-01-02, 2020-01-03, 2020-01-04, 2020-01-05]


    Explicit categories can be provided, and ordered via the ``kwargs``
    parameter::

        df = (pd.DataFrame(...)
                .as_categorical(
                    col1 = ([3, 2, 1, 4], "appearance"),
                    col2 = (['a','d','c','b'], "sort")
                    )
            )

        df['col1']
        0      2
        1      1
        2      3
        3      1
        4    NaN
        Name: col1, dtype: category
        Categories (4, int64): [3 < 2 < 1 < 4]

        df['col2']
        0    a
        1    b
        2    c
        3    d
        4    a
        Name: col2, dtype: category
        Categories (4, object): [a < b < c < d]

    When the `order` parameter is "appearance", the categories argument is used as-is;
    if the `order` is "sort", the categories argument is sorted in ascending order;
    if `order` is ``None``, then the categories argument is applied unordered.

    The ``janitor.As_Categorical`` function can also be used to make clearer
    what the arguments to the function are::

        df = (pd.DataFrame(...)
                .as_categorical(
                    col1 = As_Categorical(
                                categories = [3, 2, 1, 4],
                                order = "appearance"
                                ),
                    col2 = As_Categorical(
                                categories = ['a','d','c','b'],
                                order = "sort"
                                )
                    )
            )

    A User Warning will be generated if some or all of the unique values
    in the column are not present in the provided `categories` argument.

    .. code-block:: python

        df = (pd.DataFrame(...)
                .encode_categorical(
                    col1 = As_Categorical(
                            categories = [4, 5, 6],
                            order = "appearance"
                            )
            )

        UserWarning: None of the values in col1 are in [4, 5, 6];
                     this might create nulls for all your values
                     in the new categorical column.

        df['col1']
        0    NaN
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        Name: col1, dtype: category
        Categories (3, int64): [4 < 5 < 6]


    .. note:: if ``categories`` is None in the ``kwargs`` tuple, then the
        values for `categories` are inferred from the column; if `order`
        is None, then the values for categories are applied unordered.

    .. note:: ``column_names`` and ``kwargs`` parameters cannot be used at
        the same time.

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

    - With ``column_names``::

        categorical_cols = ['col1', 'col2', 'col4']
        df = jn.encode_categorical(
                    df,
                    columns = categorical_cols)  # one way

    - With ``kwargs``::

        df = jn.encode_categorical(
                    df,
                    col1 = (categories, order),
                    col2 = jn.As_Categorical(
                                categories = [values],
                                order="sort"/"appearance"/None
                                )
                )

    Method chaining syntax:

    - With ``column_names``::

        categorical_cols = ['col1', 'col2', 'col4']
        df = (pd.DataFrame(...)
                .encode_categorical(columns=categorical_cols)
            )

    - With ``kwargs``::

        df = (
            pd.DataFrame(...)
            .encode_categorical(
                col1 = (categories, order),
                col2 = jn.As_Categorical(
                            categories = [values],
                            order="sort"/"appearance"/None
                            )
        )


    :param df: The pandas DataFrame object.
    :param column_names: A column name or an iterable (list or
        tuple) of column names.
    :param kwargs: A pairing of column name to a tuple of (`categories`, `order`).
        There is also the `janitor.As_Categorical` function, which creates a
        namedtuple of (`categories`, `order`) to make it clearer what the arguments
        are. This is useful in creating categorical columns that are ordered, or
        if the user needs to explicitly specify the categories.
    :returns: A pandas DataFrame.
    :raises JanitorError: if a column specified within ``column_names``
        is not found in the DataFrame.
    :raises JanitorError: if ``column_names`` is not hashable
        nor iterable.
    :raises ValueError: if both ``column_names`` and ``kwargs`` are provided.
    """  # noqa: E501

    df = df.copy()

    if all((column_names, kwargs)):
        raise ValueError(
            """
            Only one of `column_names` or `kwargs`
            can be provided.
            """
        )
    # column_names deal with only category dtype (unordered)
    # kwargs takes care of scenarios where user wants an ordered category
    # or user supplies specific categories to create the categorical
    if column_names:
        if isinstance(column_names, (list, Tuple)):
            for col in column_names:
                if col not in df.columns:
                    raise JanitorError(
                        f"{col} missing from DataFrame columns!"
                    )
                df[col] = pd.Categorical(df[col])
        elif isinstance(column_names, Hashable):
            if column_names not in df.columns:
                raise JanitorError(
                    f"{column_names} missing from DataFrame columns!"
                )
            df[column_names] = pd.Categorical(df[column_names])
        else:
            raise JanitorError(
                "kwarg `column_names` must be hashable or iterable!"
            )
        return df

    df = _computations_as_categorical(df, **kwargs)
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def label_encode(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable]
) -> pd.DataFrame:
    """Convert labels into numerical data.

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
    :param column_names: A column name or an iterable (list
        or tuple) of column names.
    :returns: A pandas DataFrame.
    """
    df = _label_encode(df, column_names)
    return df


@dispatch(pd.DataFrame, (list, tuple))
def _label_encode(df, column_names):
    le = LabelEncoder()
    check_column(df, column_names=column_names, present=True)
    for col in column_names:
        df[f"{col}_enc"] = le.fit_transform(df[col])
    return df


@dispatch(pd.DataFrame, str)  # noqa: F811
def _label_encode(df, column_names):  # noqa: F811
    le = LabelEncoder()
    check_column(df, column_names=column_names, present=True)
    df[f"{column_names}_enc"] = le.fit_transform(df[column_names])
    return df


@pf.register_dataframe_method
@deprecated_alias(old="old_column_name", new="new_column_name")
def rename_column(
    df: pd.DataFrame, old_column_name: str, new_column_name: str
) -> pd.DataFrame:
    """Rename a column in place.

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
    """Rename columns in place.

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
    """Reorder DataFrame columns by specifying desired order as list of col names.

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
    :raises IndexError: if a column within ``column_order`` is not found
        within the DataFrame.
    """
    check("column_order", column_order, [list, tuple, pd.Index])

    if any(col not in df.columns for col in column_order):
        raise IndexError(
            "A column in ``column_order`` was not found in the DataFrame."
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
    new_column_name: Optional[str] = None,
    delete_columns: bool = True,
) -> pd.DataFrame:
    """Coalesce two or more columns of data in order of column names provided.

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
    :param new_column_name: The new column name after combining.
    :param delete_columns: Whether to delete the columns being coalesced
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
    """Convert Excel's serial date format into Python datetime format.

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
    :param column_name: A column name.
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
    """Convert Matlab's serial date number into Python datetime format.

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
    :param column_name: A column name.
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
    """Convert unix epoch time into Python datetime format.

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
    :param column_name: A column name.
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
    """Fill `NaN` values in specified columns with a given value.

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
    check_column(df, column_names)
    return _fill_empty(df, column_names, value=value)


@dispatch(pd.DataFrame, (list, tuple))
def _fill_empty(df, column_names, value=None):
    """Fill empty function for the case that column_names is list or tuple."""
    fill_mapping = {c: value for c in column_names}
    return df.fillna(value=fill_mapping)


@dispatch(pd.DataFrame, str)  # noqa: F811
def _fill_empty(df, column_names, value=None):  # noqa: F811
    """Fill empty function for the case that column_names is a string."""
    fill_mapping = {column_names: value}
    return df.fillna(value=fill_mapping)


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def expand_column(
    df: pd.DataFrame, column_name: Hashable, sep: str, concat: bool = True
) -> pd.DataFrame:
    """Expand a categorical column with multiple labels into dummy-coded columns.

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
    :param concat: Whether to return the expanded column concatenated to
        the original dataframe (`concat=True`), or to return it standalone
        (`concat=False`).
    :returns: A pandas DataFrame with an expanded column.
    """
    expanded_df = df[column_name].str.get_dummies(sep=sep)
    if concat:
        df = df.join(expanded_df)
        return df
    return expanded_df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def concatenate_columns(
    df: pd.DataFrame,
    column_names: List[Hashable],
    new_column_name,
    sep: str = "-",
) -> pd.DataFrame:
    """Concatenates the set of columns into a single column.

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
    :raises JanitorError: if at least two columns are not provided
        within ``column_names``.
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
    sep: Optional[str] = None,
    new_column_names: Optional[Union[List[str], Tuple[str]]] = None,
    autoname: str = None,
    preserve_position: bool = False,
) -> pd.DataFrame:
    """De-concatenates a single column into multiple columns.

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
    :raises ValueError: if ``column_name`` is not present in the
        DataFrame.
    :raises ValueError: if ``sep`` is not provided and the column values
        are of type ``str``.
    :raises ValueError: if either ``new_column_names`` or ``autoname``
        is not supplied.
    :raises JanitorError: if incorrect number of names is provided
        within ``new_column_names``.
    """

    if column_name not in df.columns:
        raise ValueError(f"column name {column_name} not present in DataFrame")

    if isinstance(df[column_name].iloc[0], str):
        if sep is None:
            raise ValueError(
                "`sep` must be specified if the column values "
                "are of type `str`."
            )
        df_deconcat = df[column_name].str.split(sep, expand=True)
    else:
        df_deconcat = pd.DataFrame(
            df[column_name].to_list(), columns=new_column_names, index=df.index
        )

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
            "to `new_column_names`"
        )

    df_deconcat.columns = new_column_names
    df_new = pd.concat([df, df_deconcat], axis=1)

    if preserve_position:
        df_original = df.copy()
        cols = list(df_original.columns)
        index_original = cols.index(column_name)

        for i, col_new in enumerate(new_column_names):
            cols.insert(index_original + i, col_new)

        df_new = df_new.select_columns(search_column_names=cols).drop(
            columns=column_name
        )

    return df_new


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_string(
    df: pd.DataFrame,
    column_name: Hashable,
    search_string: str,
    complement: bool = False,
) -> pd.DataFrame:
    """Filter a string-based column according to whether it contains a substring.

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
    return df[criteria]


@pf.register_dataframe_method
def filter_on(
    df: pd.DataFrame, criteria: str, complement: bool = False
) -> pd.DataFrame:
    """Return a dataframe filtered on a particular criteria.

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
    return df.query(criteria)


@pf.register_dataframe_method
@deprecated_alias(column="column_name", start="start_date", end="end_date")
def filter_date(
    df: pd.DataFrame,
    column_name: Hashable,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
    years: Optional[List] = None,
    months: Optional[List] = None,
    days: Optional[List] = None,
    column_date_options: Optional[Dict] = None,
    format: Optional[str] = None,  # skipcq: PYL-W0622
) -> pd.DataFrame:
    """Filter a date-based column based on certain criteria.

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

    if start_date and end_date and start_date > end_date:
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
    """Filter a dataframe for values in a column that exist in another iterable.

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
    :raises ValueError: if ``iterable`` does not have a length of ``1``
        or greater.
    """
    if len(iterable) == 0:
        raise ValueError(
            "`iterable` kwarg must be given an iterable of length 1 or greater"
        )
    criteria = df[column_name].isin(iterable)

    if complement:
        return df[~criteria]
    return df[criteria]


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def remove_columns(
    df: pd.DataFrame, column_names: Union[str, Iterable[str], Hashable]
) -> pd.DataFrame:
    """Remove the set of columns specified in `column_names`.

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
    """Change the type of a column.

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
    :raises ValueError: if unknown option provided for
        ``ignore_exception``.
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
    """Add a column to the dataframe.

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

    :param df: A pandas DataFrame.
    :param column_name: Name of the new column. Should be a string, in order
        for the column name to be compatible with the Feather binary
        format (this is a useful thing to have).
    :param value: Either a single value, or a list/tuple of values.
    :param fill_remaining: If value is a tuple or list that is smaller than
        the number of rows in the DataFrame, repeat the list or tuple
        (R-style) to the end of the DataFrame.
    :returns: A pandas DataFrame with an added column.
    :raises ValueError: if attempting to add a column that already exists.
    :raises ValueError: if ``value`` has more elements that number of
        rows in the DataFrame.
    :raises ValueError: if attempting to add an iterable of values with
        a length not equal to the number of DataFrame rows.
    :raises ValueError: if ``value`` has length of ``0``.
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
                "`value` has more elements than number of rows "
                f"in your `DataFrame`. vals: {len(value)}, "
                f"df: {nrows}"
            )
        if len(value) != nrows and not fill_remaining:
            raise ValueError(
                "Attempted to add iterable of values with length"
                " not equal to number of DataFrame rows"
            )

        if len(value) == 0:
            raise ValueError(
                "`value` has to be an iterable of minimum length 1"
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
    """Add multiple columns to the dataframe.

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
    """Truncate column sizes to a specific length.

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
    col_name_count = {}

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
    """Elevates a row to be the column names of a DataFrame.

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
    """Round all values in a column to a fraction.

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
    dest_column_name: Optional[str] = None,
    elementwise: bool = True,
) -> pd.DataFrame:
    """Transform the given column in-place using the provided function.

    Functions can be applied one of two ways:

    - Element-wise (default; ``elementwise=True``)
    - Column-wise  (alternative; ``elementwise=False``)

    If the function is applied "elementwise",
    then the first argument of the function signature
    should be the individual element of each function.
    This is the default behaviour of ``transform_column``,
    because it is easy to understand.
    For example:

    .. code-block:: python

        def elemwise_func(x):
            modified_x = ... # do stuff here
            return modified_x

        df.transform_column(column_name="my_column", function=elementwise_func)

    On the other hand, columnwise application of a function
    behaves as if the function takes in a pandas Series
    and emits back a sequence that is of identical length to the original.
    One place where this is desirable
    is to gain access to `pandas` native string methods,
    which are super fast!

    .. code-block:: python

        def columnwise_func(s: pd.Series) -> pd.Series:
            return s.str[0:5]

        df.transform_column(
            column_name="my_column",
            lambda s: s.str[0:5],
            elementwise=False
        )

    This method does not mutate the original DataFrame.

    Let's say we wanted to apply a log10 transform a column of data.

    Originally one would write code like this:

    .. code-block:: python

        # YOU NO LONGER NEED TO WRITE THIS!
        df[column_name] = df[column_name].apply(np.log10)

    With the method chaining syntax, we can do the following instead:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .transform_column(column_name, np.log10)
        )

    With the functional syntax:

    .. code-block:: python

        df = pd.DataFrame(...)
        df = transform_column(df, column_name, np.log10)

    :param df: A pandas DataFrame.
    :param column_name: The column to transform.
    :param function: A function to apply on the column.
    :param dest_column_name: The column name to store the transformation result
        in. Defaults to None, which will result in the original column
        name being overwritten. If a name is provided here, then a new column
        with the transformed values will be created.
    :param elementwise: Whether to apply the function elementwise or not.
        If elementwise is True, then the function's first argument
        should be the data type of each datum in the column of data,
        and should return a transformed datum.
        If elementwise is False, then the function's should expect
        a pandas Series passed into it, and return a pandas Series.

    :returns: A pandas DataFrame with a transformed column.
    """
    if dest_column_name is None:
        dest_column_name = column_name

    if elementwise:
        result = df[column_name].apply(function)
    else:
        result = function(df[column_name])

    df = df.assign(**{dest_column_name: result})
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_names="new_column_names")
def transform_columns(
    df: pd.DataFrame,
    column_names: Union[List[str], Tuple[str]],
    function: Callable,
    suffix: Optional[str] = None,
    elementwise: bool = True,
    new_column_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Transform multiple columns through the same transformation.

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
    :param elementwise: Passed on to `transform_column`; whether or not
        to apply the transformation function elementwise (True)
        or columnwise (False).
    :param new_column_names: (optional) An explicit mapping of old column names
        to new column names.
    :returns: A pandas DataFrame with transformed columns.
    :raises ValueError: if both ``suffix`` and ``new_column_names`` are
        specified
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
        df = transform_column(
            df, old_col, function, new_col, elementwise=elementwise
        )

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
    """Scales data to between a minimum and maximum value.

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
    :param old_min: (optional) Overrides for the current minimum
        value of the data to be transformed.
    :param old_max: (optional) Overrides for the current maximum
        value of the data to be transformed.
    :param new_min: (optional) The minimum value of the data after
        it has been scaled.
    :param new_max: (optional) The maximum value of the data after
        it has been scaled.
    :param column_name: (optional) The column on which to perform scaling.
    :returns: A pandas DataFrame with scaled data.
    :raises ValueError: if ``old_max`` is not greater than ``old_min``.
    :raises ValueError: if ``new_max`` is not greater than ``new_min``.
    """
    if (
        (old_min is not None)
        and (old_max is not None)
        and (old_max <= old_min)
    ):
        raise ValueError("`old_max` should be greater than `old_min`")

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
    """Flatten multi-level column dataframe to a single level.

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
    cleaning_style: Optional[str] = None,
    cast_non_numeric: Optional[dict] = None,
    fill_all_non_numeric: Optional[Union[float, int]] = None,
    remove_non_numeric: bool = False,
) -> pd.DataFrame:
    """Convert currency column to numeric.

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
    df: pd.DataFrame,
    search_column_names: Union[str, callable, Pattern, slice, list],
    invert: bool = False,
) -> pd.DataFrame:
    """
    Method-chainable selection of columns.

    Not applicable to MultiIndex columns.

    It accepts a string, shell-like glob strings (*string*),
    regex, slice, array-like object, or a list of the previous options.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of columns available as well.

    Examples
    --------

    ::

        import pandas as pd
        import janitor
        import numpy as np
        import datetime
        import re
        from janitor import patterns

        df = pd.DataFrame(
                {
                    "id": [0, 1],
                    "Name": ["ABC", "XYZ"],
                    "code": [1, 2],
                    "code1": [4, np.nan],
                    "code2": ["8", 5],
                    "type": ["S", "R"],
                    "type1": ["E", np.nan],
                    "type2": ["T", "U"],
                    "code3": pd.Series(["a", "b"], dtype="category"),
                    "type3": pd.to_datetime([np.datetime64("2018-01-01"),
                                            datetime.datetime(2018, 1, 1)]),
                }
            )

        df

           id Name  code  code1 code2 type type1 type2 code3    type3
        0   0  ABC     1    4.0     8    S     E     T     a 2018-01-01
        1   1  XYZ     2    NaN     5    R   NaN     U     b 2018-01-01


    - Select by string::

        df.select_columns("id")
           id
       0   0
       1   1

    Select via shell-like glob strings (*) is possible::

        df.select_columns("*type*")

           type type1 type2      type3
        0    S     E     T 2018-01-01
        1    R   NaN     U 2018-01-01

    - Select by slice::

        df.select_columns(slice("code1", "type1"))

           code1 code2 type type1
        0    4.0     8    S     E
        1    NaN     5    R   NaN

    - Select by callable (the callable is applied to every column
      and should return a single ``True`` or ``False`` per column)::

        df.select_columns(pd.api.types.is_datetime64_dtype)

               type3
        0 2018-01-01
        1 2018-01-01

        df.select_columns(lambda x: x.name.startswith("code") or
                                    x.name.endswith("1"))

           code  code1 code2 type1 code3
        0     1    4.0     8     E     a
        1     2    NaN     5   NaN     b

        df.select_columns(lambda x: x.isna().any())

             code1 type1
        0    4.0     E
        1    NaN   NaN

    - Select by regular expression::

        df.select_columns(re.compile("\\d+"))

           code1 code2 type1 type2 code3      type3
        0    4.0     8     E     T     a 2018-01-01
        1    NaN     5   NaN     U     b 2018-01-01

        # same as above, with janitor.patterns
        # simply a wrapper around re.compile

        df.select_columns(patterns("\\d+"))

           code1 code2 type1 type2 code3      type3
        0    4.0     8     E     T     a 2018-01-01
        1    NaN     5   NaN     U     b 2018-01-01

    - Select via a list (you can combine any of the previous options)::

        df.select_columns(["id", "code*", slice("code", "code2")])

           id  code  code1 code2 code3
        0   0     1    4.0     8     a
        1   1     2    NaN     5     b

    - You can also pass a list of booleans::

        df.select_columns([True, False, True, True, True,
                           False, False, False, True, False])

           id  code  code1 code2 code3
        0   0     1    4.0     8     a
        1   1     2    NaN     5     b

    - Setting ``invert`` to ``True``
      returns the complement of the columns provided::

        df.select_columns(["id", "code*", slice("code", "code2")],
                          invert = True)

           Name type type1 type2      type3
        0  ABC    S     E     T 2018-01-01
        1  XYZ    R   NaN     U 2018-01-01

    Functional usage example::

       import pandas as pd
       import janitor as jn

       df = pd.DataFrame(...)

       df = jn.select_columns(['a', 'b', 'col_*'],
                              invert=True)

    Method-chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .select_columns(['a', 'b', 'col_*'],
              invert=True))

    :param df: A pandas DataFrame.
    :param search_column_names: Valid inputs include:

        - an exact column name to look for
        - a shell-style glob string (e.g., `*_thing_*`)
        - a regular expression
        - a callable which is applicable to each Series in the dataframe
        - a list of all the aforementioned options.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the columns
        provided.
    :returns: A pandas DataFrame with the specified columns selected.
    :raises KeyError: if one or more of the specified column names or
        search strings are not found in DataFrame columns.
    :raises ValueError: if the columns is a MultiIndex.

    .. # noqa: DAR402
    """

    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            """
            MultiIndex columns
            not supported for `select_columns`.
            """
        )

    # applicable for any
    # list-like object (ndarray, Series, pd.Index, tuple, ...)
    if is_list_like(search_column_names):
        search_column_names = list(search_column_names)

    full_column_list = _select_columns(search_column_names, df)

    if invert:
        return df.drop(columns=full_column_list)
    return df.loc[:, full_column_list]


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
@deprecated_alias(statistic="statistic_column_name")
def impute(
    df: pd.DataFrame,
    column_name: Hashable,
    value: Optional[Any] = None,
    statistic_column_name: Optional[str] = None,
) -> pd.DataFrame:
    """Method-chainable imputation of values in a column.

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
    :raises ValueError: if both ``value`` and ``statistic`` are provided.
    :raises KeyError: if ``statistic`` is not one of ``mean``, ``average``
        ``median``, ``mode``, ``minimum``, ``min``, ``maximum``, or ``max``.
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
    """Add an arbitrary function to run in the ``pyjanitor`` method chain.

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
def also(df: pd.DataFrame, func: Callable, *args, **kwargs) -> pd.DataFrame:
    """Add an arbitrary function with no return value to run in the
    ``pyjanitor`` method chain. This returns the input dataframe instead,
    not the output of `func`.

    This method does not mutate the original DataFrame.

    Example usage:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .query(...)
            .also(lambda df: print(f"DataFrame shape is: {df.shape}"))
            .transform_column(...)
            .also(lambda df: df.to_csv("midpoint.csv"))
            .also(
                lambda df: print(
                    f"Column col_name has these values: {set(df['col_name'].unique())}"
                )
            )
            .group_add(...)
        )

    :param df: A pandas dataframe.
    :param func: A function you would like to run in the method chain.
        It should take one DataFrame object as a parameter and have no return.
        If there is a return, it will be ignored.
    :param args: Optional arguments for ``func``.
    :param kwargs: Optional keyword arguments for ``func``.
    :returns: The input pandas DataFrame.
    """  # noqa: E501
    func(df.copy(), *args, **kwargs)
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def dropnotnull(df: pd.DataFrame, column_name: Hashable) -> pd.DataFrame:
    """Drop rows that do not have null values in the given column.

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
def find_replace(
    df: pd.DataFrame, match: str = "exact", **mappings
) -> pd.DataFrame:
    """Perform a find-and-replace action on provided columns.

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

    Example 1 for exact matching

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
    :returns: A pandas DataFrame with replaced values.
    """  # noqa: E501
    for column_name, mapper in mappings.items():
        df = _find_replace(df, column_name, mapper, match=match)
    return df


def _find_replace(
    df: pd.DataFrame, column_name: str, mapper: Dict, match: str = "exact"
) -> pd.DataFrame:
    """Utility function for ``find_replace``.

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
    :raises ValueError: is trying to use null replacement. Kindly use
        ``.fillna()`` instead.
    :raises ValueError: if ``match`` is not one of 'exact' or 'regex'.
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
            df.loc[condition, column_name] = v
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
                condition=("a > 2 and b < 8",
                target_column_name='c',
                target_val=10)
            )
        # a b  c
        # 1 5  0
        # 2 6  0
        # 3 7 10
        # 4 8  0

    :param df: The pandas DataFrame object.
    :param conditions: Conditions used to update a target column
        and target value.
    :param target_column_name: Column to be updated. If column does not exist
        in dataframe, a new column will be created; note that entries that do
        not get set in the new column will be null.
    :param target_val: Value to be updated
    :returns: An updated pandas DataFrame.
    :raises IndexError: if ``conditions`` does not have the same length as
        ``df``.
    :raises TypeError: if ``conditions`` is not a pandas-compatible string
        query.

    .. # noqa: DAR402
    """

    # use query mode if a string expression is passed
    if isinstance(conditions, str):
        conditions_index = df.query(conditions).index
    else:
        conditions_index = df.loc[conditions].index
    df.loc[conditions_index, target_column_name] = target_val

    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def to_datetime(
    df: pd.DataFrame, column_name: Hashable, **kwargs
) -> pd.DataFrame:
    """Method-chainable to_datetime.

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
    """Shortcut for assigning a groupby-transform to a new column.

    This method does not mutate the original DataFrame.

    Without this function, we would have to write a verbose line:

    .. code-block:: python

        df = df.assign(...=df.groupby(...)[...].transform(...))

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
    :returns: A pandas DataFrame.
    """
    df = df.copy()
    # convert to list
    # needed when creating a mapping through the iteration
    if isinstance(by, str):
        by = [by]
    # this is a temporary measure, till the minimum Pandas version is 1.1,
    # which supports null values in the group by
    # If any of the grouping columns has null values, we temporarily
    # replace the values with some outrageous value, that should not exist
    # in the column. Also, the hasnans property is significantly faster than
    # .isnull().any()
    if any(df[col].hasnans for col in by):

        mapping = {
            column: ".*^%s1ho1go1logoban?*&-|/\\gos1he()#_" for column in by
        }

        df[new_column_name] = (
            df.fillna(mapping).groupby(by)[agg_column_name].transform(agg)
        )

    else:
        df[new_column_name] = df.groupby(by)[agg_column_name].transform(agg)
    return df


@pf.register_dataframe_accessor("data_description")
class DataDescription:
    """High-level description of data present in this DataFrame.

    This is a custom data accessor.
    """

    def __init__(self, data):
        """Initialize DataDescription class."""
        self._data = data
        self._desc = {}

    def _get_data_df(self) -> pd.DataFrame:
        df = self._data

        data_dict = {}
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
        """Update the description for each of the columns in the DataFrame.

        :param desc: The structure containing the descriptions to update
        :raises ValueError: if length of description list does not match
            number of columns in DataFrame.
        """
        if isinstance(desc, list):
            if len(desc) != len(self._data.columns):
                raise ValueError(
                    "Length of description list "
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
    labels: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a new column that labels bins for a specified numeric column.

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
    :raises ValueError: if number of labels do not match number of bins.
    """
    if not labels:
        df[str(to_column_name)] = pd.cut(
            df[str(from_column_name)], bins=num_bins
        )
    else:
        if not len(labels) == num_bins:
            raise ValueError("Number of labels must match number of bins.")

        df[str(to_column_name)] = pd.cut(
            df[str(from_column_name)], bins=num_bins, labels=labels
        )

    return df


@pf.register_dataframe_method
def drop_duplicate_columns(
    df: pd.DataFrame, column_name: Hashable, nth_index: int = 0
) -> pd.DataFrame:
    """Remove a duplicated column specified by column_name, its index.

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
    """Take the first row within each group specified by `subset`.

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
    """Shuffle the rows of the DataFrame.

    This method does not mutate the original DataFrame.

    Super-sugary syntax! Underneath the hood, we use ``df.sample(frac=1)``,
    with the option to set the random state.

    Example usage:

    .. code-block:: python

        df = pd.DataFrame(...).shuffle()

    :param df: A pandas DataFrame
    :param random_state: (optional) A seed for the random number generator.
    :param reset_index: (optional) Resets index to default integers
    :returns: A shuffled pandas DataFrame.
    """
    result = df.sample(frac=1, random_state=random_state)
    if reset_index:
        result = result.reset_index(drop=True)
    return result


@pf.register_dataframe_method
def join_apply(
    df: pd.DataFrame, func: Callable, new_column_name: str
) -> pd.DataFrame:
    """Join the result of applying a function across dataframe rows.

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
    :returns: A pandas DataFrame with new column appended.
    """
    df = df.copy().join(df.apply(func, axis=1).rename(new_column_name))
    return df


@pf.register_dataframe_method
def flag_nulls(
    df: pd.DataFrame,
    column_name: Optional[Hashable] = "null_flag",
    columns: Optional[Union[str, Iterable[str], Hashable]] = None,
) -> pd.DataFrame:
    """Creates a new column to indicate whether you have null values in a given
    row. If the columns parameter is not set, looks across the entire
    DataFrame, otherwise will look only in the columns you set.

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(
            {'a': [1, 2, None, 4],
             'b': [5.0, None, 7.0, 8.0]})

        df.flag_nulls()
        #  'a' | 'b'  | 'null_flag'
        #   1  | 5.0  |   0
        #   2  | NaN  |   1
        #  NaN | 7.0  |   1
        #   4  | 8.0  |   0

        jn.functions.flag_nulls(df)
        #  'a' | 'b'  | 'null_flag'
        #   1  | 5.0  |   0
        #   2  | NaN  |   1
        #  NaN | 7.0  |   1
        #   4  | 8.0  |   0

        df.flag_nulls(columns=['b'])
        #  'a' | 'b'  | 'null_flag'
        #   1  | 5.0  |   0
        #   2  | NaN  |   1
        #  NaN | 7.0  |   0
        #   4  | 8.0  |   0


    :param df: Input Pandas dataframe.
    :param column_name: Name for the output column. Defaults to 'null_flag'.
    :param columns: List of columns to look at for finding null values. If you
        only want to look at one column, you can simply give its name. If set
        to None (default), all DataFrame columns are used.
    :returns: Input dataframe with the null flag column.
    :raises ValueError: if ``column_name`` is already present in the
        DataFrame.
    :raises ValueError: if a column within ``columns`` is no present in
        the DataFrame.

    .. # noqa: DAR402
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
    """Generates a running total of cumulative unique values in a given column.

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
    otherwise, the case of any letters will not matter.

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
    """Return a set of the values.

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
    """Adds Gaussian noise (jitter) to the values of a column.

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
    :param random_state: An integer or 1-d array value used to set the random
        seed, default to None.

    :returns: A pandas DataFrame with a new column containing Gaussian-
        jittered values from another column.
    :raises TypeError: if ``column_name`` is not numeric.
    :raises ValueError: if ``scale`` is not a numerical value
        greater than ``0``.
    :raises ValueError: if ``clip`` is not an iterable of length ``2``.
    :raises ValueError: if ``clip[0]`` is not less than ``clip[1]``.
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
    """Sort a DataFrame by a column using "natural" sorting.

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
    :returns: A sorted pandas DataFrame.
    """
    new_order = index_natsorted(df[column_name], **natsorted_kwargs)
    return df.iloc[new_order, :]


@pf.register_dataframe_method
def expand_grid(
    df: Optional[pd.DataFrame] = None,
    df_key: Optional[str] = None,
    others: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Creates a dataframe from a cartesian combination of all inputs.

    This works with a dictionary of name value pairs,
    or keyword arguments (`kwargs`);
    it is also not restricted to dataframes;
    it can work with any list-like structure
    that is 1 or 2 dimensional.
    MultiIndex objects are not supported though.

    If method-chaining to a dataframe,
    a key to represent the column name in the output must be provided.

    Note that if a MultiIndex dataframe or series is passed, the index/columns
    will be discarded, and a single indexed dataframe will be returned.

    Existing data types are preserved in this function.
    This includes Pandas' extension array dtypes.

    The output will always be a dataframe.

    Example:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame({"x":range(1,3), "y":[2,1]})
        others = {"z" : range(1,4)}

        df.expand_grid(df_key="df",others=others)

        # df_x |   df_y |   z
        #    1 |      2 |   1
        #    1 |      2 |   2
        #    1 |      2 |   3
        #    2 |      1 |   1
        #    2 |      1 |   2
        #    2 |      1 |   3

        # create a dataframe from all combinations in a dictionary
        data = {"x":range(1,4), "y":[1,2]}

        jn.expand_grid(others=data)

        #  x |   y
        #  1 |   1
        #  1 |   2
        #  2 |   1
        #  2 |   2
        #  3 |   1
        #  3 |   2


    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.expand_grid(df=df, df_key="...", others={...})

    Method-chaining usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...).expand_grid(df_key="bla",others={...})

    Usage independent of a dataframe

    .. code-block:: python

        import pandas as pd
        from janitor import expand_grid

        df = expand_grid({"x":range(1,4), "y":[1,2]})

    :param df: A pandas dataframe.
    :param df_key: name of key for the dataframe.
        It becomes part of the column names of the dataframe.
    :param others: A dictionary that contains the data
        to be combined with the dataframe.
        If no dataframe exists, all inputs
        in others will be combined to create a dataframe.
    :param kwargs: Keyword arguments are accepted.
    :returns: A pandas dataframe of all combinations of name value pairs.
    :raises TypeError: if `others` is not a dictionary
    :raises KeyError: if there is a dataframe and no key is provided.
    :raises ValueError: if `others` is empty.

    .. # noqa: DAR402

    """

    check("others", others, [dict])

    others = {**others, **kwargs}

    # if there is a dataframe, for the method chaining,
    # it must have a key, to create a name value pair
    if df is not None:
        df = df.copy()

        if not df_key:
            raise KeyError(
                """
                Using `expand_grid` as part of a DataFrame method chain
                requires that a string `df_key` be passed in.
                """
            )

        check("df_key", df_key, [str])

        others = {**{df_key: df}, **others}

    if not others:
        raise ValueError("""`others` cannot be empty.""")

    return _computations_expand_grid(others)


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def process_text(
    df: pd.DataFrame,
    column_name: str,
    new_column_names: Optional[Union[str, list]] = None,
    merge_frame: Optional[bool] = False,
    string_function: Optional[str] = None,
    **kwargs: str,
) -> pd.DataFrame:
    """
    Apply a Pandas string method to an existing column and return a dataframe.

    This function aims to make string cleaning easy, while chaining,
    by simply passing the string method name to the ``process_text`` function.
    This modifies an existing column and can also be used to create a new
    column.

    .. note:: In versions < 0.20.11, this function did not support creation of
        new columns.

    A list of all the string methods in Pandas can be accessed `here
    <https://pandas.pydata.org/docs/user_guide/text.html#method-summary>`__.

    Example:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame({"text" : ["Ragnar",
                                    "sammywemmy",
                                    "ginger"],
                           "code" : [1, 2, 3]})

        df.process_text(column_name = "text",
                        string_function = "lower")

          text          code
        0 ragnar         1
        1 sammywemmy     2
        2 ginger         3

    For string methods with parameters, simply pass the keyword arguments::

        df.process_text(
            column_name = "text",
            string_function = "extract",
            pat = r"(ag)",
            expand = False,
            flags = re.IGNORECASE
            )

          text     code
        0 ag        1
        1 NaN       2
        2 NaN       3

    A new column can be created, leaving the existing column unmodified::

        df.process_text(
            column_name = "text",
            new_column_names = "new_text",
            string_function = "extract",
            pat = r"(ag)",
            flags = re.IGNORECASE
            )

          text           code     new_text
        0 Ragnar          1          ag
        1 sammywemmy      2          NaN
        2 ginger          3          NaN


    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.process_text(
            df = df,
            column_name,
            new_column_names = None/string/list_of_strings,
            merge_frame = True/False,
            string_function = "string_func_name_here",
            kwargs
            )

    Method-chaining usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = (
            pd.DataFrame(...)
            .process_text(
                column_name,
                new_column_names = None/string/list_of_strings,
                merge_frame = True/False
                string_function = "string_func_name_here",
                kwargs
                )
        )


    :param df: A pandas dataframe.
    :param column_name: String column to be operated on.
    :param new_column_names: Name(s) to assign to the new column(s) created
        from the text processing. `new_column_names` can be a string, if
        the result of the text processing is a Series or string; if the
        result of the text processing is a dataframe, then `new_column_names`
        is treated as a prefix for each of the columns in the new dataframe.
        `new_column_names` can also be a list of strings to act as new
        column names for the new dataframe. The existing `column_name`
        stays unmodified if `new_column_names` is not None.
    :param merge_frame: This comes into play if the result of the text
        processing is a dataframe. If `True`, the resulting dataframe
        will be merged with the original dataframe, else the resulting
        dataframe, not the original dataframe, will be returned.
    :param string_function: Pandas string method to be applied.
    :param kwargs: Keyword arguments for parameters of the `string_function`.
    :returns: A pandas dataframe with modified column(s).
    :raises KeyError: if ``string_function`` is not a Pandas string method.
    :raises TypeError: if wrong ``arg`` or ``kwarg`` is supplied.
    :raises ValueError: if `column_name` not found in dataframe.
    :raises ValueError: if `new_column_names` is not None and is found in
        dataframe.

    .. # noqa: DAR402
    """
    df = df.copy()

    check("column_name", column_name, [str])
    check_column(df, [column_name])

    # new_column_names should not already exist in the dataframe
    if new_column_names:
        check("new_column_names", new_column_names, [list, str])
        if isinstance(new_column_names, str):
            check_column(df, [new_column_names], present=False)
        else:
            check_column(df, new_column_names, present=False)

    if merge_frame:
        check("merge_frame", merge_frame, [bool])

    pandas_string_methods = [
        func.__name__
        for _, func in inspect.getmembers(pd.Series.str, inspect.isfunction)
        if not func.__name__.startswith("_")
    ]

    if not string_function:
        return df

    if string_function not in pandas_string_methods:
        raise KeyError(f"{string_function} is not a Pandas string method.")

    if string_function == "extractall" and merge_frame:
        # create unique indices
        # comes in handy for executing joins if there are
        # duplicated indices in the original dataframe
        df = df.set_index(np.arange(len(df)), append=True)  # extra_index_line

    result = getattr(df.loc[:, column_name].str, string_function)(**kwargs)

    # TODO: Support for str.cat with `join` parameter
    # need a robust way to handle the results
    # if there is a `join` parameter, as this could create more
    # or less rows with varying indices or even duplicate indices

    return _process_text(
        result,
        df=df,
        column_name=column_name,
        new_column_names=new_column_names,
        merge_frame=merge_frame,
    )


@pf.register_dataframe_method
def fill_direction(
    df: pd.DataFrame,
    directions: Dict[Hashable, str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Provide a method-chainable function for filling missing values
    in selected columns.

    Missing values are filled using the next or previous entry.
    The columns are paired with the directions in a dictionary.
    It is a wrapper for ``pd.Series.ffill`` and ``pd.Series.bfill``.

    .. code-block:: python

        import pandas as pd
        import numpy as np
        import janitor as jn

        df = pd.DataFrame({"text": ["ragnar", np.nan, "sammywemmy",
                                    np.nan, "ginger"],
                           "code" : [np.nan, 2, 3, np.nan, 5]})

        df

           text          code
        0 ragnar         NaN
        1 NaN            2.0
        2 sammywemmy     3.0
        3 NaN            NaN
        4 ginger         5.0



    Fill on a single column::

        df.fill_direction({"text" : "up"})

           text          code
        0 ragnar         NaN
        1 sammywemmy     2.0
        2 sammywemmy     3.0
        3 ginger         NaN
        4 ginger         5.0

    Fill on multiple columns::

        df.fill_direction({"text" : "down", "code" : "down"})

           text          code
        0 ragnar         NaN
        1 ragnar         2.0
        2 sammywemmy     3.0
        3 sammywemmy     3.0
        4 ginger         5.0

    Fill multiple columns in different directions::

        df.fill_direction({"text" : "up", "code" : "down"})

           text          code
        0 ragnar         NaN
        1 sammywemmy     2.0
        2 sammywemmy     3.0
        3 ginger         3.0
        4 ginger         5.0

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.fill_direction(
            df = df,
            directions = {column_1 : direction_1,
                          column_2 : direction_2,
                          ...},
            limit = None # limit must be None or greater than 0
            )

    Method-chaining usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = (
            pd.DataFrame(...)
            .fill_direction(
            directions = {column_1 : direction_1,
                          column_2 : direction_2,
                          ...},
            limit = None # limit must be None or greater than 0
            )
        )

    :param df: A pandas dataframe.
    :param directions: Key - value pairs of columns and directions. Directions
        can be either `down` (default), `up`, `updown` (fill up then down) and
        `downup` (fill down then up).
    :param limit: number of consecutive null values to forward/backward fill.
        Value must `None` or greater than 0.
    :returns: A pandas dataframe with modified column(s).
    :raises ValueError: if column supplied is not in the dataframe.
    :raises ValueError: if direction supplied is not one of `down`, `up`,
        `updown`, or `downup`.

    .. # noqa: DAR402
    """
    df = df.copy()
    if not directions:
        return df

    check("directions", directions, [dict])

    if limit is not None:
        check("limit", limit, [int])
        # pandas raises error if limit is not greater than zero
        # so no need for a check on pyjanitor's end

    check_column(df, directions)

    for _, direction in directions.items():
        if direction not in {"up", "down", "updown", "downup"}:
            raise ValueError(
                """
                The direction should be a string and should be one of
                `up`, `down`, `updown`, or `downup`.
                """
            )

    # TODO: option to specify limit per column; current implementation
    # is one `limit` for all the columns. Might need refactoring, or an
    # API change.
    for column, direction in directions.items():
        if direction == "up":
            df.loc[:, column] = df.loc[:, column].bfill(limit=limit)
        elif direction == "down":
            df.loc[:, column] = df.loc[:, column].ffill(limit=limit)
        elif direction == "updown":
            df.loc[:, column] = (
                df.loc[:, column].bfill(limit=limit).ffill(limit=limit)
            )
        else:  # downup
            df.loc[:, column] = (
                df.loc[:, column].ffill(limit=limit).bfill(limit=limit)
            )
    return df


@pf.register_dataframe_method
def groupby_topk(
    df: pd.DataFrame,
    groupby_column_name: Hashable,
    sort_column_name: Hashable,
    k: int,
    sort_values_kwargs: Dict = None,
) -> pd.DataFrame:
    """
    Return top `k` rows from a groupby of a set of columns.

    Returns a dataframe that has the top `k` values grouped by `groupby_column_name`
    and sorted by `sort_column_name`.
    Additional parameters to the sorting (such as ascending=True)
    can be passed using `sort_values_kwargs`.

    List of all sort_values() parameters can be found here_.

    .. _here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html


    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame({'age' : [20, 22, 24, 23, 21, 22],
                           'ID' : [1,2,3,4,5,6],
                           'result' : ["pass", "fail", "pass",
                                       "pass", "fail", "pass"]})

        # Ascending top 3:
        df.groupby_topk('result', 'age', 3)
        #       age  ID  result
        #result
        #fail   21   5   fail
        #       22   2   fail
        #pass   20   1   pass
        #       22   6   pass
        #       23   4   pass

        #Descending top 2:
        df.groupby_topk('result', 'age', 2, {'ascending':False})
        #       age  ID result
        #result
        #fail   22   2   fail
        #       21   5   fail
        #pass   24   3   pass
        #       23   4   pass

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.groupby_topk(
            df = df,
            groupby_column_name = 'groupby_column',
            sort_column_name = 'sort_column',
            k = 5
            )

    Method-chaining usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = (
            pd.DataFrame(...)
            .groupby_topk(
            df = df,
            groupby_column_name = 'groupby_column',
            sort_column_name = 'sort_column',
            k = 5
            )
        )

    :param df: A pandas dataframe.
    :param groupby_column_name: Column name to group input dataframe `df` by.
    :param sort_column_name: Name of the column to sort along the
        input dataframe `df`.
    :param k: Number of top rows to return from each group after sorting.
    :param sort_values_kwargs: Arguments to be passed to sort_values function.
    :returns: A pandas dataframe with top `k` rows that are grouped by
        `groupby_column_name` column with each group sorted along the
        column `sort_column_name`.
    :raises ValueError: if `k` is less than 1.
    :raises ValueError: if `groupby_column_name` not in dataframe `df`.
    :raises ValueError: if `sort_column_name` not in dataframe `df`.
    :raises KeyError: if `inplace:True` is present in `sort_values_kwargs`.
    """  # noqa: E501

    # Convert the default sort_values_kwargs from None to empty Dict
    sort_values_kwargs = sort_values_kwargs or {}

    # Check if groupby_column_name and sort_column_name exists in the dataframe
    check_column(df, [groupby_column_name, sort_column_name])

    # Check if k is greater than 0.
    if k < 1:
        raise ValueError(
            "Numbers of rows per group to be returned must be greater than 0."
        )

    # Check if inplace:True in sort values kwargs because it returns None
    if (
        "inplace" in sort_values_kwargs.keys()
        and sort_values_kwargs["inplace"]
    ):
        raise KeyError("Cannot use `inplace=True` in `sort_values_kwargs`.")

    return df.groupby(groupby_column_name).apply(
        lambda d: d.sort_values(sort_column_name, **sort_values_kwargs).head(k)
    )


@pf.register_dataframe_method
def complete(
    df: pd.DataFrame,
    columns: List[Union[List, Tuple, Dict, str]] = None,
    fill_value: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    This function turns implicit missing values into explicit missing values.

    It is modeled after tidyr's `complete` function, and is a wrapper around
    `pd.DataFrame.merge` and `pd.DataFrame.fillna`.

    Combinations of column names or a list/tuple of column names, or even a
    dictionary of column names and new values are possible.

    It can also handle duplicated data.

    `Source <https://tidyr.tidyverse.org/reference/complete.html#examples>`_

    .. code-block:: python

        import pandas as pd
        import janitor as jn

            group	item_id	    item_name	value1	value2
        0	1	    1	        a	1	4
        1	2	    2	        b	2	5
        2	1	    2	        b	3	6

    To find all the unique combinations of `group`, `item_id`, and `item_name`,
    including combinations not present in the data, each variable should be
    passed in a list to the `columns` parameter::

        df.complete(columns = ['group', 'item_id', 'item_name'])

              group	item_id	    item_name	value1	value2
        0	1	    1	        a	1.0	4.0
        1	1	    1	        b	NaN	NaN
        2	1	    2	        a	NaN	NaN
        3	1	    2	        b	3.0	6.0
        4	2	    1	        a	NaN	NaN
        5	2	    1	        b	NaN	NaN
        6	2	    2	        a	NaN	NaN
        7	2	    2	        b	2.0	5.0

    To expose just the missing values based only on the existing data,
    `item_id` and `item_name` can be wrapped in a tuple, while `group`
    is passed in as a separate variable::

        df.complete(columns = ["group", ("item_id", "item_name")])
            group	item_id	    item_name	value1	   value2
        0	1	    1	        a	  1.0	    4.0
        1	1	    2	        b	  3.0	    6.0
        2	2	    1	        a	  NaN 	    NaN
        3	2	    2	        b	  2.0	    5.0

    Let's look at another example:

    `Source Data <http://imachordata.com/2016/02/05/you-complete-me/>`_

    .. code-block:: python

            Year      Taxon         Abundance
        0   1999    Saccharina         4
        1   2000    Saccharina         5
        2   2004    Saccharina         2
        3   1999     Agarum            1
        4   2004     Agarum            8

    Note that Year 2000 and Agarum pairing is missing. Let's make it
    explicit::

        df.complete(columns = ['Year', 'Taxon'])

           Year      Taxon     Abundance
        0  1999     Agarum         1.0
        1  1999     Saccharina     4.0
        2  2000     Agarum         NaN
        3  2000     Saccharina     5.0
        4  2004     Agarum         8.0
        5  2004     Saccharina     2.0

    The null value can be replaced with the fill_value argument::

        df.complete(columns = ['Year', 'Taxon'],
                    fill_value = {"Abundance" : 0})

           Year      Taxon     Abundance
        0  1999     Agarum         1.0
        1  1999     Saccharina     4.0
        2  2000     Agarum         0.0
        3  2000     Saccharina     5.0
        4  2004     Agarum         8.0
        5  2004     Saccharina     2.0

    What if we wanted the explicit missing values for all the years from
    1999 to 2004? Easy - simply pass a dictionary pairing the column name
    with the new values::

        df.complete(columns = [{"Year": lambda x : range(x.Year.min(),
                                                         x.Year.max() + 1)},
                                "Taxon"],
                    fill_value={"Abundance" : 0})

            Year      Taxon     Abundance
        0   1999     Agarum         1.0
        1   1999    Saccharina      4.0
        2   2000     Agarum         0.0
        3   2000    Saccharina      5.0
        4   2001     Agarum         0.0
        5   2001    Saccharina      0.0
        6   2002     Agarum         0.0
        7   2002    Saccharina      0.0
        8   2003     Agarum         0.0
        9   2003     Saccharina     0.0
        10  2004     Agarum         8.0
        11  2004    Saccharina      2.0

    .. note:: MultiIndex columns are not supported.

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.complete(
            df = df,
            columns= [
                column_label,
                (column1, column2, ...),
                {column1: new_values, ...}
            ],
            fill_value = None
        )

    Method chaining syntax:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .complete(columns=[
                column_label,
                (column1, column2, ...),
                {column1: new_values, ...},
            ],
            fill_value=None,
        )

    :param df: A pandas dataframe.
    :param columns: This is a list containing the columns to be
        completed. It could be column labels (string type),
        a list/tuple of column labels, or a dictionary that pairs
        column labels with new values.
    :param fill_value: Dictionary pairing the columns with the null
        replacement value.
    :returns: A pandas dataframe with modified column(s).
    :raises TypeError: if `columns` is not a list.
    :raises TypeError: if `fill_value` is not a dictionary.
    :raises ValueError: if entry in `columns` is not a
        str/dict/list/tuple.
    :raises ValueError: if entry in `columns` is a dict/list/tuple
        and is empty.

    .. # noqa: DAR402
    """

    df = df.copy()

    df = _computations_complete(df, columns, fill_value)

    return df


def patterns(regex_pattern: Union[str, Pattern]) -> Pattern:
    """
    This function converts a string into a compiled regular expression;
    it can be used to select columns in the index or columns_names
    arguments of ``pivot_longer`` function.

    :param regex_pattern: string to be converted to compiled regular
        expression.
    :returns: A compile regular expression from provided
        ``regex_pattern``.
    """
    check("regular expression", regex_pattern, [str, Pattern])

    return re.compile(regex_pattern)


@pf.register_dataframe_method
def pivot_longer(
    df: pd.DataFrame,
    index: Optional[Union[List, Tuple, str, Pattern]] = None,
    column_names: Optional[Union[List, Tuple, str, Pattern]] = None,
    names_to: Optional[Union[List, Tuple, str]] = "variable",
    values_to: Optional[str] = "value",
    column_level: Optional[Union[int, str]] = None,
    names_sep: Optional[Union[str, Pattern]] = None,
    names_pattern: Optional[Union[List, Tuple, str, Pattern]] = None,
    sort_by_appearance: Optional[bool] = False,
    ignore_index: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Unpivots a DataFrame from 'wide' to 'long' format.

    This method does not mutate the original DataFrame.

    It is a wrapper around `pd.melt` and is meant to serve as a single point
    for transformations that require `pd.melt` or `pd.wide_to_long`.

    It is modeled after the `pivot_longer` function in R's tidyr package, and
    offers more functionality and flexibility than `pd.wide_to_long`.

    This function is useful to massage a DataFrame into a format where
    one or more columns are considered measured variables, and all other
    columns are considered as identifier variables.

    All measured variables are “unpivoted” (and typically duplicated) along the
    row axis.

    See the `Example notebooks <https://pyjanitor.readthedocs.io/notebooks/>`_
    for a more in depth exploration of `pivot_longer`.


    Example 1: The following DataFrame contains heartrate data for patients
    treated with two different drugs, 'a' and 'b'.

    .. code-block:: python

              name   a   b
        0   Wilbur  67  56
        1  Petunia  80  90
        2  Gregory  64  50

    The column names 'a' and 'b' are actually the names of a measured variable
    (i.e. the name of a drug), but the values are a different measured variable
    (heartrate). We would like to unpivot these 'a' and 'b' columns into a
    'drug' column and a 'heartrate' column.

    .. code-block:: python

        df = (pd.DataFrame(...)
              .pivot_longer(
                  column_names = ['a', 'b'],
                  names_to = 'drug',
                  values_to = 'heartrate',
                  sort_by_appearance = True
                  )
            )

              name drug  heartrate
        0   Wilbur    a         67
        1   Wilbur    b         56
        2  Petunia    a         80
        3  Petunia    b         90
        4  Gregory    a         64
        5  Gregory    b         50

    Note how the data is stacked in order of first appearance. If, however,
    you do not care for order of appearance, and want to wring out some
    more performance, you can set `sort_by_appearance` to ``False`` (the
    default is ``False``).

    .. code-block:: python

        df = (pd.DataFrame(...)
              .pivot_longer(
                  column_names = ['a', 'b'],
                  names_to = 'drug',
                  values_to = 'heartrate',
                  sort_by_appearance = False
                  )
            )

                name     drug  heartrate
        0	Wilbur	   a	67
        1	Petunia	   a	80
        2	Gregory    a	64
        3	Wilbur	   b	56
        4	Petunia	   b	90
        5	Gregory	   b	50

    You can set `ignore_index` to ``False``, if you wish to reuse the index
    from the source dataframe (the index will be repeated as many times as
    necessary):

    .. code-block:: python

        df = (pd.DataFrame(...)
              .pivot_longer(
                  column_names = ['a', 'b'],
                  names_to = 'drug',
                  values_to = 'heartrate',
                  sort_by_appearance = False,
                  ignore_index = False
                  )
            )

                name     drug  heartrate
        0	Wilbur	   a	67
        1	Petunia	   a	80
        2	Gregory    a	64
        0	Wilbur	   b	56
        1	Petunia	   b	90
        2	Gregory	   b	50

    MultiIndex dataframes are unpivoted in the same form that you would
    expect from pandas' `melt`:

    .. code-block:: python

            A  B  C
            D  E  F
        0   a  1  2
        1   b  3  4
        2   c  5  6

        df = (pd.DataFrame(...)
               .pivot_longer(
                   index = [("A", "D")],
                   names_to = ["first", "second"]
                   )
            )

             (A, D)  first   second   value
        0	a	B	E	1
        1	b	B	E	3
        2	c	B	E	5
        3	a	C	F	2
        4	b	C	F	4
        5	c	C	F	6

    You can also unpivot on a specific level:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .pivot_longer(
                  index = "A",
                  names_to = "first",
                  column_level = 0
                  )
            )

           A      first  value
        0  a        B      1
        1  b        B      3
        2  c        B      5


    Example 2: The dataframe below has year and month variables embedded within
    the column names.

    .. code-block:: python

              col1	    2019-12	 2020-01	 2020-02
        0	a	   -1.085631	-1.506295	-2.426679
        1	b	    0.997345	-0.578600	-0.428913
        2	c	    0.282978	 1.651437	 1.265936

    Pivot_longer can conveniently reshape the dataframe into long format, with
    new columns for the year and month. You simply pass in the new column names
    to `names_to`, and pass the hyphen '-' to the `names_sep` argument.

    .. code-block:: python

        df = (pd.DataFrame(...)
              .pivot_longer(
                  index = 'col1',
                  names_to = ('year', 'month'),
                  names_sep = '-',
                  sort_by_appearance = True
                  )
            )

           col1 year   month      value
        0    a  2019     12     -1.085631
        1    a  2020     01     -1.506295
        2    a  2020     02     -2.426679
        3    b  2019     12      0.997345
        4    b  2020     01     -0.578600
        5    b  2020     02     -0.428913
        6    c  2019     12      0.282978
        7    c  2020     01      1.651437
        8    c  2020     02      1.265936

    Example 3: The dataframe below has names embedded in it
    (‘measure1’, ‘measure2’) that we would love to reuse as column names.

    .. code-block:: python

            treat1-measure1     treat1-measure2 treat2-measure1 treat2-measure2
        0                1              4                   2               5
        1                2              5                   3               4

    For this, we use the `.value` variable, which signals to `pivot_longer`
    to treat the part of the column names corresponding to `.value` as new
    column names. The `.value` variable is similar to `stubnames` in pandas'
    ``wide_to_long`` function, but with more flexibility.

    .. code-block:: python

        df = (pd.DataFrame(...)
              .pivot_longer(
                  names_to = ("group", '.value'),
                  names_sep = '-',
                  sort_by_appearance = True
                  )
            )

            group  measure1  measure2
        0  treat1         1         4
        1  treat2         2         5
        2  treat1         2         5
        3  treat2         3         4

    Let's break down the `.value` idea. When `.value` is used, `pivot_longer`
    creates a pairing. In the example above, we get a pairing
    ``{"group":["treat1", "treat2"], ".value":["measure1", "measure2"]}``. All
    the values associated with `.value` become new column names, while those
    not associated with `.value`(`treat1` and `treat2`) become values in a
    new column `group`. `values_to` is overridden during this process.

    .. note:: The values not associated with ".value" (in the example above,
        this is the `group` column) are returned as object dtypes. You can
        change it to your preferred dtype using pandas' ``astype`` method.

    Example 4: You can also unpivot from wide to long using regular expressions

    .. code-block:: python

            n_1  n_2  n_3  pct_1  pct_2  pct_3
        0   10   20   30   0.1    0.2    0.3

        df = (pd.DataFrame(...)
              .pivot_longer(
                  names_to = (".value", "name"),
                  names_pattern = "(.*)_(.)"
                  )
            )

            name    n  pct
        0     1  10.0  0.1
        1     2  20.0  0.2
        2     3  30.0  0.3

    The same idea of `.value` works here as well. Based on the capturing groups
    in the regex in `names_pattern`, we have two pairings -->
    ``{".value":["n", "pct"], "name":[1,2,3]}``. Just like in the previous
    example, the values associated with `.value` become new column names,
    while those not associated with `.value` become values in the new column
    ``name``.

    Note that there are no limits to the pairing; however, you can only have
    one `.value` in ``names_to``.

    Example 5: You can also pass a list/tuple of regular expressions that match
    specific patterns to ``names_pattern``, along with a list/tuple of new
    names to ``names_to``; this can come in handy if `.value` falls short:

    .. code-block:: python

          GameID   Date	        Visitor	     Score_V	   Home	        Score_H
        0  1     9/10/2020   Houston Texans     20    Kansas City Chiefs   34
        1  2     9/13/2020   Seattle Seahawks   38    Atlanta Falcons      25

    .. code-block:: python

        df = (pd.DataFrame(...)
              .pivot_longer(
                    ['GameID','Date'],
                    names_to=("Team","Score"),
                    names_pattern=("^Visitor|Home", "^Score"))
            )

               GameID       Date              Team       Score
        0       1       9/10/2020      Houston Texans     20
        1       2       9/13/2020    Seattle Seahawks     38
        2       1       9/10/2020  Kansas City Chiefs     34
        3       2       9/13/2020     Atlanta Falcons     25

    Note that in the code above, the number of entries in both `names_to` and
    `names_pattern` must match. Essentially, what the code does is look for
    columns that start with `Visitor` or `Home` (using the regex supplied) and
    puts all the values associated with these columns under a new column name
    `Team`. It then looks for columns that start with `Score` and collate all
    the values associated with these columns to a single column named `Score`.

    You can also take advantage of `janitor.patterns` function, which allows
    selection of columns via a regular expression; this can come in handy if
    you have a lot of column names to pass to the `index` or `column_names`
    paramenters, and you do not wish to manually type them all.

    .. code-block:: python

             name    wk1   wk2   wk3   wk4
        0    Alice     5     9    20    22
        1    Bob       7    11    17    33
        2    Carla     6    13    39    40

        df = (pd.DataFrame(...)
              .pivot_longer(index = janitor.patterns("^(?!wk)"))
              )

             name   variable  value
        0   Alice      wk1      5
        1     Bob      wk1      7
        2   Carla      wk1      6
        3   Alice      wk2      9
        4     Bob      wk2     11
        5   Carla      wk2     13
        6   Alice      wk3     20
        7     Bob      wk3     17
        8   Carla      wk3     39
        9   Alice      wk4     22
        10    Bob      wk4     33
        11  Carla      wk4     40

    .. note:: Unpivoting a dataframe with MultiIndex columns, when
        either `names_sep` or `names_pattern` is provided is not
        supported.


    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)
        df = jn.pivot_longer(
            df = df,
            index = [column1, column2, ...],
            column_names = [column3, column4, ...],
            names_to = new_column_name,
            names_sep = string/regular expression,
            names_pattern = string/regular expression,
            values_to= new_column_name,
            column_level=None/int/str,
            sort_by_appearance = True/False,
            ignore_index = True/False,
        )

    Method chaining syntax:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .pivot_longer(
                index = [column1, column2, ...],
                column_names = [column3, column4, ...],
                names_to = new_column_name,
                names_sep = string/regular expression,
                names_pattern = string/regular expression,
                values_to= new_column_name,
                column_level=None/int/str,
                sort_by_appearance = True/False,
                ignore_index = True/False,
            )
        )

    :param df: A pandas dataframe.
    :param index: Name(s) of columns to use as identifier variables.
        Should be either a single column name, or a list/tuple of
        column names. The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        Index should be a list of tuples if the columns are a MultiIndex.
    :param column_names: Name(s) of columns to unpivot. Should be either
        a single column name or a list/tuple of column names.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        Column_names should be a list of tuples
        if the columns are a MultiIndex.
    :param names_to: Name of new column as a string that will contain
        what were previously the column names in `column_names`.
        The default is `variable` if no value is provided. It can
        also be a list/tuple of strings that will serve as new column
        names, if `name_sep` or `names_pattern` is provided.
        If `.value` is in `names_to`, new column names will be extracted
        from part of the existing column names and `values_to` will be
        replaced.
    :param names_sep: Determines how the column name is broken up, if
        `names_to` contains multiple values. It takes the same
        specification as pandas' `str.split` method, and can be a string
        or regular expression. `names_sep` does not work with MultiIndex
        columns.
    :param names_pattern: Determines how the column name is broken up.
        It can be a regular expression containing matching groups (it takes
        the same specification as pandas' `str.extract` method), or a
        list/tuple of regular expressions. If it is a single regex, the
        number of groups must match the length of ``names_to`` ( if the
        length of ``names_to`` is 3, then the number of groups must be 3.
        If ``names_to`` is a string, then there should be only one group
        in ``names_pattern``). For a list/tuple of regular expressions,
        ``names_to`` must also be a list/tuple and the lengths of both
        arguments must match(if the length of `names_to` is 4, then the
        length of `names_pattern` must also be 4). The entries in both
        arguments must also match positionally, i.e  if
        ``names_to = ("name1", "name2", "name3")``, then ``names_pattern``
        should be ("regex1", "regex2", "regex3"), with "name1" pairing
        "regex1", "name2" pairing "regex2", and "name3" pairing "regex3".
        `names_pattern` does not work with MultiIndex columns.
    :param values_to: Name of new column as a string that will contain what
        were previously the values of the columns in `column_names`.
    :param column_level: If columns are a MultiIndex, then use this level to
        unpivot the dataframe. Provided for compatibility with pandas' melt,
        and applies only if neither `names_sep` nor `names_pattern` is
        provided.
    :param sort_by_appearance: Default `False`. Boolean value that determines
        the final look of the dataframe. If `True`, the unpivoted dataframe
        will be stacked in order of first appearance. See examples for more
        details. `pivot_longer` is usually more performant if
        `sort_by_appearance` is ``False``.
    :param ignore_index: Default `True`. If True, original index is ignored.
        If False, the original index is retained and the Index labels will be
        repeated as necessary.
    :returns: A pandas DataFrame that has been unpivoted from wide to long
        format.
    :raises TypeError: if `index` or `column_names` is not a string, or a
        list/tuple of column names, or a `janitor.patterns` function.
    :raises TypeError: if `names_to` or `column_names` is not a string, or a
        list/tuple of strings.
    :raises TypeError: if `names_sep` is not a string or regular expression.
    :raises TypeError: if `names_pattern` is not a regular expression, or a
        list/tuple of regular expressions.
    :raises TypeError: if `values_to` is not a string.
    :raises TypeError: if `sort_by_appearance` is not a boolean.
    :raises TypeError: if `ignore_index` is not a boolean.
    :raises ValueError: if `names_to` is a string or a list/tuple of length 1,
        and `names_sep` is provided.
    :raises ValueError: if `names_to` is a string, and the number of extracted
        columns is greater than 1.
    :raises ValueError: if `names_to` is a list/tuple, and its length does not
        match the number of extracted columns.
    :raises ValueError: if the dataframe contains MultiIndex columns, and
        `index` or `column_names` is not a list of tuples.
    :raises ValueError: if the dataframe contains MultiIndex columns, and
        either `names_sep` or `names_pattern` is provided.

    .. # noqa: DAR402
    """

    # this code builds on the wonderful work of @benjaminjack’s PR
    # https://github.com/benjaminjack/pyjanitor/commit/e3df817903c20dd21634461c8a92aec137963ed0

    df = df.copy()

    (
        df,
        index,
        column_names,
        names_to,
        values_to,
        column_level,
        names_sep,
        names_pattern,
        sort_by_appearance,
        ignore_index,
    ) = _data_checks_pivot_longer(
        df,
        index,
        column_names,
        names_to,
        values_to,
        column_level,
        names_sep,
        names_pattern,
        sort_by_appearance,
        ignore_index,
    )

    df = _computations_pivot_longer(
        df,
        index,
        column_names,
        names_to,
        values_to,
        column_level,
        names_sep,
        names_pattern,
        sort_by_appearance,
        ignore_index,
    )

    return df


@pf.register_dataframe_method
def pivot_wider(
    df: pd.DataFrame,
    index: Optional[Union[List, str]] = None,
    names_from: Optional[Union[List, str]] = None,
    values_from: Optional[Union[List, str]] = None,
    names_sort: Optional[bool] = False,
    flatten_levels: Optional[bool] = True,
    names_from_position: Optional[str] = "first",
    names_prefix: Optional[str] = None,
    names_sep: Optional[str] = "_",
    aggfunc: Optional[Union[str, list, dict, Callable]] = None,
    fill_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """
    Reshapes data from long to wide form. The number of columns are
    increased, while decreasing the number of rows.

    It is the inverse of the `pivot_longer` method, and is a
    wrapper around `pd.DataFrame.unstack` method.

    This method does not mutate the original DataFrame.

    Reshaping to wide form :

    .. code-block:: python

             name variable  value
        0   Alice      wk1      5
        1   Alice      wk2      9
        2   Alice      wk3     20
        3   Alice      wk4     22
        4     Bob      wk1      7
        5     Bob      wk2     11
        6     Bob      wk3     17
        7     Bob      wk4     33
        8   Carla      wk1      6
        9   Carla      wk2     13
        10  Carla      wk3     39
        11  Carla      wk4     40

        df = (
            pd.DataFrame(...)
            .pivot_wider(
                index = "name",
                names_from = "variable",
                values_from = "value"
            )

             name    wk1   wk2   wk3   wk4
        0    Alice     5     9    20    22
        1    Bob       7    11    17    33
        2    Carla     6    13    39    40

    Pivoting on multiple columns is possible :

    .. code-block:: python

            name    n  pct
        0     1  10.0  0.1
        1     2  20.0  0.2
        2     3  30.0  0.3

        df = (
            pd.DataFrame(...)
            .assign(num = 0)
            .pivot_wider(
                index = "num",
                names_from = "name",
                values_from = ["n", "pct"],
                names_sep = "_"
             )
         )

            num n_1  n_2  n_3  pct_1  pct_2  pct_3
        0   0   10   20   30   0.1    0.2    0.3

    Aggregations are also possible with the ``aggfunc`` parameter::

        df = pd.DataFrame([{'id': 'a', 'name': 'Adam', 'value': 5},
                           {'id': 'b', 'name': 'Eve', 'value': 6},
                           {'id': 'c', 'name': 'Adam', 'value': 4},
                           {'id': 'a', 'name': 'Eve', 'value': 3},
                           {'id': 'd', 'name': 'Seth', 'value': 2},
                           {'id': 'b', 'name': 'Adam', 'value': 4},
                           {'id': 'a', 'name': 'Adam', 'value': 2}])

        id  name    value
        a   Adam    5
        b   Eve     6
        c   Adam    4
        a   Eve     3
        d   Seth    2
        b   Adam    4
        a   Adam    2

        df.pivot_wider(
            index = "id",
            names_from = "name",
            aggfunc = np.sum,
            values_from = "value",
            flatten_levels = True,
            fill_value = 0
            )

            id  Adam  Eve  Seth
        0   a     7    3     0
        1   b     4    6     0
        2   c     4    0     0
        3   d     0    0     2


    .. note:: You may choose not to collapse the levels by passing `False`
        to the ``flatten_levels`` argument.

    .. note:: An error is raised if the index is not unique and
        `aggfunc` is None.

    Functional usage syntax:

    .. code-block:: python

        import pandas as pd
        import janitor as jn

        df = pd.DataFrame(...)

        df = jn.pivot_wider(
            df = df,
            index = [column1, column2, ...],
            names_from = [column3, column4, ...],
            value_from = [column5, column6, ...],
            names_sort = True/False,
            names_prefix = string,
            names_sep = string,
            flatten_levels = True/False,
            names_from_position = "first"/"last",
            aggfunc,
            fill_value = fill_value
        )

    Method chaining syntax:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .pivot_wider(
                index = [column1, column2, ...],
                names_from = [column3, column4, ...],
                value_from = [column5, column6, ...],
                names_sort = True/False,
                names_prefix = string,
                names_sep = string,
                flatten_levels = True/False,
                names_from_position = "first"/"last",
                aggfunc,
                fill_value = fill_value
                )
        )

    :param df: A pandas dataframe.
    :param index: Name(s) of columns to use as identifier variables.
        Should be either a single column name, or a list of column names.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        If `index` is not provided, the current dataframe's index is used.
    :param names_from: Name(s) of column(s) to use to make the new
        dataframe's columns. Should be either a single column name, or a
        list of column names.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        A label or labels must be provided for ``names_from``.
    :param values_from: Name(s) of column(s) that will be used for populating
        the new dataframe's values. Should be either a single column name,
        or a list of column names.
        The `janitor.select_columns` syntax is supported here,
        allowing for flexible and dynamic column selection.
        If ``values_from`` is not specified,
        all remaining columns will be used. If `flatten_levels` is ``False``,
        a MultiIndex dataframe is created.
    :param names_sort: Default is `False`. Sorts columns by order of
        appearance. Applicable only if ``flatten_levels`` is `True`.
        Set as `True` to get the columns sorted lexicographicially,
        or if the columns are of category type.
    :param flatten_levels: Default is `True`. If `False`, the dataframe stays
        as a MultiIndex.
    :param names_from_position: By default, the values in ``names_from`` stay
        at the front of the new column names, even when ``values_from`` or
        ``aggfunc`` is a list. This can be changed to "last"; this places the
        values in ``names_from`` at the tail of the column names. Applicable
        only when ``flatten_levels`` is ``True``. Default is "first".
    :param names_prefix: String to be added to the front of each output column.
        Can be handy if the values in ``names_from`` are numeric data types.
        Applicable only if ``flatten_levels`` is True.
    :param names_sep: If ``names_from`` or ``values_from`` contain multiple
        variables, this will be used to join their values into a single string
        to use as a column name. Default is ``_``. Applicable only if
        ``flatten_levels`` is ``True``.
    :param aggfunc: An aggregate function. It can be a function, a string,
        list of functions, or a dictionary, pairing column name with aggregate
        function.
    :param fill_value: Scalar value to replace missing values with
        (after pivoting).
    :returns: A pandas DataFrame that has been unpivoted from long to wide
        form.
    :raises TypeError: if `index` or `names_from` is not a string, or a list of
        strings.
    :raises ValueError: if `names_from` is None.
    :raises TypeError: if `names_sep` is not a string.
    :raises TypeError: if `values_from` is not a string or a list of strings.
    :raises TypeError: if `names_sort` is not a boolean.
    :raises TypeError: if `flatten_levels` is not a boolean.
    :raises ValueError: if values in `index` or `names_from` or `values_from`
        do not exist in the dataframe.
    :raises ValueError: if the combination of `index` and `names_from` is not
        unique and ``aggfunc`` is ``None``.


    .. # noqa: DAR402
    """

    df = df.copy()

    (
        df,
        index,
        names_from,
        values_from,
        names_sort,
        flatten_levels,
        names_from_position,
        names_prefix,
        names_sep,
        aggfunc,
        fill_value,
    ) = _data_checks_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        names_sort,
        flatten_levels,
        names_from_position,
        names_prefix,
        names_sep,
        aggfunc,
        fill_value,
    )

    df = _computations_pivot_wider(
        df,
        index,
        names_from,
        values_from,
        names_sort,
        flatten_levels,
        names_from_position,
        names_prefix,
        names_sep,
        aggfunc,
        fill_value,
    )

    return df
