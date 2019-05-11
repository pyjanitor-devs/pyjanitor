"""
General purpose data cleaning functions.
"""
import datetime as dt
import re
import warnings
from fnmatch import translate
from functools import partial, reduce
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

from .errors import JanitorError
from .utils import deprecated_alias


def _strip_underscores(df, strip_underscores=None):
    """
    Strip underscores from DataFrames column names.
    Underscores can be stripped from the beginning, end or both.

    .. code-block:: python

        df = _strip_underscores(df, strip_underscores='left')

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer underscores from all
        column names. Default None keeps outer underscores. Values can be
        either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
        and True.
    :returns: A pandas DataFrame with underscores removed.
    """
    underscore_options = [None, "left", "right", "both", "l", "r", True]
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )

    if strip_underscores in ["left", "l"]:
        df = df.rename(columns=lambda x: x.lstrip("_"))
    elif strip_underscores in ["right", "r"]:
        df = df.rename(columns=lambda x: x.rstrip("_"))
    elif strip_underscores == "both" or strip_underscores is True:
        df = df.rename(columns=lambda x: x.strip("_"))
    return df


@pf.register_dataframe_method
def clean_names(
    df,
    strip_underscores: str = None,
    case_type: str = "lower",
    remove_special: bool = False,
    preserve_original_columns: bool = True,
):
    """
    Clean column names.

    Takes all column names, converts them to lowercase, then replaces all
    spaces with underscores.

    Functional usage example:

    .. code-block:: python

        df = clean_names(df)

    Method chaining example:

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
        Current case may be preserved with 'preserve'. Default 'lower'
        makes all characters lowercase.
    :param remove_special: (optional) Remove special characters from columns.
        Only letters, numbers and underscores are preserved.
    :returns: A pandas DataFrame.
    :param preserve_original_columns: (optional) Preserve original names.
        This is later retrievable using `df.original_columns`.
    """
    original_column_names = list(df.columns)

    assert case_type.lower() in {
        "preserve",
        "upper",
        "lower",
    }, "case_type argument must be one of ('preserve', 'upper', 'lower')"

    if case_type.lower() != "preserve":
        if case_type.lower() == "upper":
            df = df.rename(columns=lambda x: x.upper())

        elif case_type.lower() == "lower":
            df = df.rename(columns=lambda x: x.lower())

    df = df.rename(
        columns=lambda x: x.replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("'", "")
        .replace("’", "")
        .replace(",", "_")
        .replace("?", "_")
        .replace("-", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(".", "_")
    )

    def _remove_special(col):
        return "".join(item for item in col if item.isalnum() or "_" in item)

    if remove_special:
        df = df.rename(columns=_remove_special)

    df = df.rename(columns=lambda x: re.sub("_+", "_", x))
    df = _strip_underscores(df, strip_underscores)

    # Store the original column names, if enabled by user
    if preserve_original_columns:
        df.__dict__["original_columns"] = original_column_names
    return df


@pf.register_dataframe_method
def remove_empty(df):
    """
    Drop all rows and columns that are completely null. This method also
    resets the index(by default) since it doesn't make sense to preserve
    the index of a completely empty row.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/38884538/python-pandas-find-all-rows-where-all-values-are-nan  # noqa: E501

    Functional usage example:

    .. code-block:: python

        df = remove_empty(df)

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).remove_empty()

    :param df: The pandas DataFrame object.

    :returns: A pandas DataFrame.
    """
    nanrows = df.index[df.isnull().all(axis=1)]
    df = df.drop(index=nanrows).reset_index(drop=True)

    nancols = df.columns[df.isnull().all(axis=0)]
    df = df.drop(columns=nancols)

    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def get_dupes(df, column_names=None):
    """
    Return all duplicate rows.

    Functional usage example:

    .. code-block:: python

        df = pd.DataFrame(...)
        df = get_dupes(df)

    Method chaining example:

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
def encode_categorical(df, column_names):
    """
    Encode the specified columns with Pandas' `category`_ dtype.

    Functional usage example:

    .. code-block:: python

        categorical_cols = ['col1', 'col2', 'col4']
        df = df.encode_categorical(columns=categorical_cols)  # one way

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        categorical_cols = ['col1', 'col2', 'col4']
        df = pd.DataFrame(...).encode_categorical(columns=categorical_cols)

    :param df: The pandas DataFrame object.
    :param str/iterable column_names: A column name or an iterable (list or
        tuple) of column names.
    :returns: A pandas DataFrame

    .. _category: http://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html  # noqa: E501
    """
    if isinstance(column_names, list) or isinstance(column_names, tuple):
        for col in column_names:
            assert col in df.columns, JanitorError(
                "{col} missing from dataframe columns!".format(col=col)
            )
            df[col] = pd.Categorical(df[col])
    elif isinstance(column_names, str):
        assert column_names in df.columns, JanitorError(
            "{column_names} missing from dataframe columns!".format(
                column_names=column_names
            )
        )
        df[column_names] = pd.Categorical(df[column_names])
    else:
        raise JanitorError(
            "kwarg `column_names` must be a string or iterable!"
        )
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def label_encode(df, column_names):
    """
    Convert labels into numerical data.

    This function will create a new column with the string "_enc" appended
    after the original column's name. Consider this to be syntactic sugar.

    This function behaves differently from `encode_categorical`. This function
    creates a new column of numeric data. `encode_categorical` replaces the
    dtype of the original column with a "categorical" dtype.

    Functional usage example:

    .. code-block:: python

        label_encode(df, column_names="my_categorical_column")  # one way

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        categorical_cols = ['col1', 'col2', 'col4']
        df = pd.DataFrame(...).label_encode(column_names=categorical_cols)

    :param df: The pandas DataFrame object.
    :param str/iterable column_names: A column name or an iterable (list or
        tuple) of column names.
    :returns: A pandas DataFrame.
    """
    le = LabelEncoder()
    if isinstance(column_names, list) or isinstance(column_names, tuple):
        for col in column_names:
            assert col in df.columns, JanitorError(
                f"{col} missing from column_names"
            )  # noqa: E501
            df[f"{col}_enc"] = le.fit_transform(df[col])
    elif isinstance(column_names, str):
        assert column_names in df.columns, JanitorError(
            f"{column_names} missing from column_names"
        )  # noqa: E501
        df[f"{column_names}_enc"] = le.fit_transform(df[column_names])
    else:
        raise JanitorError(
            "kwarg `column_names` must be a string or iterable!"
        )
    return df


@pf.register_dataframe_method
@deprecated_alias(
    target_columns="target_column_names",
    feature_columns="feature_column_names",
)
def get_features_targets(df, target_column_names, feature_column_names=None):
    """
    Get the features and targets as separate DataFrames/Series.

    The behaviour is as such:

    - `target_column_names` is mandatory.
    - If `feature_column_names` is present, then we will respect the column names
    inside there.
    - If `feature_column_names` is not passed in, then we will assume that the
    rest of the columns are feature columns, and return them.

    Functional usage example:

    .. code-block:: python

        X, y = get_features_targets(df, target_column_names="measurement")

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...)
        target_cols = ['output1', 'output2']
        X, y = df.get_features_targets(target_column_names=target_cols)  # noqa: E501

    :param df: The pandas DataFrame object.
    :param str/iterable target_column_names: Either a column name or an iterable\
        (list or tuple) of column names that are the target(s) to be predicted.
    :param str/iterable feature_column_names: (optional) The column name or \
        iterable of column names that are the features (a.k.a. predictors) \
        used to predict the targets.
    :returns: (X, Y) the feature matrix (X) and the target matrix (Y). Both \
        are pandas DataFrames.
    """
    Y = df[target_column_names]

    if feature_column_names:
        X = df[feature_column_names]
    else:
        if isinstance(target_column_names, str):
            xcols = [c for c in df.columns if target_column_names != c]
        elif isinstance(target_column_names, list) or isinstance(
            target_column_names, tuple
        ):  # noqa: W503
            xcols = [c for c in df.columns if c not in target_column_names]
        X = df[xcols]
    return X, Y


@pf.register_dataframe_method
@deprecated_alias(old="old_column_name", new="new_column_name")
def rename_column(df, old_column_name, new_column_name):
    """
    Rename a column in place.

    Functional usage example:

    .. code-block:: python

        df = rename_column("old_column_name", "new_column_name")

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).rename_column("old_column_name", "new_column_name")  # noqa: E501

    This is just syntactic sugar/a convenience function for renaming one column
    at a time. If you are convinced that there are multiple columns in need of
    changing, then use the :py:meth:`pandas.DataFrame.rename` method.

    :param str old_column_name: The old column name.
    :param str new_column_name: The new column name.
    :returns: A pandas DataFrame with renamed columns.
    """
    if old_column_name not in df.columns:
        raise ValueError(
            f"{old_column_name} not present in dataframe columns!"
        )
    return df.rename(columns={old_column_name: new_column_name})


@pf.register_dataframe_method
def reorder_columns(
    df: pd.DataFrame, column_order: Union[List, pd.Index]
) -> pd.DataFrame:
    """
    Reorder DataFrame columns by specifying desired order as list of col names

    Columns not specified retain their order and follow after specified cols.

    Validates column_order to ensure columns are all present in DataFrame.

    Functional usage example:

    Given `DataFrame` with column names `col1`, `col2`, `col3`:

    .. code-block:: python

        df = reorder_columns(df, ['col2', 'col3'])

    Method chaining example:

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

    check("column_order", column_order, [list, pd.Index])

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
def coalesce(df, column_names, new_column_name):
    """

    Coalesces two or more columns of data in order of column names provided.

    Functional usage example:

    .. code-block:: python

        df = coalesce(df, column_names=['col1', 'col2'])

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).coalesce(['col1', 'col2'])

    The result of this function is that we take the first non-null value across
    rows.

    This is more syntactic diabetes! For R users, this should look familiar to
    `dplyr`'s `coalesce` function; for Python users, the interface
    should be more intuitive than the :py:meth:`pandas.Series.combine_first`
    method (which we're just using internally anyways).

    :param df: A pandas DataFrame.
    :param column_names: A list of column names.
    :param str new_column_name: The new column name after combining.
    :returns: A pandas DataFrame with coalesced columns.
    """
    series = [df[c] for c in column_names]

    def _coalesce(series1, series2):
        return series1.combine_first(series2)

    df = df.drop(columns=column_names)
    df[new_column_name] = reduce(_coalesce, series)  # noqa: F821
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_excel_date(df, column_name):
    """
    Convert Excel's serial date format into Python datetime format.

    Implementation is also from `Stack Overflow`.

    .. _Stack Overflow: https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas  # noqa: E501

    Functional usage example:

    .. code-block:: python

        df = convert_excel_date(df, column_name='date')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).convert_excel_date('date')

    :param df: A pandas DataFrame.
    :param str column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """
    df[column_name] = pd.TimedeltaIndex(
        df[column_name], unit="d"
    ) + dt.datetime(
        1899, 12, 30
    )  # noqa: W503
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_matlab_date(df, column_name):
    """
    Convert Matlab's serial date number into Python datetime format.

    Implementation is also from `Stack Overflow`.

    .. _Stack Overflow: https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python  # noqa: E501

    Functional usage example:

    .. code-block:: python

        df = convert_matlab_date(df, column_name='date')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).convert_matlab_date('date')

    :param df: A pandas DataFrame.
    :param str column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """
    days = pd.Series([dt.timedelta(v % 1) for v in df[column_name]])
    df[column_name] = (
        df[column_name].astype(int).apply(dt.datetime.fromordinal)
        + days
        - dt.timedelta(days=366)
    )
    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def convert_unix_date(df, column_name):
    """
    Convert unix epoch time into Python datetime format.
    Note that this ignores local tz and convert all
    timestamps to naive datetime based on UTC!

    Functional usage example:

    .. code-block:: python

        df = convert_unix_date(df, column_name='date')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).convert_unix_date('date')

    :param df: A pandas DataFrame.
    :param str column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """

    def _conv(value):
        try:
            date = dt.datetime.utcfromtimestamp(value)
        except ValueError:  # year of of rang means milliseconds.
            date = dt.datetime.utcfromtimestamp(value / 1000)
        return date

    df[column_name] = df[column_name].astype(int).apply(_conv)
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def fill_empty(df, column_names, value):
    """
    Fill `NaN` values in specified columns with a given value.

    Super sugary syntax that wraps :py:meth:`pandas.DataFrame.fillna`.

    Functional usage example:

    .. code-block:: python

        df = fill_empty(df, column_names=['col1', 'col2'], value=0)

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).fill_empty(df, column_names='col1', value=0)

    :param df: A pandas DataFrame.
    :param column_names: Either a `str` or `list` or `tuple`. If a string
        is passed in, then only that column will be filled; if a list or tuple
        of strings are passed in, then they will all be filled with the same
        value.
    :param value: The value that replaces the `NaN` values.
    :returns: A pandas DataFrame with `Nan` values filled.
    """
    if isinstance(column_names, list) or isinstance(column_names, tuple):
        for col in column_names:
            assert (
                col in df.columns
            ), "{col} missing from dataframe columns!".format(col=col)
            df[col] = df[col].fillna(value)
    else:
        assert (
            column_names in df.columns
        ), "{col} missing from dataframe columns!".format(col=column_names)
        df[column_names] = df[column_names].fillna(value)

    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def expand_column(df, column_name, sep, concat=True):
    """
    Expand a categorical column with multiple labels into dummy-coded columns.

    Super sugary syntax that wraps :py:meth:`pandas.Series.str.get_dummies`.

    Functional usage example:

    .. code-block:: python

        df = expand_column(df,
                           column_name='col_name',
                           sep=', ')  # note space in sep

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).expand_column(df,
                                             column_name='col_name',
                                             sep=', ')

    :param df: A pandas DataFrame.
    :param column_name: A `str` indicating which column to expand.
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
    df, column_names: List, new_column_name: str, sep: str = "-"
):
    """
    Concatenates the set of columns into a single column.

    Used to quickly generate an index based on a group of columns.

    Functional usage example:

    .. code-block:: python

        df = concatenate_columns(df,
                                 column_names=['col1', 'col2'],
                                 new_column_name='id',
                                 sep='-')

    Method chaining example:

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
    assert len(column_names) >= 2, "At least two columns must be specified"
    for i, col in enumerate(column_names):
        if i == 0:
            df[new_column_name] = df[col].astype(str)
        else:
            df[new_column_name] = (
                df[new_column_name] + sep + df[col].astype(str)
            )  # noqa: E501

    return df


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def deconcatenate_column(
    df, column_name: str, new_column_names: List, sep: str
):
    """
    De-concatenates a single column into multiple columns.

    This is the inverse of the `concatenate_columns` function.

    Used to quickly split columns out of a single column.

    Functional usage example:

    .. code-block:: python

        df = deconcatenate_columns(df,
                                   column_name='id',
                                   new_column_names=['col1', 'col2'],
                                   sep='-')

    Method chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...).
              deconcatenate_columns(column_name='id',
                                    new_column_names=['col1', 'col2'],
                                    sep='-'))

    :param df: A pandas DataFrame.
    :param column_name: The column to split.
    :param new_column_names: A list of new column names post-splitting.
    :param sep: The separator delimiting the column's data.
    :returns: A pandas DataFrame with a deconcatenated column.
    """
    assert (
        column_name in df.columns
    ), f"column name {column_name} not present in dataframe"  # noqa: E501
    deconcat = df[column_name].str.split(sep, expand=True)
    assert (
        len(new_column_names) == deconcat.shape[1]
    ), "number of new column names not correct."
    deconcat.columns = new_column_names
    return df.join(deconcat)


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def filter_string(
    df, column_name: str, search_string: str, complement: bool = False
):
    """
    Filter a string-based column according to whether it contains a substring.

    This is super sugary syntax that builds on top of
    `pandas.Series.str.contains`.

    Because this uses internally `pandas.Series.str.contains`, which allows a
    regex string to be passed into it, thus `search_string` can also be a regex
    pattern.

    This function allows us to method chain filtering operations:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_string('column', search_string='pattern', complement=False)  # noqa: E501
              ...)  # chain on more data preprocessing.

    This stands in contrast to the in-place syntax that is usually used:

    .. code-block:: python

        df = pd.DataFrame(...)
        df = df[df['column'].str.contains('pattern')]]

    As can be seen here, the API design allows for a more seamless flow in
    expressing the filtering operations.

    Functional usage example:

    .. code-block:: python

        df = filter_string(df,
                           column_name='column',
                           search_string='pattern'
                           complement=False)

    Method chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_string(column_name='column',
                             search_string='pattern'
                             complement=False)
              ...)

    :param df: A pandas DataFrame.
    :param column_name: The column to filter. The column should contain strings.
    :param search_string: A regex pattern or a (sub-)string to search.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """
    criteria = df[column_name].str.contains(search_string)
    if complement:
        return df[~criteria]
    else:
        return df[criteria]


@pf.register_dataframe_method
def filter_on(df, criteria, complement=False):
    """
    Return a dataframe filtered on a particular criteria.

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

    Functional usage example:

    .. code-block:: python

        df = filter_on(df,
                       'score < 50',
                       complement=False)

    Method chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_on('score < 50', complement=False)
              ...)

    Credit to Brant Peterson for the name.

    :param df: A pandas DataFrame.
    :param criteria: A filtering criteria that returns an array or Series of\
        booleans, on which pandas can filter on.
    :param complement: Whether to return the complement of the filter or not.
    :returns: A filtered pandas DataFrame.
    """
    if complement:
        return df.query("not " + criteria)
    else:
        return df.query(criteria)


@pf.register_dataframe_method
def filter_date(
    df: pd.DataFrame,
    column: str,
    start: dt.date = None,
    end: dt.date = None,
    years: List = None,
    months: List = None,
    days: List = None,
    column_date_options: Dict = None,
    format: str = None,
):
    """
    :Description:

    Filter a date-based column based on certain criteria

    Dates may be finicky and this function builds on top of the "magic" from
    the pandas `to_datetime` function that is able to parse dates well.

    Additional options to parse the date type of your column may be found at
    the official pandas documentation:

    pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

    **Note:** This method will cast your column to a Timestamp!

    :param df: A pandas dataframe.
    :param column: The column which to apply the fraction transformation.
    :param start: The beginning date to use to filter the DataFrame.
    :param end: The end date to use to filter the DataFrame.
    :param years: The years to use to filter the DataFrame.
    :param months: The months to use to filter the DataFrame.
    :param days: The days to use to filter the DataFrame.
    :param column_date_options: 'Special options to use when parsing the date\
    column in the original DataFrame. The options may be found at the official\
    Pandas documentation.'
    :param format: 'It you're using a format for start or end that is not\
    recognized natively by pandas' to_datetime function, you may supply the\
    format yourself. Python date and time formats may be found at\
    http://strftime.org/.'
    :returns: A filtered pandas DataFrame.

    **Note:** This only affects the format of the `start` and `end` parameters.
     If there's an issue with the format of the DataFrame being parsed, you
     would pass `{'format': your_format}` to `column_date_options`.

    :Setup:

    .. code-block:: python

        import pandas as pd
        import janitor

        date_list = [
            [1, "01/28/19"], [2, "01/29/19"], [3, "01/30/19"],
            [4, "01/31/19"], [5, "02/01/19"], [6, "02/02/19"],
            [7, "02/03/19"], [8, "02/04/19"], [9, "02/05/19"],
            [10, "02/06/19"], [11, "02/07/20"], [12, "02/08/20"],
            [13, "02/09/20"], [14, "02/10/20"], [15, "02/11/20"],
            [16, "02/12/20"], [17, "02/07/20"], [18, "02/08/20"],
            [19, "02/09/20"], [20, "02/10/20"], [21, "02/11/20"],
            [22, "02/12/20"], [23, "03/08/20"], [24, "03/09/20"],
            [25, "03/10/20"], [26, "03/11/20"], [27, "03/12/20"]]

        example_dataframe = pd.DataFrame(date_list,
                                         columns = ['AMOUNT', 'DATE'])


    :Example 1: Filter dataframe between two dates

    .. code-block:: python

        start = "01/29/19"
        end = "01/30/19"

        example_dataframe.filter_date('DATE', start=start, end=end)


    :Output:

    .. code-block:: python

           AMOUNT       DATE
        1       2 2019-01-29
        2       3 2019-01-30

    :Example 2: Using a different date format for filtering

    .. code-block:: python

        end = "01$$$30$$$19"
        format = "%m$$$%d$$$%y"

        example_dataframe.filter_date('DATE', end=end, format=format)


    :Output:

    .. code-block:: python

           AMOUNT       DATE
        0       1 2019-01-28
        1       2 2019-01-29
        2       3 2019-01-30

    :Example 3: Filtering by year

    .. code-block:: python

        years = [2019]

        example_dataframe.filter_date('DATE', years=years)


    :Output:

    .. code-block:: python


           AMOUNT       DATE
        0       1 2019-01-28
        1       2 2019-01-29
        2       3 2019-01-30
        3       4 2019-01-31
        4       5 2019-02-01
        5       6 2019-02-02
        6       7 2019-02-03
        7       8 2019-02-04
        8       9 2019-02-05
        9      10 2019-02-06

    :Example 4: Filtering by year and month

    .. code-block:: python

        years = [2020]
        months = [3]

        example_dataframe.filter_date('DATE', years=years, months=months)


    :Output:

    .. code-block:: python

            AMOUNT       DATE
        22      23 2020-03-08
        23      24 2020-03-09
        24      25 2020-03-10
        25      26 2020-03-11
        26      27 2020-03-12

    :Example 5: Filtering by year and day

    .. code-block:: python

        years = [2020]
        days = range(10,12)

        example_dataframe.filter_date('DATE', years=years, days=days)


    :Output:

    .. code-block:: python

            AMOUNT       DATE
        13      14 2020-02-10
        14      15 2020-02-11
        19      20 2020-02-10
        20      21 2020-02-11
        24      25 2020-03-10
        25      26 2020-03-11
    """

    check("column", column, [str])

    def _date_filter_conditions(conditions):
        """
        Taken from: https://stackoverflow.com/a/13616382
        """
        return reduce(np.logical_and, conditions)

    def _get_year(x):
        return x.year

    def _get_month(x):
        return x.month

    def _get_day(x):
        return x.day

    if column_date_options:
        df.loc[:, column] = pd.to_datetime(
            df.loc[:, column], **column_date_options
        )
    else:
        df.loc[:, column] = pd.to_datetime(df.loc[:, column])

    _filter_list = []

    if start:
        start_date = pd.to_datetime(start, format=format)
        _filter_list.append(df.loc[:, column] >= start_date)

    if end:
        end_date = pd.to_datetime(end, format=format)
        _filter_list.append(df.loc[:, column] <= end_date)

    if years:
        _filter_list.append(df.loc[:, column].apply(_get_year).isin(years))

    if months:
        _filter_list.append(df.loc[:, column].apply(_get_month).isin(months))

    if days:
        _filter_list.append(df.loc[:, column].apply(_get_day).isin(days))

    if start and end:
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
    column_name: str,
    iterable: Iterable,
    complement: bool = False,
):
    """
    Filters a dataframe based on whether the values of a given column are
    present inside another iterable.

    Assumes exact matching; fuzzy matching not implemented

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
def remove_columns(df: pd.DataFrame, column_names: List):
    """
    Removes the set of columns specified in `column_names`.

    Intended to be the method-chaining alternative to `del df[col]`.

    Method chaining example:

    .. code-block:: python

        df = pd.DataFrame(...).remove_columns(column_names=['col1', ['col2']])

    :param df: A pandas DataFrame
    :param column_names: The columns to remove.
    :returns: A pandas DataFrame.
    """
    return df.drop(columns=column_names)


@pf.register_dataframe_method
@deprecated_alias(column="column_name")
def change_type(df, column_name: str, dtype, ignore_exception=False):
    """
    Changes the type of a column.

    Exceptions that are raised can be ignored. For example, if one has a mixed
    dtype column that has non-integer strings and integers, and you want to
    coerce everything to integers, you can optionally ignore the non-integer
    strings and replace them with ``NaN``s or keep the original value

    Intended to be the method-chaining alternative to::

        df[col] = df[col].astype(dtype)

    Method chaining example:

    .. code-block:: python

        df = pd.DataFrame(...).change_type('col1', str)

    :param df: A pandas dataframe.
    :param column_name: A column in the dataframe.
    :param dtype: The datatype to convert to. Should be one of the standard
        Python types, or a numpy datatype.
    :param ignore_exception: one of {False, "fillna", "keep_values"}.
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
def add_column(df, column_name: str, value, fill_remaining: bool = False):
    """
    Adds a column to the dataframe.

    Intended to be the method-chaining alternative to::

        df[column_name] = value

    Method chaining example adding a column with only a single value:

    .. code-block:: python

        # This will add a column with only one value.
        df = pd.DataFrame(...).add_column(column_name="new_column", 2)

    Method chaining example adding a column with more than one value:

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

    :Setup:

    .. code-block:: python

        import pandas as pd
        import janitor
        data = {
            "a": [1, 2, 3] * 3,
            "Bell__Chart": [1, 2, 3] * 3,
            "decorated-elephant": [1, 2, 3] * 3,
            "animals": ["rabbit", "leopard", "lion"] * 3,
            "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
        }
        df = pd.DataFrame(data)

    :Example 1: Create a new column with a single value:

    .. code-block:: python

        df.add_column("city_pop", 100000)

    :Output:

    .. code-block:: python

           a  Bell__Chart  decorated-elephant  animals     cities  city_pop
        0  1            1                   1   rabbit  Cambridge    100000
        1  2            2                   2  leopard   Shanghai    100000
        2  3            3                   3     lion      Basel    100000
        3  1            1                   1   rabbit  Cambridge    100000
        4  2            2                   2  leopard   Shanghai    100000
        5  3            3                   3     lion      Basel    100000
        6  1            1                   1   rabbit  Cambridge    100000
        7  2            2                   2  leopard   Shanghai    100000
        8  3            3                   3     lion      Basel    100000

    :Example 2: Create a new column with an iterator \
    which fills to the column size:

    .. code-block:: python

        df.add_column("city_pop", range(3), fill_remaining=True)

    :Output:

    .. code-block:: python

           a  Bell__Chart  decorated-elephant  animals     cities  city_pop
        0  1            1                   1   rabbit  Cambridge         0
        1  2            2                   2  leopard   Shanghai         1
        2  3            3                   3     lion      Basel         2
        3  1            1                   1   rabbit  Cambridge         0
        4  2            2                   2  leopard   Shanghai         1
        5  3            3                   3     lion      Basel         2
        6  1            1                   1   rabbit  Cambridge         0
        7  2            2                   2  leopard   Shanghai         1
        8  3            3                   3     lion      Basel         2

    :Example 3: Add new column based on mutation of other columns:

    .. code-block:: python

        df.add_column("city_pop", df.Bell__Chart - 2 * df.a)

    :Output:

    .. code-block:: python

           a  Bell__Chart  decorated-elephant  animals     cities  city_pop
        0  1            1                   1   rabbit  Cambridge        -1
        1  2            2                   2  leopard   Shanghai        -2
        2  3            3                   3     lion      Basel        -3
        3  1            1                   1   rabbit  Cambridge        -1
        4  2            2                   2  leopard   Shanghai        -2
        5  3            3                   3     lion      Basel        -3
        6  1            1                   1   rabbit  Cambridge        -1
        7  2            2                   2  leopard   Shanghai        -2
        8  3            3                   3     lion      Basel        -3

    """
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
def add_columns(df: pd.DataFrame, fill_remaining: bool = False, **kwargs):
    """
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
def limit_column_characters(df, column_length: int, col_separator: str = "_"):
    """
    Truncates column sizes to a specific length.

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

    :Example Setup:

    .. code-block:: python

        import pandas as pd
        import janitor
        data_dict = {
            "really_long_name_for_a_column": range(10),
            "another_really_long_name_for_a_column": \
            [2 * item for item in range(10)],
            "another_really_longer_name_for_a_column": list("lllongname"),
            "this_is_getting_out_of_hand": list("longername"),
        }

    :Example: Standard truncation:

    .. code-block:: python

        example_dataframe = pd.DataFrame(data_dict)
        example_dataframe.limit_column_characters(7)

    :Output:

    .. code-block:: python

               really_  another another_1 this_is
        0        0        0         l       l
        1        1        2         l       o
        2        2        4         l       n
        3        3        6         o       g
        4        4        8         n       e
        5        5       10         g       r
        6        6       12         n       n
        7        7       14         a       a
        8        8       16         m       m
        9        9       18         e       e

    :Example: Standard truncation with different separator character:

    .. code-block:: python

        example_dataframe2 = pd.DataFrame(data_dict)
        example_dataframe2.limit_column_characters(7, ".")

    .. code-block:: python

               really_  another another.1 this_is
        0        0        0         l       l
        1        1        2         l       o
        2        2        4         l       n
        3        3        6         o       g
        4        4        8         n       e
        5        5       10         g       r
        6        6       12         n       n
        7        7       14         a       a
        8        8       16         m       m
        9        9       18         e       e

    """

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
    df,
    row_number: int = None,
    remove_row: bool = False,
    remove_rows_above: bool = False,
):
    """
    Elevates a row to be the column names of a DataFrame.

    Contains options to remove the elevated row from the DataFrame along with
    removing the rows above the selected row.

    :param df: A pandas DataFrame.
    :param row_number: The row containing the variable names
    :param remove_row: Whether the row should be removed from the DataFrame.
        Defaults to False.
    :param remove_rows_above: Whether the rows above the selected row should
        be removed from the DataFrame. Defaults to False.
    :returns: A pandas DataFrame with set column names.

    :Setup:

    .. code-block:: python

        import pandas as pd
        import janitor
        data_dict = {
            "a": [1, 2, 3] * 3,
            "Bell__Chart": [1, 2, 3] * 3,
            "decorated-elephant": [1, 2, 3] * 3,
            "animals": ["rabbit", "leopard", "lion"] * 3,
            "cities": ["Cambridge", "Shanghai", "Basel"] * 3
        }

    :Example: Move first row to column names:

    .. code-block:: python

        example_dataframe = pd.DataFrame(data_dict)
        example_dataframe.row_to_names(0)

    :Output:

    .. code-block:: python

           1  1  1   rabbit  Cambridge
        0  1  1  1   rabbit  Cambridge
        1  2  2  2  leopard   Shanghai
        2  3  3  3     lion      Basel
        3  1  1  1   rabbit  Cambridge
        4  2  2  2  leopard   Shanghai
        5  3  3  3     lion      Basel
        6  1  1  1   rabbit  Cambridge
        7  2  2  2  leopard   Shanghai

    :Example: Move first row to column names and remove row:

    .. code-block:: python

        example_dataframe = pd.DataFrame(data_dict)
        example_dataframe.row_to_names(0, remove_row=True)

    :Output:

    .. code-block:: python

           1  1  1   rabbit  Cambridge
        1  2  2  2  leopard   Shanghai
        2  3  3  3     lion      Basel
        3  1  1  1   rabbit  Cambridge
        4  2  2  2  leopard   Shanghai
        5  3  3  3     lion      Basel
        6  1  1  1   rabbit  Cambridge
        7  2  2  2  leopard   Shanghai
        8  3  3  3     lion      Basel

    :Example: Move first row to column names, remove row, \
    and remove rows above selected row:

    .. code-block:: python

        example_dataframe = pd.DataFrame(data_dict)
        example_dataframe.row_to_names(2, remove_row=True, \
            remove_rows_above=True)

    :Output:

    .. code-block:: python

           3  3  3     lion      Basel
        3  1  1  1   rabbit  Cambridge
        4  2  2  2  leopard   Shanghai
        5  3  3  3     lion      Basel
        6  1  1  1   rabbit  Cambridge
        7  2  2  2  leopard   Shanghai
        8  3  3  3     lion      Basel

    """

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
    df,
    column_name: str = None,
    denominator: float = None,
    digits: float = np.inf,
):
    """
    Round all values in a column to a fraction.

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
    #     example_dataframe2.limit_column_characters('a', 3)

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
    #     example_dataframe2.limit_column_characters('a', 3, 4)

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


    check("column_name", column_name, [str])

    if denominator:
        check("denominator", denominator, [float, int])

    if digits:
        check("digits", digits, [float, int])

    def _round_to_fraction(number, denominator, digits=np.inf):
        num = round(number * denominator, 0) / denominator
        if not np.isinf(digits):
            num = round(num, digits)
        return num

    _round_to_fraction_partial = partial(
        _round_to_fraction, denominator=denominator, digits=digits
    )

    df[column_name] = df[column_name].apply(_round_to_fraction_partial)

    return df


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", dest_col_name="dest_column_name")
def transform_column(
    df, column_name: str, function, dest_column_name: str = None
):
    """
    Transforms the given column in-place using the provided function.

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
        in. By default, replaces contents of original column.
    :returns: A pandas DataFrame with a transformed column.
    """

    if dest_column_name is None:
        dest_column_name = column_name

    df[dest_column_name] = df[column_name].apply(function)
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_names="new_column_names")
def transform_columns(
    df,
    column_names: List[str],
    function,
    suffix: str = None,
    new_column_names: Dict[str, str] = None,
):
    """
    Super syntactic sugar to transform a list of columns by the same
    transformation.

    Basically wraps `transform_column` and calls it repeatedly over all column
    names provided.

    User can optionally supply either a suffix to create a new set of columns
    with the specified suffix, or provide a dictionary mapping each original
    column name to its corresponding new column name.

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
    df, old_min=None, old_max=None, column_name=None, new_min=0, new_max=1
):
    """
    Scales data to between a minimum and maximum value.

    If `minimum` and `maximum` are provided, the true min/max of the
    `DataFrame` or column is ignored in the scaling process and replaced with
    these values, instead.

    One can optionally set a new target minimum and maximum value using the
    `new_min` and `new_max` keyword arguments. This will result in the
    transformed data being bounded between `new_min` and `new_max`.

    If a particular column name is specified, then only that column of data
    are scaled. Otherwise, the entire dataframe is scaled.

    Method chaining example:

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
def collapse_levels(df: pd.DataFrame, sep: str = "_"):
    """
    Given a `DataFrame` containing multi-level columns, flatten to single-
    level by string-joining the column labels in each level.

    After a `groupby` / `aggregate` operation where `.agg()` is passed a
    list of multiple aggregation functions, a multi-level `DataFrame` is
    returned with the name of the function applied in the second level.

    It is sometimes convenient for later indexing to flatten out this
    multi-level configuration back into a single level. This function does
    this through a simple string-joining of all the names across different
    levels in a single column.

    Method chaining example given two value columns `['max_speed', 'type']`:

    data = {"class": ["bird", "bird", "bird", "mammal", "mammal"],
            "max_speed": [389, 389, 24, 80, 21],
            "type": ["falcon", "falcon", "parrot", "Lion", "Monkey"]}


    df = (
        pd.DataFrame(data)
            .groupby('class')
            .agg(['mean', 'median'])
            .collapse_levels(sep='_')
    )

    Before applying `.collapse_levels`, the `.agg` operation returns a
    multi-level column `DataFrame` whose columns are (level 1, level 2):
    `[('class', ''), ('max_speed', 'mean'), ('max_speed', 'median'),
    ('type', 'mean'), ('type', 'median')]`
    `.collapse_levels` then flattens the column names to:
    `['class', 'max_speed_mean', 'max_speed_median',
    'type_mean', 'type_median']`

    :param df: A pandas DataFrame.
    :param sep: String separator used to join the column level names
    :returns: A flattened pandas DataFrame.
    """

    check("sep", sep, [str])

    # if already single-level, just return the DataFrame
    if not isinstance(df.columns.values[0], tuple):
        return df

    df.columns = [
        sep.join([str(el) for el in tup if str(el) != ""])
        for tup in df.columns.values
    ]

    return df


@pf.register_dataframe_method
def reset_index_inplace(df: pd.DataFrame, *args, **kwargs):
    """
    Returns the dataframe with an inplace resetting of the index.

    Compared to non-inplace resetting, this avoids data copying, thus
    providing a potential speedup.

    In Pandas, `reset_index()`, when used in place, does not return a
    `DataFrame`, preventing this option's usage in the function-chaining
    scheme. `reset_index_inplace()` provides one the ability to save
    computation time and memory while still being able to use the chaining
    syntax core to pyjanitor. This function, therefore, is the chaining
    equivalent of:


data = {"class": ["bird", "bird", "bird", "mammal", "mammal"],
        "max_speed": [389, 389, 24, 80, 21],
        "index": ["falcon", "falcon", "parrot", "Lion", "Monkey"]}

df = (
    pd.DataFrame(data).set_index("index")
        .drop_duplicates()
)

df.reset_index(inplace=True)

instead, being called simply as:

df = (
    pd.DataFrame(data).set_index("index")
        .drop_duplicates()
        .reset_index_inplace()
)

    All supplied parameters are sent directly to `DataFrame.reset_index()`.

    :param df: A pandas DataFrame.
    :param args: Arguments supplied to `DataFrame.reset_index()`
    :param kwargs: Arguments supplied to `DataFrame.reset_index()`
    :returns: A pandas DataFrame with reset indexes.
    """

    # Deprecation Warning
    warnings.warn(
        "reset_index_inplace will be deprecated in the "
        "upcoming 0.18 release. Use .reset_index() instead"
    )

    kwargs.update(inplace=True)

    df.reset_index(*args, **kwargs)
    return df


def check(varname: str, value, expected_types: list):
    """
    One-liner syntactic sugar for checking types.

    Should be used like this::

        check('x', x, [int, float])

    :param varname: The name of the variable.
    :param value: The value of the varname.
    :param expected_types: The types we expect the item to be.
    :returns: TypeError if data is not the expected type.
    """
    is_expected_type = False
    for t in expected_types:
        if isinstance(value, t):
            is_expected_type = True
            break

    if not is_expected_type:
        raise TypeError(
            "{varname} should be one of {expected_types}".format(
                varname=varname, expected_types=expected_types
            )
        )


def _clean_accounting_column(x):
    """
    This function performs the logic for the `type == "accounting"`
    attribute in currency_column_to_numeric.

    It is intended to be used in a pandas `apply` method.

    :returns: An object with a cleaned column.
    """

    y = x.strip()
    y = y.replace(",", "")
    y = y.replace(")", "")
    y = y.replace("(", "-")
    if y == "-":
        return 0.00
    return float(y)


def _currency_column_to_numeric(x, cast_non_numeric=None):
    """
    This function performs the logic for the changing cell values in
    the currency_column_to_numeric function.

    It is intended to be used in a pandas `apply` method, after being passed
    through `partial`.
    """
    acceptable_currency_characters = {
        "-",
        ".",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    }
    if len(x) == 0:
        return "ORIGINAL_NA"

    if cast_non_numeric:
        if x in cast_non_numeric.keys():
            check(
                "{%r: %r}" % (x, str(cast_non_numeric[x])),
                cast_non_numeric[x],
                [int, float],
            )
            return cast_non_numeric[x]
        else:
            return "".join(i for i in x if i in acceptable_currency_characters)
    else:
        return "".join(i for i in x if i in acceptable_currency_characters)


def _replace_empty_string_with_none(x):
    if isinstance(x, int):
        return x

    elif isinstance(x, float):
        return x

    elif len(x):
        return x


def _replace_original_empty_string_with_none(x):
    if x != "ORIGINAL_NA":
        return x


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", type="cleaning_style")
def currency_column_to_numeric(
    df,
    column_name: str,
    cleaning_style: str = None,
    cast_non_numeric: dict = None,
    fill_all_non_numeric: float = None,
    remove_non_numeric: bool = False,
):
    """
    This method allows one to take a column containing currency values,\
    inadvertently imported as a string, and cast it as a float. This is\
    usually the case when reading CSV files that were modified in Excel.\
    Empty strings (i.e. `''`) are retained as `NaN` values.

    :param df: The DataFrame
    :param column_name: The column to modify
    :param cleaning_style: What style of cleaning to perform. If None, standard
        cleaning is applied. Options are: 'accounting'.
    :param cast_non_numeric: A dict of how to coerce certain strings. For
        example, if there are values of 'REORDER' in the DataFrame,
        {'REORDER': 0} will cast all instances of 'REORDER' to 0.
    :param fill_all_non_numeric: Similar to `cast_non_numeric`, but fills all
        strings to the same value. For example,  fill_all_non_numeric=1, will
        make everything that doesn't coerce to a currency 1.
    :param remove_non_numeric: Will remove rows of a DataFrame that contain
        non-numeric values in the `column_name` column. Defaults to `False`.
    :returns: A mutated DataFrame.

    :Example Setup:

    .. code-block:: python

        import pandas as pd
        import janitor
        data = {
            "a": ["-$1.00", "", "REPAY"] * 2 + ["$23.00", "", "Other Account"],
            "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
            "decorated-elephant": [1, 2, 3] * 3,
            "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
            "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
        }
        df = pd.DataFrame(data)

    :Example 1: Coerce numeric values in column to float:

    .. code-block:: python

        df.currency_column_to_numeric("a")

    :Output:

    .. code-block:: python

              a  Bell__Chart  decorated-elephant animals@#$%^     cities
        0  -1.0     1.234523                   1       rabbit  Cambridge
        1   NaN     2.456234                   2      leopard   Shanghai
        2   NaN     3.234612                   3         lion      Basel
        3  -1.0     1.234523                   1       rabbit  Cambridge
        4   NaN     2.456234                   2      leopard   Shanghai
        5   NaN     3.234612                   3         lion      Basel
        6  23.0     1.234523                   1       rabbit  Cambridge
        7   NaN     2.456234                   2      leopard   Shanghai
        8   NaN     3.234612                   3         lion      Basel

    :Example 2: Coerce numeric values in column to float, and replace a string\
    value with a specific value:

    .. code-block:: python

        cast_non_numeric = {"REPAY": 22}
        df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric)

    :Output:

    .. code-block:: python

              a  Bell__Chart  decorated-elephant animals@#$%^     cities
        0  -1.0     1.234523                   1       rabbit  Cambridge
        1   NaN     2.456234                   2      leopard   Shanghai
        2  22.0     3.234612                   3         lion      Basel
        3  -1.0     1.234523                   1       rabbit  Cambridge
        4   NaN     2.456234                   2      leopard   Shanghai
        5  22.0     3.234612                   3         lion      Basel
        6  23.0     1.234523                   1       rabbit  Cambridge
        7   NaN     2.456234                   2      leopard   Shanghai
        8   NaN     3.234612                   3         lion      Basel

    :Example 3: Coerce numeric values in column to float, and replace all\
        string value with a specific value:

    .. code-block:: python

        df.currency_column_to_numeric("a", fill_all_non_numeric=35)

    :Output:

    .. code-block:: python


              a  Bell__Chart  decorated-elephant animals@#$%^     cities
        0  -1.0     1.234523                   1       rabbit  Cambridge
        1   NaN     2.456234                   2      leopard   Shanghai
        2  35.0     3.234612                   3         lion      Basel
        3  -1.0     1.234523                   1       rabbit  Cambridge
        4   NaN     2.456234                   2      leopard   Shanghai
        5  35.0     3.234612                   3         lion      Basel
        6  23.0     1.234523                   1       rabbit  Cambridge
        7   NaN     2.456234                   2      leopard   Shanghai
        8  35.0     3.234612                   3         lion      Basel

    :Example 4: Coerce numeric values in column to float, replace a string\
        value with a specific value, and replace remaining string values with\
        a specific value:

    .. code-block:: python

        df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
        fill_all_non_numeric=35)

    :Output:

    .. code-block:: python


              a  Bell__Chart  decorated-elephant animals@#$%^     cities
        0  -1.0     1.234523                   1       rabbit  Cambridge
        1   NaN     2.456234                   2      leopard   Shanghai
        2  22.0     3.234612                   3         lion      Basel
        3  -1.0     1.234523                   1       rabbit  Cambridge
        4   NaN     2.456234                   2      leopard   Shanghai
        5  22.0     3.234612                   3         lion      Basel
        6  23.0     1.234523                   1       rabbit  Cambridge
        7   NaN     2.456234                   2      leopard   Shanghai
        8  35.0     3.234612                   3         lion      Basel

    :Example 5: Coerce numeric values in column to float, and remove string\
        values:

    .. code-block:: python

        df.currency_column_to_numeric("a", remove_non_numeric=True)

    :Output:

    .. code-block:: python


              a  Bell__Chart  decorated-elephant animals@#$%^     cities
        0  -1.0     1.234523                   1       rabbit  Cambridge
        1   NaN     2.456234                   2      leopard   Shanghai
        3  -1.0     1.234523                   1       rabbit  Cambridge
        4   NaN     2.456234                   2      leopard   Shanghai
        6  23.0     1.234523                   1       rabbit  Cambridge
        7   NaN     2.456234                   2      leopard   Shanghai

    :Example 6: Coerce numeric values in column to float, replace a string\
        value with a specific value, and remove remaining string values:

    .. code-block:: python

        df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
        remove_non_numeric=True)

    :Output:

    .. code-block:: python


              a  Bell__Chart  decorated-elephant animals@#$%^     cities
        0  -1.0     1.234523                   1       rabbit  Cambridge
        1   NaN     2.456234                   2      leopard   Shanghai
        2  22.0     3.234612                   3         lion      Basel
        3  -1.0     1.234523                   1       rabbit  Cambridge
        4   NaN     2.456234                   2      leopard   Shanghai
        5  22.0     3.234612                   3         lion      Basel
        6  23.0     1.234523                   1       rabbit  Cambridge
        7   NaN     2.456234                   2      leopard   Shanghai
        """

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
    column_series = column_series.apply(_replace_empty_string_with_none)

    if fill_all_non_numeric is not None:
        check("fill_all_non_numeric", fill_all_non_numeric, [int, float])
        column_series = column_series.fillna(fill_all_non_numeric)

    column_series = column_series.apply(
        _replace_original_empty_string_with_none
    )

    df = df.assign(**{column_name: pd.to_numeric(column_series)})

    return df


@pf.register_dataframe_method
@deprecated_alias(search_cols="search_column_names")
def select_columns(
    df: pd.DataFrame, search_column_names: List, invert: bool = False
):
    """
    Method-chainable selection of columns.

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
        This will result in selection of the complement of the columns\
        provided.
    :returns: A pandas DataFrame with the specified columns selected.
    """

    full_column_list = []

    for col in search_column_names:
        search_string = translate(col)
        columns = [col for col in df if re.match(search_string, col)]
        full_column_list.extend(columns)

    return (
        df.drop(columns=full_column_list) if invert else df[full_column_list]
    )


@pf.register_dataframe_method
def impute(df, column: str, value=None, statistic=None):
    """
    Method-chainable imputation of values in a column.

    Underneath the hood, this function calls the `.fillna()` method available
    to every pandas.Series object.

    Method-chaining example:

    import numpy as np

    data = {
        "a": [1, 2, 3],
        "sales": np.nan,
        "score": [np.nan, 3, 2]}
    df = (
        pd.DataFrame(data)
            # Impute null values with 0
            .impute(column='sales', value=0.0)
            # Impute null values with median
            .impute(column='score', statistic='median')
    )

    Either one of ``value`` or ``statistic`` should be provided.

    If ``value`` is provided, then all null values in the selected column will
        take on the value provided.

    If ``statistic`` is provided, then all null values in the selected column
    will take on the summary statistic value of other non-null values.

    Currently supported ``statistic``s include:

    - ``mean`` (also aliased by ``average``)
    - ``median``
    - ``mode``
    - ``minimum`` (also aliased by ``min``)
    - ``maximum`` (also aliased by ``max``)

    :param df: A pandas DataFrame
    :param column: The name of the column on which to impute values.
    :param value: (optional) The value to impute.
    :param statistic: (optional) The column statistic to impute.
    :returns: An imputed pandas DataFrame.
    """

    # Firstly, we check that only one of `value` or `statistic` are provided.
    if value is not None and statistic is not None:
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
    if statistic is not None:
        # Check that the statistic keyword argument is one of the approved.
        if statistic not in funcs.keys():
            raise KeyError(f"`statistic` must be one of {funcs.keys()}")

        value = funcs[statistic](df[column].dropna().values)
        # special treatment for mode, because scipy stats mode returns a
        # moderesult object.
        if statistic is "mode":
            value = value.mode[0]

    # The code is architected this way - if `value` is not provided but
    # statistic is, we then overwrite the None value taken on by `value`, and
    # use it to set the imputation column.
    if value is not None:
        df[column] = df[column].fillna(value)
    return df


@pf.register_dataframe_method
def then(df: pd.DataFrame, func) -> pd.DataFrame:
    """
    Add an arbitrary function to run in the pyJanitor method chain.

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
def dropnotnull(df, column: str):
    """
    Drop rows that do not have null values in the given column.

    Example usage:

    .. code-block:: python

        df = pd.DataFrame(...).dropnotnull('column3')

    :param column: The column name to drop rows from.
    :param df: A pandas DataFrame.
    :returns: A pandas DataFrame with dropped rows.
    """
    return df[pd.isnull(df[column])]


@pf.register_dataframe_method
def find_replace(df: pd.DataFrame, column: str, mapper: dict):
    """
    Performs a find-and-replace action on a column of data.

    For example, let's say we have a column for which we want to replace all
    of the values 'a' with 1, 'b' with 2, 'c' with 3. We would use the
    following function call:

    .. code-block:: python

        df = (
            pd.DataFrame(...)
            .find_replace('column_name', {'a': 1, 'b': 2, 'c': 3})
        )

    This find-and-replace functionality does an exact match only. Hence,
    substring matches do not work. The value of a cell in the dataframe
    must be exactly 'a', 'b', or 'c', otherwise the replacement will not
    happen and the original data will be left in-place.

    :param df: A pandas DataFrame.
    :param column: The column on which the find/replace action is to be made.
    :param mapper: A dictionary that maps "thing to find" -> "thing to
        replace".
    :returns: A pandas DataFrame.
    """
    df[column] = df[column].apply(lambda x: mapper.get(x, x))
    return df


@pf.register_dataframe_method
def update_where(
    df: pd.DataFrame, conditions: None, target_col: None, target_val: None
):
    """
    Add multiple conditions to update a column in the dataframe.
    Example usage:

    .. code-block:: python

        # The dataframe must be assigned to a variable first.
        data = {
        "a": [1, 2, 3] * 3,
        "Bell__Chart": [1, 2, 3] * 3,
        "decorated-elephant": [1, 2, 3] * 3,
        "animals": ["rabbit", "leopard", "lion"] * 3,
        "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
        }
        df = pd.DataFrame(data)
        df = (
            df
            .update_where(
                condition=(df['column A'] == 'x') & (df['column B'] == 'y'),
                target_col='column C',
                target_val='z')
            )

    :param condition: conditions used to update a target column
        and target value
    :param target_col: Column to be updated
    :param target_val: Value to be updated
    :returns: An updated pandas DataFrame.
    """
    df.loc[conditions, target_col] = target_val
    return df


@pf.register_dataframe_method
def to_datetime(df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
    """

    Makes the pandas to_datetime method work as a chainable method.

    Functional usage example:

    .. code-block:: python

        df = to_datetime(df, 'col1', format='%Y%m%d')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).to_datetime('col1', format='%Y%m%d')

    :param df: A pandas DataFrame.
    :param column: Column name.
    :param kwargs: provide any kwargs that pd.to_datetime can take.
    :returns: A pandas DataFrame with updated datetime data.
    """

    df[column] = pd.to_datetime(df[column], **kwargs)

    return df


@pf.register_dataframe_method
def groupby_agg(
    df: pd.DataFrame,
    by: str,
    new_column: str,
    agg_column: str,
    agg: Union[Callable, str, List, Dict],
    axis: int = 0,
) -> pd.DataFrame:
    """

    Allow one to chain a groupby and a merge

    Without this function, we would have to break out of method chaining:

    .. code-block:: python

        df_grp = df.groupby(...).agg(...)
        df = df.merge(df_grp, ...)

    Now, this function can be method-chained:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).groupby_agg(df,
                                           by='col1',
                                           agg='mean',
                                           new_column='col1_mean')

    :param df: A pandas DataFrame.
    :param by: Column(s) to groupby on, either a `str` or
               a `list` of `str`
    :param new_column: Name of the aggregation output column.
    :param agg_column: Name of the column to aggregate over.
    :param agg: How to aggregate.
    :param axis: Split along rows (0) or columns (1).
    :returns: A pandas DataFrame.
    """

    df_grp = (
        df.groupby(by, axis=axis)
        .agg(agg, axis=axis)
        .reset_index()
        .rename(columns={agg_column: new_column})
    )

    if isinstance(by, list) or isinstance(by, tuple):
        df_grp = df_grp[[*by, new_column]]
    else:
        df_grp = df_grp[[by, new_column]]

    df = df.merge(df_grp, on=by)

    return df


@pf.register_dataframe_accessor("data_description")
class DataDescription:
    """
    Accessor that provides high-level description of data present
    in this DataFrame.
    """

    def __init__(self, data):
        self._data = data
        self._desc = dict()

    def _get_data_df(self):
        df = self._data

        data_dict = dict()
        data_dict["column_name"] = df.columns.tolist()
        data_dict["type"] = df.dtypes.tolist()
        data_dict["count"] = df.count().tolist()
        data_dict["pct_missing"] = (1 - (df.count() / len(df))).tolist()
        data_dict["description"] = [self._desc.get(c, "") for c in df.columns]

        return pd.DataFrame(data_dict).set_index("column_name")

    @property
    def df(self):
        """
        Get a table of descriptive information in a DataFrame format.
        """
        return self._get_data_df()

    def display(self):
        """
        Print the table of descriptive information about this DataFrame.
        """
        print(self._get_data_df())

    def set_description(self, desc: Union[List, Dict]):
        """
        Update the description for each of the columns in the DataFrame.

        :param desc: The structure containing the descriptions to update
        :type desc: list or dict
        """
        if isinstance(desc, list):
            assert len(desc) == len(self._data.columns)
            self._desc = dict(zip(self._data.columns, desc))

        elif isinstance(desc, dict):
            self._desc = desc


@pf.register_dataframe_method
def bin_numeric(
    df: pd.DataFrame,
    from_column: str,
    to_column: str,
    num_bins: int = 5,
    labels: str = None,
):
    """
    Makes use of pandas cut() function to bin data of one column, generating a
    new column with the results.

    .. code-block:: python

        import pandas as pd
        import janitor
        df = (
            pd.DataFrame(...)
            .bin_numeric(
                from_column='col1',
                to_column='col1_binned',
                num_bins=3,
                labels=['1-2', '3-4', '5-6']
                )
            )


    :param df: A pandas DataFrame.
    :param from_column: The column whose data you want binned.
    :param to_column: The new column to be created with the binned data.
    :param num_bins: The number of bins to be utilized.
    :param labels: Optionally rename numeric bin ranges with labels. Number of
    label names must match number of bins specified.

    :return: A pandas DataFrame.
    """

    if not labels:
        df[str(to_column)] = pd.cut(df[str(from_column)], bins=num_bins)
    else:
        if not len(labels) == num_bins:
            raise ValueError(f"Number of labels must match number of bins.")

        df[str(to_column)] = pd.cut(
            df[str(from_column)], bins=num_bins, labels=labels
        )

    return df


@pf.register_dataframe_method
def drop_duplicate_columns(
    df: pd.DataFrame, column_name: str, nth_index: int = 0
) -> pd.DataFrame:
    """
    Removes a duplicated column specified by column_name, its index

    Column order 0 is to remove the first column,
           order 1 is to remove the second column, and etc

    The corresponding tidyverse R's library is:
    `select(-<column_name>_<nth_index + 1>)`

    Method chaining example:

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
