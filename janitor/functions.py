"""
pyjanitor functions.

New data cleaning functions should be implemented here.
"""
import datetime as dt
import re
from functools import reduce
from warnings import warn

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pandas_flavor as pf

from .errors import JanitorError
from typing import List


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
    :returns: A pandas DataFrame.
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
def clean_names(df, strip_underscores=None, preserve_case=False):
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

    Example of transformation:

    .. code-block::

        Columns before: First Name, Last Name, Employee Status, Subject
        Columns after: first_name, last_name, employee_status, subject

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer underscores from all
        column names. Default None keeps outer underscores. Values can be
        either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
        and True.
    :param preserve_case: (optional) Allows you to choose whether to make all
        column names lowercase, or to preserve current cases. Default False
        makes all characters lowercase.
    :returns: A pandas DataFrame.
    """
    if preserve_case is False:
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

    df = df.rename(columns=lambda x: re.sub("_+", "_", x))
    df = _strip_underscores(df, strip_underscores)
    return df


@pf.register_dataframe_method
def remove_empty(df):
    """
    Drop all rows and columns that are completely null.

    Implementation is shamelessly copied from `StackOverflow`_.

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
    df.drop(index=nanrows, inplace=True)

    nancols = df.columns[df.isnull().all(axis=0)]
    df.drop(columns=nancols, inplace=True)

    return df


@pf.register_dataframe_method
def get_dupes(df, columns=None):
    """
    Return all duplicate rows.

    Functional usage example:

    .. code-block:: python

        get_dupes(df)

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).get_dupes()

    :param df: The pandas DataFrame object.
    :param str/iterable columns: (optional) A column name or an iterable (list
        or tuple) of column names. Following pandas API, this only considers
        certain columns for identifying duplicates. Defaults to using all
        columns.
    :returns: The duplicate rows, as a pandas DataFrame.
    """
    dupes = df.duplicated(subset=columns, keep=False)
    return df[dupes == True]  # noqa: E712


@pf.register_dataframe_method
def encode_categorical(df, columns):
    """
    Encode the specified columns as categorical column in pandas.

    Functional usage example:

    .. code-block:: python

        encode_categorical(df, columns="my_categorical_column")  # one way

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...)
        categorical_cols = ['col1', 'col2', 'col4']
        df = df.encode_categorical(columns=categorical_cols)

    :param df: The pandas DataFrame object.
    :param str/iterable columns: A column name or an iterable (list or tuple)
        of column names.
    :returns: A pandas DataFrame
    """
    msg = """If you are looking to encode categorical to use with scikit-learn,
    please use the label_encode method instead."""
    warn(msg)
    if isinstance(columns, list) or isinstance(columns, tuple):
        for col in columns:
            assert col in df.columns, JanitorError(
                "{col} missing from dataframe columns!".format(col=col)
            )
            df[col] = pd.Categorical(df[col])
    elif isinstance(columns, str):
        assert columns in df.columns, JanitorError(
            "{columns} missing from dataframe columns!".format(columns=columns)
        )
        df[columns] = pd.Categorical(df[columns])
    else:
        raise JanitorError("kwarg `columns` must be a string or iterable!")
    return df


@pf.register_dataframe_method
def label_encode(df, columns):
    """
    Convert labels into numerical data.

    This function will create a new column with the string "_enc" appended
    after the original column's name. Consider this to be syntactic sugar.

    This function behaves differently from `encode_categorical`. This function
    creates a new column of numeric data. `encode_categorical` replaces the
    dtype of the original column with a "categorical" dtype.

    Functional usage example:

    .. code-block:: python

        label_encode(df, columns="my_categorical_column")  # one way

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        categorical_cols = ['col1', 'col2', 'col4']
        df = pd.DataFrame(...).label_encode(columns=categorical_cols)

    :param df: The pandas DataFrame object.
    :param str/iterable columns: A column name or an iterable (list or tuple)
        of column names.
    :returns: A pandas DataFrame
    """
    le = LabelEncoder()
    if isinstance(columns, list) or isinstance(columns, tuple):
        for col in columns:
            assert col in df.columns, JanitorError(
                f"{col} missing from columns"
            )  # noqa: E501
            df[f"{col}_enc"] = le.fit_transform(df[col])
    elif isinstance(columns, str):
        assert columns in df.columns, JanitorError(
            f"{columns} missing from columns"
        )  # noqa: E501
        df[f"{columns}_enc"] = le.fit_transform(df[columns])
    else:
        raise JanitorError("kwarg `columns` must be a string or iterable!")
    return df


@pf.register_dataframe_method
def get_features_targets(df, target_columns, feature_columns=None):
    """
    Get the features and targets as separate DataFrames/Series.

    The behaviour is as such:

    - `target_columns` is mandatory.
    - If `feature_columns` is present, then we will respect the column names
      inside there.
    - If `feature_columns` is not passed in, then we will assume that the
      rest of the columns are feature columns, and return them.

    Functional usage example:

    .. code-block:: python

        X, y = get_features_targets(df, target_columns="measurement")

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...)
        target_cols = ['output1', 'output2']
        X, y = df.get_features_targets(target_columns=target_cols)  # noqa: E501

    :param df: The pandas DataFrame object.
    :param str/iterable target_columns: Either a column name or an iterable
        (list or tuple) of column names that are the target(s) to be predicted.
    :param str/iterable feature_columns: (optional) The column name or iterable
        of column names that are the features (a.k.a. predictors) used to
        predict the targets.
    :returns: (X, Y) the feature matrix (X) and the target matrix (Y). Both are
        pandas DataFrames.
    """
    Y = df[target_columns]

    if feature_columns:
        X = df[feature_columns]
    else:
        if isinstance(target_columns, str):
            xcols = [c for c in df.columns if target_columns != c]
        elif isinstance(target_columns, list) or isinstance(
            target_columns, tuple
        ):  # noqa: W503
            xcols = [c for c in df.columns if c not in target_columns]
        X = df[xcols]
    return X, Y


@pf.register_dataframe_method
def rename_column(df, old, new):
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

    :param str old: The old column name.
    :param str new: The new column name.
    :returns: A pandas DataFrame.
    """
    return df.rename(columns={old: new})


@pf.register_dataframe_method
def coalesce(df, columns, new_column_name):
    """
    Coalesces two or more columns of data in order of column names provided.

    Functional usage example:

    .. code-block:: python

        df = coalesce(df, columns=['col1', 'col2'])

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
    :param columns: A list of column names.
    :param str new_column_name: The new column name after combining.
    :returns: A pandas DataFrame.
    """
    series = [df[c] for c in columns]

    def _coalesce(series1, series2):
        return series1.combine_first(series2)

    df = df.drop(columns=columns)
    df[new_column_name] = reduce(_coalesce, series)  # noqa: F821
    return df


@pf.register_dataframe_method
def convert_excel_date(df, column):
    """
    Convert Excel's serial date format into Python datetime format.

    Implementation is also from `Stack Overflow`.

    .. _Stack Overflow: https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas  # noqa: E501

    Functional usage example:

    .. code-block:: python

        df = convert_excel_date(df, column='date')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).convert_excel_date('date')

    :param df: A pandas DataFrame.
    :param str column: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """
    df[column] = pd.TimedeltaIndex(df[column], unit="d") + dt.datetime(
        1899, 12, 30
    )  # noqa: W503
    return df


@pf.register_dataframe_method
def fill_empty(df, columns, value):
    """
    Fill `NaN` values in specified columns with a given value.

    Super sugary syntax that wraps :py:meth:`pandas.DataFrame.fillna`.

    Functional usage example:

    .. code-block:: python

        df = fill_empty(df, columns=['col1', 'col2'], value=0)

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).fill_empty(df, columns='col1', value=0)

    :param df: A pandas DataFrame.
    :param columns: Either a `str` or `list` or `tuple`. If a string is passed
        in, then only that column will be filled; if a list or tuple of strings
        are passed in, then they will all be filled with the same value.
    :param value: The value that replaces the `NaN` values.
    """
    if isinstance(columns, list) or isinstance(columns, tuple):
        for col in columns:
            assert (
                col in df.columns
            ), "{col} missing from dataframe columns!".format(
                col=col
            )
            df[col] = df[col].fillna(value)
    else:
        assert (
            columns in df.columns
        ), "{col} missing from dataframe columns!".format(
            col=columns
        )
        df[columns] = df[columns].fillna(value)

    return df


@pf.register_dataframe_method
def expand_column(df, column, sep, concat=True):
    """
    Expand a categorical column with multiple labels into dummy-coded columns.

    Super sugary syntax that wraps :py:meth:`pandas.Series.str.get_dummies`.

    Functional usage example:

    .. code-block:: python

        df = expand_column(df, column='colname',
                           sep=', ')  # note space in sep

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        df = pd.DataFrame(...).expand_column(df, column='colname', sep=', ')

    :param df: A pandas DataFrame.
    :param column: A `str` indicating which column to expand.
    :param sep: The delimiter. Example delimiters include `|`, `, `, `,` etc.
    :param bool concat: Whether to return the expanded column concatenated to
        the original dataframe (`concat=True`), or to return it standalone
        (`concat=False`).
    """
    expanded = df[column].str.get_dummies(sep=sep)
    if concat:
        df = df.join(expanded)
        return df
    else:
        return expanded


@pf.register_dataframe_method
def concatenate_columns(
    df, columns: list, new_column_name: str, sep: str = "-"
):
    """
    Concatenates the set of columns into a single column.

    Used to quickly generate an index based on a group of columns.

    Functional usage example:

    .. code-block:: python

        df = concatenate_columns(df,
                                 columns=['col1', 'col2'],
                                 new_column_name='id',
                                 sep='-')

    Method chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...).
              concatenate_columns(columns=['col1', 'col2'],
                                  new_column_name='id',
                                  sep='-'))

    :param df: A pandas DataFrame.
    :param columns: A list of columns to concatenate together.
    :param new_column_name: The name of the new column.
    :param sep: The separator between each column's data.
    """
    assert len(columns) >= 2, "At least two columns must be specified"
    for i, col in enumerate(columns):
        if i == 0:
            df[new_column_name] = df[col].astype(str)
        else:
            df[new_column_name] = (
                df[new_column_name] + sep + df[col].astype(str)
            )  # noqa: E501

    return df


@pf.register_dataframe_method
def deconcatenate_column(df, column: str, new_column_names: list, sep: str):
    """
    De-concatenates a single column into multiple columns.

    This is the inverse of the `concatenate_columns` function.

    Used to quickly split columns out of a single column.

    Functional usage example:

    .. code-block:: python

        df = deconcatenate_columns(df,
                                   column='id',
                                   new_column_names=['col1', 'col2'],
                                   sep='-')

    Method chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...).
              deconcatenate_columns(columns='id',
                                    new_column_name=['col1', 'col2'],
                                    sep='-'))

    :param df: A pandas DataFrame.
    :param column: The column to split.
    :param new_column_names: A list of new column names post-splitting.
    :param sep: The separator delimiting the column's data.
    """
    assert (
        column in df.columns
    ), f"column name {column} not present in dataframe"  # noqa: E501
    deconcat = df[column].str.split(sep, expand=True)
    assert (
        len(new_column_names) == deconcat.shape[1]
    ), "number of new column names not correct."
    deconcat.columns = new_column_names
    return df.join(deconcat)


@pf.register_dataframe_method
def filter_string(
    df, column: str, search_string: str, complement: bool = False
):
    """
    Filter a string-based column according to whether it contains a substring.

    This is super sugary syntax that builds on top of `filter_column` and
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
                           column='column',
                           search_string='pattern'
                           complement=False)

    Method chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_string(column='column',
                             search_string='pattern'
                             complement=False)
              ...)

    :param df: A pandas DataFrame.
    :param column: The column to filter. The column should contain strings.
    :param search_string: A regex pattern or a (sub-)string to search.
    :param complement: Whether to return the complement of the filter or not.
    """
    criteria = df[column].str.contains(search_string)
    return filter_on(df, criteria, complement=complement)


@pf.register_dataframe_method
def filter_on(df, criteria, complement=False):
    """
    Return a dataframe filtered on a particular criteria.

    This function allows us to method chain filtering operations:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_column(df['value'] < 3, complement=False)
              ...)  # chain on more data preprocessing.

    This stands in contrast to the in-place syntax that is usually used:

    .. code-block:: python
        df = pd.DataFrame(...)
        df = df[df['value'] < 3]

    As with the `filter_string` function, a more seamless flow can be expressed
    in the code.

    Functional usage example:

    .. code-block:: python

        df = filter_column(df,
                           df['value'] < 3,
                           complement=False)

    Method chaining example:

    .. code-block:: python

        df = (pd.DataFrame(...)
              .filter_column(df['value'] < 3
                             complement=False)
              ...)

    Credit to Brant Peterson for the name.

    :param df: A pandas DataFrame.
    :param criteria: A filtering criteria that returns an array or Series of
        booleans, on which pandas can filter on.
    :param complement: Whether to return the complement of the filter or not.
    """
    if complement:
        return df[~criteria]
    else:
        return df[criteria]


@pf.register_dataframe_method
def remove_columns(df: pd.DataFrame, columns: list):
    """
    Removes the set of columns specified in cols.

    Intended to be the method-chaining alternative to `del df[col]`.

    Method chaining example:

    .. code-block:: python

        df = pd.DataFrame(...).remove_column(cols=['col1', ['col2']])

    :param df: A pandas DataFrame
    :param cols: The columns to remove.
    """
    for col in columns:
        del df[col]
    return df


@pf.register_dataframe_method
def change_type(df, column: str, dtype):
    """
    Changes the type of a column.

    Intended to be the method-chaining alternative to::

        df[col] = df[col].astype(dtype)

    Method chaining example:

    .. code-block:: python

        df = pd.DataFrame(...).change_type('col1', str)

    :param df: A pandas dataframe.
    :param col: A column in the dataframe.
    :param dtype: The datatype to convert to. Should be one of the standard
        Python types, or a numpy datatype.
    """
    df[column] = df[column].astype(dtype)
    return df


@pf.register_dataframe_method
def add_column(df, colname: str, value):
    """
    Adds a column to the dataframe.

    Intended to be the method-chaining alternative to::

        df[colname] = value

    Method chaining example adding a column with only a single value:

    .. code-block:: python

        # This will add a column with only one value.
        df = pd.DataFrame(...).add_column(colname="new_column", 2)

    Method chaining example adding a column with more than one value:

    .. code-block:: python

        # This will add a column with an iterable of values.
        vals = [1, 2, 5, ..., 3, 4]  # of same length as the dataframe.
        df = pd.DataFrame(...).add_column(colname="new_column", vals)

    :param df: A pandas dataframe.
    :param colname: Name of the new column. Should be a string, in order
        for the column name to be compatible with the Feather binary
        format (this is a useful thing to have).
    :param value: Either a single value, or a list/tuple of values.
    """
    assert isinstance(colname, str), "`colname` must be a string!"
    df[colname] = value
    return df



@pf.register_dataframe_method
def limit_column_characters(df, column_length: int, col_separator: str = '_'):
    """
    Truncates column sizes to a specific length.

    Method chaining will truncate all columns to a given length and append
    a given separator character with the index of duplicate columns, except
    for the first distinct column name.

    :param df: A pandas dataframe.
    :param column_length: Character length for which to truncate all columns.
      The column separator value and number for duplicate column name does not
      contribute. Therefore, if all columns are truncated to 10 characters, the
      first distinct column will be 10 characters and the remaining will be 12
      characters (assuming a column separator of one character)
    :param col_separator: The separator to use for counting distinct column values.
      I think an underscore looks nicest, however a period is a common option as well.
      Supply an empty string (i.e. '') to remove the separator.

    """

    assert isinstance(column_length, int), "`column_length` must be an integer!"
    assert isinstance(col_separator, str), "`col_separator` must be a string!"

    col_names = df.columns
    col_names = [col_name[:column_length] for col_name in col_names]

    col_name_set = set(col_names)
    col_name_count = dict()


    for col_name_to_check in col_name_set:
        count = 0
        for idx, col_name in enumerate(col_names):
            if col_name_to_check == col_name:
                col_name_count[idx] = count
                count+=1


    final_col_names = []
    for idx, col_name in enumerate(col_names):
        if col_name_count[idx] > 0:
            col_name_to_append = col_name + col_separator + str(col_name_count[idx])
            final_col_names.append(col_name_to_append)
        else:
            final_col_names.append(col_name)

    df.columns = final_col_names
    return df