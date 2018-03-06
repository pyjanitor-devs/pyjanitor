import pandas as pd

from .errors import JanitorError


def clean_names(df):
    """
    Clean column names.

    Takes all column names, converts them to lowercase, then replaces all
    spaces with underscores.

    :param df: The pandas DataFrame object.
    :returns: A pandas DataFrame object.
    """
    columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = columns
    return df


def remove_empty(df):
    """
    Drop all rows and columns that are completely null.

    Implementation is shamelessly copied from StackOverflow:
    https://stackoverflow.com/questions/38884538/python-pandas-find-all-rows-where-all-values-are-nan  # noqa: E501

    :param df: The pandas DataFrame object.
    :returns: A pandas DataFrame object.
    """

    nanrows = df.index[df.isnull().all(axis=1)]
    df.drop(index=nanrows, inplace=True)

    nancols = df.columns[df.isnull().all(axis=0)]
    df.drop(columns=nancols, inplace=True)

    return df


def get_dupes(df, columns=None):
    """
    Returns all duplicate rows.

    :param df: The pandas DataFrame object.
    :param str/iterable columns: (optional) A column name or an iterable (list
        or tuple) of column names. Following pandas API, this only considers
        certain columns for identifying duplicates. Defaults to using all
        columns.
    """
    dupes = df.duplicated(subset=columns, keep=False)
    return df[dupes == True]  # noqa: E712


def encode_categorical(df, columns):
    """
    Encode the specified columns as categorical.

    :param df: The pandas DataFrame object.
    :param str/iterable columns: A column name or an iterable (list or tuple)
        of column names.
    """
    if isinstance(columns, list) or isinstance(columns, tuple):
        for col in columns:
            assert col in df.columns, \
                JanitorError(f"{col} missing from dataframe columns!")
            df[col] = pd.Categorical(df[col])
    elif isinstance(columns, str):
        df[columns] = pd.Categorical(df[columns])
    else:
        raise JanitorError('kwarg `columns` must be a string or iterable!')
    return df
