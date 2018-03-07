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
    https://stackoverflow.com/questions/38884538/python-pandas-find-all-rows-where-all-values-are-nan

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


def get_features_targets(df, target_columns, feature_columns=None):
    """
    Get the features and targets as separate DataFrames/Series.

    The behaviour is as such:

    1. `target_columns` is mandatory.
    1. If `feature_columns` is present, then we will respect the column names
       inside there.
    1. If `feature_columns` is not passed in, then we will assume that the
       rest of the columns are feature columns, and return them.

    :param df: The pandas DataFrame object.
    :param str/iterable target_columns: Either a column name or an iterable
        (list or tuple) of column names that are the target(s) to be predicted.
    :param str/iterable feature_columns: (optional) The column name or iterable
        of column names that are the features (a.k.a. predictors) used to
        predict the targets.
    :returns: (X, Y) the feature matrix (X) and the target matrix (Y).
    """
    Y = df[target_columns]

    if feature_columns:
        X = df[feature_columns]
    else:
        if isinstance(target_columns, str):
            xcols = [c for c in df.columns if target_columns != c]
        elif (isinstance(target_columns, list)
                or isinstance(target_columns, tuple)):
            xcols = [c for c in df.columns if c not in target_columns]
        X = df[xcols]
    return X, Y


def rename_column(df, old, new):
    """
    Rename a column in place.

    This is just syntactic sugar/a convenience function for renaming one column
    at a time. If you are convinced that there are multiple columns in need of
    changing, then use the pandas DataFrame.rename({'old': 'new'}) syntax.
    """
    return df.rename(columns={old: new})
