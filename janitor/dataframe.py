from pandas import DataFrame, Series

from .functions import (
    clean_names,
    coalesce,
    convert_excel_date,
    encode_categorical,
    expand_column,
    fill_empty,
    get_dupes,
    get_features_targets,
    label_encode,
    remove_empty,
    rename_column,
)

import warnings

msg = """Janitor's subclassed DataFrame and Series will be deprecated before
the 1.0 release. Instead of importing the Janitor DataFrame, please instead
`import janitor`, and use the functions directly attached to native pandas
dataframe."""

warnings.warn(msg)


class JanitorSeries(Series):

    @property
    def _constructor(self):
        return JanitorSeries

    @property
    def _constructor_expanddim(self):
        return JanitorDataFrame


class JanitorDataFrame(DataFrame):

    @property
    def _constructor(self):
        return JanitorDataFrame

    @property
    def _constructor_sliced(self):
        return JanitorSeries

    def clean_names(self):
        return clean_names(self)

    def remove_empty(self):
        return remove_empty(self)

    def get_dupes(self, columns=None):
        return get_dupes(self, columns)

    def encode_categorical(self, columns):
        return encode_categorical(self, columns)

    def rename_column(self, old, new):
        return rename_column(self, old, new)

    def get_features_targets(self, target_columns, feature_columns=None):
        return get_features_targets(self, target_columns, feature_columns)

    def coalesce(self, columns, new_column_name):
        return coalesce(self, columns, new_column_name)

    def convert_excel_date(self, column):
        return convert_excel_date(self, column)

    def fill_empty(self, columns, value):
        return fill_empty(self, columns, value)

    def label_encode(self, columns):
        return label_encode(self, columns)

    def expand_column(self, column, sep, concat=True):
        return expand_column(self, column, sep, concat)
