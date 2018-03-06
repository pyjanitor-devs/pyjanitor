from pandas import DataFrame, Series

from .functions import clean_names, encode_categorical, get_dupes, remove_empty


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
