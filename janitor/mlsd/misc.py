import category_encoders as ce
import sklearn.decomposition as decomposition
from fancyimpute import SimpleFill
from sklearn.externals import joblib
from sklearn.feature_extraction import text
from sklearn.preprocessing import Normalizer, MinMaxScaler
import math

import pandas as pd
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm

# paso imports
from paso.base import pasoModel,pasoError,_Check_No_NA_Values,get_paso_log,toDataFrame,is_DataFrame

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#
#

class TfIdfVectorizer(pasoModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.vectorizer = text.TfidfVectorizer(**kwargs)

    def train(self, text):
        self.vectorizer.fit(text)
        return self

    def predict(self, text):
        return self.vectorizer.transform(text)

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)
        return self

class TruncatedSVD(pasoModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.truncated_svd = decomposition.TruncatedSVD(**kwargs)

    def train(self, features):
        self.truncated_svd.fit(features)
        return self

    def predict(self, features):
        return self.truncated_svd.transform(features)

    def load(self, filepath):
        self.truncated_svd = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.truncated_svd, filepath)
        return self


@register_DataFrame_method
def paso_toCurrencyNumeric(
        oX: pd.DataFrame,
        features: List = [],
        cleaning_style: str = None,
        cast_non_numeric: dict = None,
        fill_all_non_numeric: float = None,
        remove_non_numeric: bool = False,
        inplace: bool = True,
        verbose: bool = True,
) -> pd.DataFrame:
    """
    Convert currency column to numeric.

    This method allows one to take a column containing currency values,
    imported as a string, and cast it as a float. This is
    usually the case when reading CSV files that were modified in Excel.
    Empty strings (i.e. `''`) are retained as `NaN` values.

    Parameters:
        X: dataset

    Keywords:

        features:  default: []
            The column  names to  be transform from currency str float.


        cleaning_style:  default: None
            What style of cleaning to perform. If None, standard
            cleaning is applied. Options are:
            * 'accounting': Replaces numbers in parentheses with negatives, removes commas.

        cast_non_numeric: default: None
            A dict of how to coerce certain strings. For
            example, if there are values of 'REORDER' in the DataFrame,
            {'REORDER': 0} will cast all instances of 'REORDER' to 0..

        fill_all_non_numeric: default: None
            Similar to `cast_non_numeric`, but fills all
            strings to the same value. For example,  fill_all_non_numeric=1, will
            make everything that doesn't coerce to a currency 1.

        remove_non_numeric: default: False
            Will remove rows of a DataFrame that contain
            non-numeric values in the `column_name` column. Defaults to `False`.

        verbose: Default True
            True: output
            False: silent

        inplace: Default: True
            True: replace 1st argument with resulting dataframe
            False:  (boolean)change unplace the dataframe X

    Returns: pd.DataFrame

    """
    # :Example Setup:

    #     data = {
    #         "a": ["-$1.00", "", "REPAY"] * 2 + ["$23.00", "",
    # "Other Account"],
    #         "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
    #         "decorated-elephant": [1, 2, 3] * 3,
    #         "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    #     }
    #     X = pd.DataFrame(data)

    # :Example 1: Coerce numeric values in column to float:

    # .. code-block:: python

    #     X.currency_column_to_numeric("a")

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
    #     X.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric)

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

    #     X.currency_column_to_numeric("a", fill_all_non_numeric=35)

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

    #     X.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
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

    #     X.currency_column_to_numeric("a", remove_non_numeric=True)

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

    #     X.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
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

    _fun_name = paso_toCurrencyNumeric.__name__

    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()

    if features == []:
        features = X.columns  # set features all

    for feature in features:

        column_series = X[feature]
        if (len(column_series) == 0 or column_series[0].dtype != np.str):
            raise_PasoError('non-str data: {} of {} to  {}'.format(column_series[0], feature, _fun_name))

        if cleaning_style == "accounting":
            X.loc[:, feature] = X[feature].apply(_clean_accounting_column)
            return X

        elif cast_non_numeric:
            check("cast_non_numeric", cast_non_numeric, [dict])

            _make_cc_patrial = partial(
                _currency_column_to_numeric, cast_non_numeric=cast_non_numeric
            )

            column_series = column_series.apply(_make_cc_patrial)

            if remove_non_numeric:
                X = X.loc[column_series != "", :]

            # _replace_empty_string_with_none is applied here after the check on
            # remove_non_numeric since "" is our indicator that a string was coerced
            # in the original column
            column_series = column_series.apply(_replace_empty_string_with_none)

            if fill_all_non_numeric is not None:
                check("fill_all_non_numeric", fill_all_non_numeric, [int, float])
                column_series = column_series.fillna(fill_all_non_numeric)

        column_series = column_series.apply(_replace_original_empty_string_with_none)

        X = X.assign(**{feature: pd.to_numeric(column_series)})

    if verbose:
        logger.info("{} features:: {}".format(_fun_name, features))

    return X


@register_DataFrame_method
def paso_toExcelDatedatetime(
    oX: pd.DataFrame, column_name: Hashable, inplace: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
    Convert Excel's serial date format into Python datetime format.

    This method mutates the original DataFrame.

    Implementation is also from `Stack Overflow`.

    .. _Stack Overflow: https://stackoverflow.com/questions/38454403/convert-excel-style-date-with-pandas

    Functional usage example:

    .. code-block:: python

        X = convert_excel_date(X, column_name='date')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        X = pd.DataFrame(...).convert_excel_date('date')

    X: A pandas DataFrame.
    Hashable column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """
    _fun_name = paso_toExcelDatedatetime.__name__

    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()

    X[column_name] = pd.TimedeltaIndex(X[column_name], unit="d") + dt.datetime(
        1899, 12, 30)

    if verbose:
        logger.info("{} features:: {}".format(_fun_name, column_name))

    return X


@register_DataFrame_method
def paso_toMatlabDatedatetime(
    oX: pd.DataFrame, column_name: Hashable, inplace: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
    Convert Matlab's serial date number into Python datetime format.

    Implementation is also from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python

    This method mutates the original DataFrame.

    Functional usage example:

    .. code-block:: python

        X = convert_matlab_date(X, column_name='date')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        X = pd.DataFrame(...).convert_matlab_date('date')

    X: A pandas DataFrame.
    Hashable column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """
    _fun_name = paso_toMatlabDatedatetime.__name__

    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()


    if inplace:
        X = X
    else:
        X = X.copy()

    days = pd.Series([dt.timedelta(v % 1) for v in X[column_name]])
    X[column_name] = (
        X[column_name].astype(int).apply(dt.datetime.fromordinal)
        + days
        - dt.timedelta(days=366)
    )
    if verbose:
        logger.info("{} features:: {}".format(_fun_name, column_name))
    return X


@register_DataFrame_method
def paso_toUnixDatedatetime(oX: pd.DataFrame, column_name: Hashable,
    inplace: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Convert unix epoch time into Python datetime format.

    Note that this ignores local tz and convert all timestamps to naive
    datetime based on UTC!

    This method mutates the original DataFrame.

    Functional usage example:

    .. code-block:: python

        X = convert_unix_date(X, column_name='date')

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor
        X = pd.DataFrame(...).convert_unix_date('date')

    X: A pandas DataFrame.
    Hashable column_name: A column name.
    :returns: A pandas DataFrame with corrected dates.
    """

    _fun_name = paso_toUnixDatedatetime.__name__

    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()

    def _conv(value):
        try:
            date = dt.datetime.utcfromtimestamp(value)
        except ValueError:  # year of of rang means milliseconds.
            date = dt.datetime.utcfromtimestamp(value / 1000)
        return date

    if verbose:
        logger.info("{} features:: {}".format(_fun_name, column_name))


    X[column_name] = X[column_name].astype(int).apply(_conv)
    return X
