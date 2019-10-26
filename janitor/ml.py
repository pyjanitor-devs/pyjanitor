""" Machine learning specific functions. """

import unicodedata
from typing import Hashable, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf

from .utils import deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(
    target_columns="target_column_names",
    feature_columns="feature_column_names",
)
def get_features_targets(
    df: pd.DataFrame,
    target_column_names: Union[str, Union[List, Tuple], Hashable],
    feature_column_names: Union[str, Iterable[str], Hashable] = None,
):
    """
    Get the features and targets as separate DataFrames/Series.

    This method does not mutate the original DataFrame.

    The behaviour is as such:

    - ``target_column_names`` is mandatory.
    - If ``feature_column_names`` is present, then we will respect the column
        names inside there.
    - If ``feature_column_names`` is not passed in, then we will assume that
    the rest of the columns are feature columns, and return them.

    Functional usage example:

    .. code-block:: python

        X, y = get_features_targets(df, target_column_names="measurement")

    Method chaining example:

    .. code-block:: python

        import pandas as pd
        import janitor.ml
        df = pd.DataFrame(...)
        target_cols = ['output1', 'output2']
        X, y = df.get_features_targets(target_column_names=target_cols)

    :param df: The pandas DataFrame object.
    :param str/iterable target_column_names: Either a column name or an
        iterable (list or tuple) of column names that are the target(s) to be
        predicted.
    :param str/iterable feature_column_names: (optional) The column name or
        iterable of column names that are the features (a.k.a. predictors)
        used to predict the targets.
    :returns: (X, Y) the feature matrix (X) and the target matrix (Y). Both
        are pandas DataFrames.
    """
    Y = df[target_column_names]

    if feature_column_names:
        X = df[feature_column_names]
    else:
        if isinstance(target_column_names, (list, tuple)):  # noqa: W503
            xcols = [c for c in df.columns if c not in target_column_names]
        else:
            xcols = [c for c in df.columns if target_column_names != c]

        X = df[xcols]
    return X, Y
