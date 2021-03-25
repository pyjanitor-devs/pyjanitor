# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"


from typing import Hashable, List

# import os, sys
# from pathlib import Path
import math
import pandas as pd
import numpy as np
from numba import jit
from scipy.sparse.csr import csr_matrix
from pandas.util._validators import validate_bool_kwarg

import warnings

warnings.filterwarnings("ignore")

# photonai imports
from janitor.mlsd.util import _Check_No_NA_F_Values
from janitor.mlsd.util import isDataFrame, isSeries
from janitor.mlsd.util import raise_janitor_Error
from janitor.mlsd.util import _must_be_list_tuple_int_float_str  # , _dict_value
from janitor.mlsd.util import register_DataFrame_method
from loguru import logger

def toDataFrame(X: any, labels: list = [], verbose: bool = True) -> pd.DataFrame:
    """
    Transform a list, tuple, csr_matrix, numpy 1-D or 2-D array,
    or pandas Series  into a  DataFrame.

    Parameters:
        X: dataset

    Keywords:

        labels:  default: []
            The column labels name to  be used for new DataFrame.
            If number of column names given is less than number of column names needed,
            then they will generared as Column_0...Column_n, where n is the number
            of missing column names.

        verbose:
            True: output
            False: silent (default)

    Raises:
        1. ValueError will result of unknown argument type.
        2. ValueError will result if labels is not a string or list of strings.

    Returns:  pd.DataFrame

    Note:
        A best practice is to make your dataset of type ``DataFrame`` at the start of your pipeline
        and keep the original DataFrame thoughout the pipeline of your experimental run to maximize
        speed of completion and minimize memory usage, THIS IS NOT THREAD SAFE.

        Almost all objects call ``toDataFrame(argument)`` ,which if argument
        is of type ``DataFrame``is very about 500x faster, or about 2 ns  for ``inplace=False``
        for single thread for a 1,000,000x8 DataFrame.

        If input argument is of type DataFrame,
        the return will be passed DataFrame as if `inplace=True```and ignores ``labels``

        If other than of type ``DataFrame`` then  `inplace=False``, and  `inplace`` is ignored
        and only remains for backwaeds compatability.

    """
    _fun_name = toDataFrame.__name__

    if len(X) == 0:
        raise_janitor_Error("{} X:any is of length O: {} ".format(_fun_name, str(type(X))))
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, copy=True)
        elif isinstance(X, list):
            # lists are always copied, but for consistency, we still pass the argument
            X = pd.DataFrame(np.transpose(X), copy=True)
        elif isinstance(X, (np.generic, np.ndarray)):
            if X.ndim != 2:
                raise_janitor_Error("{} X (1st arg): wrong dimension. must be 2: was {} dim ".format(
                        _fun_name, str(X.ndim)
                    )
                )
            X = pd.DataFrame(X, copy=True)
        elif isinstance(X, csr_matrix):
            X = pd.DataFrame(X.todense(), copy=True)
        else:
            raise_janitor_Error(
                "{} Unexpected input type: %s".format(_fun_name, str(type(X)))
            )

        new_col_names = labels
        nc = X.shape[1]
        for i in range(len(labels), X.shape[1]):
            new_col_names.append("c_" + str(i))

        X.columns = new_col_names

    if verbose:
        logger.info("{}  with \ncolumn names: {}".format(_fun_name, X.columns))

    return X

#
#
# helper
def _binary_value_to_integer( X: pd.DataFrame,
                    verbose: bool,
                    _fun_name: str = 'fn'
                    ) -> pd.DataFrame:
    for feature in X.columns:
        if X[feature].nunique() == 2:
            uniques = X[feature].unique()
            X[feature] = X[feature].astype("category")
            X[feature].replace(to_replace=uniques[0], value=int(0), inplace=True)
            X[feature].replace(to_replace=uniques[1], value=int(1), inplace=True)
            if verbose:
                logger.info(
                    "{} binary feature {} converted from: {} to 1/0".format(_fun_name, feature, uniques)
            )

    return X
#2
@register_DataFrame_method
def binary_value_to_integer(
    oX: pd.DataFrame, inplace: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
   Encoding and scaling
    and other data-set preprocessing should not be done here.

         Parameters:
           oX: dataset

        Keywords:
            inplace:
                True: mutate X, return X
                False: do no change X, return df-stats

            verbose:
                True: output (default)
                False: silent

        Returns:
            pd.DataFrame

        Note:
            All NaN values should be imputed or removed.
    """
    _fun_name = binary_value_to_integer.__name__
    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()

    if isSeries(X):
        X = X.to_frame()

    if isDataFrame(X):
        return(_binary_value_to_integer(X, verbose, _fun_name))
    # change  from whatever to 0,1


    return X

#
# helper
def _toCategory( X: pd.DataFrame,
                    boolean: bool = True,
                    integer: bool = True,
                    object_: str = True,
                    verbose: bool = True,
                    _fun_name: str = 'fn'
                    ) -> pd.DataFrame:
    for feature in X.columns:
        if X[feature].dtype == np.bool and boolean:
            X[feature] = X[feature].astype("category")
            if verbose:
                logger.info(
                    "{} boolean feature converted : {}".format(_fun_name, feature)
                )
        elif X[feature].dtype == np.object and object_:
            X[feature] = X[feature].astype("category")
            if verbose:
                logger.info(
                    "{} object(str) feature converted : {}".format(_fun_name, feature)
                )
        elif X[feature].dtype == np.integer and integer:
            X[feature] = X[feature].astype("category")
            if verbose:
                logger.info(
                    "{} integer feature converted : {}".format(_fun_name, feature)
                )
        else:
            pass
    return X

#3
@register_DataFrame_method
def toCategory(
    oX: pd.DataFrame,
    boolean: bool = True,
    integer: bool = True,
    object_: str = True,
    inplace: bool = True,
    verbose: bool = True
) -> pd.DataFrame:

    """
        Transforms any boolean, object or integer numpy array, list, tuple or
        any pandas DataFrame or series feature(s) type(s) to category type(s).
        The exception is continuous (``float`` or ``datetime``) which are
        returned as is. If you want to convert continuous or datetime types to
        category then use ``ContinuoustoCategory`` or ``DateTimetoCategory``
        before  (step) ``toCategory``.

        Parameters:
            oX: pd.DataFrame

        Keywords:

            boolean: bool Default: True
                If ``True`` will convert to ``category`` type.

            integer: Default: True
                If ``True`` will convert to ``category`` type.

            object_: Default: True
                If ``True`` will convert to ``category`` type.

            verbose: Default: True
                True: output
                False: silent

            inplace:
                True: (default) replace 1st argument with resulting dataframe
                False:  (boolean)change unplace the dataframe X

        Returns: pd.DataFrame

    Note:
        Assumes `data
        cleaning steps (such as removal of Null and NA values)
        have already been applied.

        ``datetime`` features should call ``toDatetimeComponents()``
        previous to this step so that ``datetime`` components (which are of type
        ``np.nmnber``) can be converted to ``category``. The default
        behavior of this step is NOT to convert ``datetime`` to ``category``.
   """
    _fun_name = toCategory.__name__
    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()

    if isSeries(X):
        X = X.to_frame()

    if isDataFrame(X):
        return(_toCategory(X, boolean, integer,
            object_, verbose,  _fun_name))

#4
@register_DataFrame_method
def toContinuousCategory(
    oX: pd.DataFrame,
    features: list = [],
    integer: bool = True,
    float_: bool = True,
    quantile: bool = True,
    nbin: int = 10,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Transforms any float, continuous integer values of
     a pandas dataframe to category values.

    Parameters:
        X: dataset

    Keywords:

        features:  [] (default)
            The column  names to  be transform from continuous to category.

        int_: True (default)
            set integer=False if not continuous and not to transform into category.

        float_: True (default)
            set floaty=False if not continuous and not to transform into category.

        quantile: True use quantile bin. (default)
            quantile is simular to v/(maxy-miny), works on any any scale.
            False, use fixed-width bin. miny,maxy arguments are ignored.

        nbin:  10 (default)
            Alternately ``nbins`` can be integer for number of bins. Or it can be
            array of quantiles, e.g. [0, .25, .5, .75, 1.]
            or array of fixed-width bin boundaries i.e. [0., 4., 10, 100].

        verbose:  True (default)
            True: output
            False: silent

        inplace: True (default)
            True: replace 1st argument with resulting dataframe
            False:  (boolean)change unplace the dataframe X

    Returns: pd.DataFrame

    Raises:
        TypeError('" requires boolean type.")

    Note:
        Binning, also known as quantization is used for
        transforming continuous numeric features
        (``np.number`` type) into ``category`` type.
        These categories group the continuous values
        into bins. Each bin represents a range of continuous numeric values.
        Specific strategies of binning data include fixed-width
        (``quantile_bins=False``) and adaptive binning (``quantile_bins = True``).

        Datasets that are used as ``train``, ``valid``, and ``test``
        must have same bin widths and labels and thus the
        same categories.

        Assumes  data
        cleaning steps (such as removal of Null and NA values)
        have already been applied.

        Fixed-width bin, only works, WITHOUT SCALING, with datasets with multiple features
        for tree-based models such as CART, random forest, xgboost, lightgbm,
        catboost,etc. Namely Deep Learning using neural nets won't work.
        quantile is similar to min-max scaling:  v/(maxy-miny)
        works on any any scale

        **Statistical problems with linear binning.**

        Binning increases type I and type II error; (simple proof is that as number
        of bins approaches infinity then information loss approaches zero).
        In addition, changing the number of bins will alter the bin distrution shape,
        unless the distribution is uniformLY FLAT.

        **Quantile binning can only be used with a singular data set.**

        Transforming a Continuous featuree ino a Category feature based on percentiles (QUANTILES) is WRONG
        if you have a train and test data sets. Quaniles are based on the data set and will be different unless
        each data set is distribution is equal. In rhe limit there are only two bins,
        then almost no relationship can be modeled. We are essentially doing a t-test.

        **if there are nonlinear or even nonmonotonic relationships between features**

        If you need linear binning, not quantile, use
        ``quantile_bins=False`` and specify the bin width (``delta``) or  fixed bin boundaries
        of any distribution of cuts you wish with ``nbin`` = [ cut-1, cut-2...cut-n ]

        **If you want Quantile-binning.**

        Despite the above warnings, your use case may require. qantile binning.
        Quantile based binning is a faily good strategy to use for adaptive binning.
        Quantiles are specific values or cut-points which partition
        the continuous valued distribution of a feature into
        discrete contiguous bins or intervals. Thus, q-Quantiles
        partition a numeric attribute into q equal (percetage-width) partitions.

        Well-known examples of quantiles include the 2-Quantile ,median,
        divides the data distribution into two equal (percetage-width) bins, 4-Quantiles,
        ,standard quartiles, 4 equal bins (percetage-width) and 10-Quantiles,
        deciles, 10 equal width (percetage-width) bins.

        **You should maybe looking for outliers AFTER applying a Gaussian transformation.**

    """
    _fun_name = toContinuousCategory.__name__
    if inplace:
        X = oX
    else:
        X = oX.copy()

    validate_bool_kwarg(integer, "inteer")
    validate_bool_kwarg(float_, "float_")
    # handles float, continuous integer. set integer=False if not contunuous
    # any other dataframe value type left as is.
    if isSeries(X):
        X = X.toframe()
    if features == []:
        features = X.columns

    for nth, feature in enumerate(features):
        if (float_ and X[feature].dtype == float) or (integer and X[feature].dtype == int):
            nbin = _must_be_list_tuple_int_float_str(nbin)
            # import pdb; pdb.set_trace() # debugging starts here
            if quantile:
                # quantile is similar to min-max scaling:  v/(maxy-miny)
                # works on any any scale
                X[feature + "q"] = pd.qcut(X[feature], nbin, duplicates="drop")
            else:
                # fixed-width bin, only works, WITHOUT SCALING, with datasets with multiple features
                # for tree-based models such as CART, random forest, xgboost, lightgbm,
                X[feature + "fw"] = pd.cut(X[feature], nbin, duplicates="drop")

    if verbose:
        logger.info("{} features:: {}".format(_fun_name, features))

    return X

#5
@register_DataFrame_method
def toColumnNamesFixedLen(
    oX: pd.DataFrame,
    column_length: int = 3,
    column_separator: str = "_",
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Truncate column name to a specific length.  If column length is
    shorter, then column length left as is.

    This method mutates the original DataFrame.

    Method chaining will truncate all columns to a given length and append
    a given separator character with the index of duplicate columns, except
    for the first distinct column name.

    Parameters:
        X: dataset

    Keywords:

        column_length: 3 (default)
            Character length for which to truncate all columns.
            The column separator value and number for duplicate
            column name does not contribute to total string length.
            If all columns string lenths are truncated to 10
            characters, the first distinct column will be 10
            characters and the remaining columns are
            12 characters with a column separator of one
            character).

        column_separator: "_" (default)
            The separator to append plus incremental Int to create
            unique column names. Care should be taken in choosing
            non-default str so as to create legal pandas column
            names.

        verbose: True (default)
            True: output
            False: silent

        inplace: True  (default)
            True: replace 1st argument with resulting dataframe
            False:  (boolean)change unplace the dataframe X

    Returns: A pandas DataFrame (pd.DataFrame) with truncated column lengths.

`
    """

    _fun_name = toColumnNamesFixedLen.__name__

    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()

    if isSeries(X):
        X = X.toframe()

    col_names = X.columns
    col_names = [col_name[:column_length] for col_name in col_names]

    col_name_set = set(col_names)
    col_name_count = dict()

    # If no columns are duplicates, we can skip the loops below.
    if len(col_name_set) == len(col_names):
        X.columns = col_names
        return X

    # case idenical. number from 1...n-1, where n idenical names
    for col_name_to_check in col_name_set:
        count = 0
        for idx, col_name in enumerate(col_names):
            if col_name_to_check == col_name:
                col_name_count[idx] = count
                count += 1

    final_col_names = []
    for idx, col_name in enumerate(col_names):
        if col_name_count[idx] > 0:
            col_name_to_append = col_name + column_separator + str(col_name_count[idx])
            final_col_names.append(col_name_to_append)
        else:
            final_col_names.append(col_name)

    if verbose:
        logger.info("{} features:: {}".format(_fun_name, final_col_names))

    X.columns = final_col_names
    return X


class DatetimetoComponents(object):
    _PREFIX_LENGTH_ = 3
    COMPONENT_DICT = {
        "Year": 100,
        "Month": 12,
        "Week": 52,
        "Day": 31,
        "Dayofweek": 5,
        "Dayofyear": 366,
        "Elapsed": 0,
        "Is_month_end": 1,
        "Is_month_start": 1,
        "Is_quarter_end": 1,
        "Is_quarter_start": 1,
        "Is_year_end": 1,
        "Is_year_start": 1,
    }


    @classmethod
    def _add_DatetimetoComponents_Year(cls):
        DatetimetoComponents.COMPONENT_DICT["Elapsed"] = (
            #            DatetimetoComponents.COMPONENT_DICT["Year"] *
                DatetimetoComponents.COMPONENT_DICT["Dayofyear"]
                * 24
                * 3600
        )  # unit seconds/year
        return cls


DatetimetoComponents._add_DatetimetoComponents_Year()


@register_DataFrame_method
def DatetimeComponents():
    return [k for k in DatetimetoComponents.COMPONENT_DICT.keys()]


@register_DataFrame_method
def toDatetimeComponents(
        oX: pd.DataFrame,
        drop: bool = True,
        components: list = [],
        prefix: bool = True,
        inplace: bool = True,
        verbose: bool = True,
) -> pd.DataFrame:
    #    import pdb; pdb.set_trace() # debugging starts here
    """
    Parameters:
        X: dataset

    Keywords:
        drop: True (default)
            If True then the datetime feature/column will be removed.

        components: [] (default) which results in all components
            list of column(feature) names for which datetime components
            are created.

            One or more of : [Year', 'Month', 'Week', 'Day','Dayofweek'
            , 'Dayofyear','Elapsed','Is_month_end'
            , 'Is_month_start', 'Is_quarter_end'
            , 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

        prefix: True (default)
            If True then the feature will be the prefix of the created datetime
            component fetures. The posfix will be _<component> to create the new
            feature column <feature>_<component>.

            if False only first _PREFIX_LENGTH_ characters of feature string eill be used to
            create the new feature name/column <featurename[0:2]>_<component>.

        verbose: True (default)
            True: output
            False: silent

        inplace: True (default)
            True: replace 1st argument with resulting dataframe
            False:  (boolean)change unplace the dataframe X

    Returns: pd.DataFrame  transformed into datetime feature components

    Raises:
        1. ValueError: if any dt_features = [].
        2. ValueError: if any feature has NA values.

    Note:
        Successful coercion to ``datetime`` costs approximately 100x more than if
        X[[dt_features]] was already of type datetime.

        Because of cost, a possible date will **NOT** be
        converted to ``datetime`` type.  -l 88 "$FilePath$"
        Another way, using a double negative is,
        if X[[dt_features]] is not of datetime type  (such as ``object`` type)
        then there **IS NO** attempt to coerce X[[dt_features]] to ``datetime`` type is made.

        It is best if raw data field
        is read/input in as ``datetime`` rather than ``object``. Another way, is to convert
        dataframe column using.

        Assumes  `data
        cleaning steps (such as removal of Null and NA values)
        have already been applied.
    """
    _fun_name = toDatetimeComponents.__name__

    # todo put in decorator
    if inplace:
        X = oX
    else:
        X = oX.copy()

    if isSeries(X):
        X = X.toframe()

    if components == []:
        components = [k for k in DatetimetoComponents.COMPONENT_DICT.keys()]
    if not isDataFrame(X):
        raise_janitor_Error("{} not passed DataFrame".format(_fun_name))

    for feature in X.columns:
        _Check_No_NA_F_Values(X, feature)
        try:
            # object/srtr converted to dt, if possible
            Xt = X[feature].dtype
            if Xt == np.object:
                X[feature] = pd.to_datetime(X[feature])
            # set new component feature name
            if prefix:
                fn = feature + "_"
            else:
                fn = feature[0: DatetimetoComponents._PREFIX_LENGTH_] + "_"

            for component in components:
                if component.lower() == "Elapsed".lower():
                    X[fn + "Elapsed"] = (X[feature].astype(np.int64) // 10 ** 9).astype(
                        np.int32
                    )
                else:
                    X[fn + component] = getattr(
                        X[feature].dt, component.lower()
                    )  # ns to seconds

                if verbose:
                    logger.info(
                        "datetime feature component added: {}".format(fn + component)
                    )
            if verbose:
                logger.info("datetime feature dropped: {}".format(feature))
        except:
            pass  # tryed but in dt format

    return X


