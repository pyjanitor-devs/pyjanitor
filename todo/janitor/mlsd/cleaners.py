# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

from typing import List, Union, Hashable

# any,
#    Callable,
#    Dict,
#    ,
#    Iterable,
#    Set,
#    Tuple,


import numpy as np
import pandas as pd
import re
import unicodedata

# from pandas.util._validators import validate_bool_kwarg
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

# import pandas_flavor as pf


warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer

# paso import
from paso.base import _Check_No_NA_Values, _array_to_string
from paso.base import _dict_value, _check_non_optional_kw
from paso.base import DataFrame_to_Xy, Xy_to_DataFrame
from paso.base import pasoFunction, pasoDecorators, raise_PasoError
from paso.base import _must_be_list_tuple_int, _merge_dicts
from paso.base import register_DataFrame_method
from loguru import logger


class Imputers(pasoFunction):
    """
          Class to impute NaM from dataset. Encoding and scaling
          and other data-set preprocessing should not be done here.

            description_file:

                project: HPKinetics/paso #[optional]
                verbose: True  #[optional]
                inplace: True #[optional]
                kind:
                    <strategy>:
                    description: "most_frequent value of feature is used for all NaN values"
                    genus: Imputer                             #[optional]
                    type: all-types

               strategy: (list)
                  _imputers_: substituting np.nan with strategy keywords.
                  class global given below

          Note:
              Impute before other cleaning and encoding, These steps all expect
              that NaNs have been removed. Use method Cleaners.Values_to_nan()
              beforehand to change ant incatitot values to NaN.

          """

    _imputer_type_supported_ = ("numeric", "all")
    _imputer_simple_stategies_ = {
        "median": "median",
        "mean": "mean",
        "most_frequent": "most_frequent",
        "random": "random",
    }

    _imputer_advanced_stategies_ = {"knn": "knn", "mice": "mice"}

    _imputers_ = _merge_dicts(_imputer_advanced_stategies_, _imputer_simple_stategies_)

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs) -> None:
        """
        Keywords:
            description_file:

        Returns: instance of class imputer

        """

        super().__init__()

    @staticmethod
    def imputers() -> List[str]:
        """
        Parameters:
            None

        Returns: List of available class inputers names.
        """
        return [k for k in Imputers._imputers_.keys()]

    @pasoDecorators.TTWrap(array=False, _Check_No_NAs=False)
    def transform(
        self,
        X: pd.DataFrame,
        verbose: bool = True,
        inplace: bool = True,
        features: List[str] = None,
    ) -> pd.DataFrame:
        """
        method to transform dataset by imputing NaN values. Encoding and scaling
        and other data-set preprocessing should not be done here.

        Parameters:
            X: dataset

        Keywords:
           features: default: None
                None: do all features (columns) in X
                [a,b....z] do only features liseted

            inplace:
                True: mutate X, return X
                False: do no change X, return X.copy()
                type: any to be laced with np.NaN

            verbose:
                True: output
                False: silent

        Returns: X or X.copy - see inplace

        """

        # todo pull datatypes automatically, just numericall and all??
        # todo support other data types besides i.e. most_frequent can support
        # cat/object/string if na
        # todo incorporate this A Comparison of Six Methods for Missing Data Imputation
        # https://www.omicsonline.org/open-access/a-comparison-of-six-methods-for-missing-data-imputation-2155-6180-1000224.pdf
        # https://impyute.readthedocs.io/en/master/index.html
        # todo checking passed arguments types are correct
        # enforce order of method calls

        # currently support only one learner, very brittle parser
        if self.kind == {}:
            raise_PasoError(
                "keyword kind must be present at top level:{}:".format(
                    self.ontology_kwargs
                )
            )

        if self.kind_name not in Imputers._imputers_:
            raise_PasoError(
                "no operation named: {} in imputers;: {}".format(
                    self.kind_name, Imputers._imputers_.keys()
                )
            )

        if self.inplace:
            nX = X
        else:
            nX = X.copy()

        if features == None:
            features = nX.columns
            logger.warning(
                "\nKeyword arg:features: not passed. All features of dataset will be checked for imputation\n{}".format(
                    features
                )
            )
        else:
            pass

        ## some of features not in columns
        if not len(set(features).difference(set(self.Xcolumns))) == 0:
            raise_PasoError(
                "\nfeature given {} not in \ndataset columns:{}".format(
                    features, nX.columns
                )
            )

        self.imputer_name = self.kind_name

        # check each attirbute for all nand
        for feature in features:
            if nX[feature].nunique() <= 1:
                raise_PasoError(
                    "Impute.transform: 1 or less unique values: {}.\n Remove this feature before calling Impute.".format(
                        nX[feature].unique()
                    )
                )

        if self.kind_name in Imputers._imputer_simple_stategies_:
            # assume target does not have missung values
            # todo handling target rows with missing values
            #  with simple row elimination? The idea being you should not try to predict missing values?
            imputer = SimpleImputer(strategy=self.kind_name)
            imputer.fit(nX[features])
            nX[features] = imputer.transform(nX[features])
        elif self.kind_name in Imputers._imputer_advanced_stategies_:
            self.imputer = Imputers._imputer_advanced_stategies_[self.imputer_name](
                **self.kind_name_kwargs
            )
            y = None  # todo fix _imputer_advanced_stategies_
            self.imputer.fit(nX, y)
            self.imputer_type = self.type
            nX[features] = self.imputer.transform(nX[features])

        return nX


@register_DataFrame_method
def paso_impute(
    X: pd.DataFrame,
    class_instance: Imputers,
    features: List[str] = None,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    method to transform dataset by imputing NaN values. Encoding and scaling
    and other data-set preprocessing should not be done here.

    Parameters:
        X: dataset

    Keywords:
       class_instance:

       target:

        inplace:
            True: mutate X, return X
            False: do no change X, return X.copy()
            type: any to be laced with np.NaN

        verbose:
            True: output
            False: silent

    Returns: X or X.copy - see inplace

    """
    if inplace:
        nX = X
    else:
        nX = X.copy()

    return class_instance.transform(
        nX, features=features, inplace=True, verbose=verbose
    )


class Cleaners(pasoFunction):
    """
    The class of data cleaners.

    """

    _statistics = [
        "kurt",
        "mad",
        "max",
        "mean",
        "median",
        "min",
        "sem",
        "skew",
        "sum",
        "std",
        "var",
        "nunique",
        "all",
    ]

    col_name_NaN_ratio = "NaN_ratio"


####### 1
@register_DataFrame_method
def paso_values_to_nan(*args, **kwargs) -> pd.DataFrame:
    return values_to_nan(*args, **kwargs)


def values_to_nan(
    X: pd.DataFrame, values: List[str] = [], inplace: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
    Different values can indicate, a value is missing. For example,
    - ``999`` could mean "did not answer" in some features.
    - ``NA`` could mean not-applicable for this feature/record.
    - ``-1`` could mean missing for this feature/record-1could mean missing for this feature/record`.
    and so on.

    Parameters:
        X: dataset

        values:

        inplace:
            True: 0mutate X, return X
            False: do no change X, return X.copy()

        verbose:
            True: output
            False: silent

    Returns: , X or X.copy - see inplace

    """
    values = _must_be_list_tuple_int(values)
    if values == []:
        return X

    y = X.replace(to_replace=values, value=np.nan, inplace=inplace)

    if verbose:
        logger.info("Values_to_nan {}".format(str(values)))

    if inplace:
        return X
    else:
        return y  # .reoplace accommplishes the inplace


#
########## 2
@register_DataFrame_method
def paso_calculate_NaN_ratio(*args, **kwargs) -> pd.DataFrame:
    return calculate_NaN_ratio(*args, **kwargs)


def calculate_NaN_ratio(
    X: pd.DataFrame, axis=1, inplace: bool = True, verbose: bool = True
) -> [pd.DataFrame, pd.DataFrame]:
    """
    For a row with a large ratio of missing values (an observation)
    renders it statistically irrelevant. However, the row is not removed.
    instead the nulls/total_feature_count is calculated for each row and
    a new feature  ``NA_ratio``is added to the returned **pandas** DataFrame.

    Note:
    Detecting and correcting for missing and outlier (good or bad)
    values is an evolving area of research.

        Parameters:
            X: dataset

        Keywords:
           axis : {0 or ‘index’, 1 or ‘columns’}
                Whether to drop labels from the index (0 or ‘index’)
                or columns (1 or ‘columns’).

            inplace:
                True: mutate X, return X
                False: do no change X, return X.copy()
                type: any to be laced with np.NaN

            verbose:
                True: output
                False: silent

        Returns: X or X.copy - see inplace

    Returns:
        axis = 0 pd.DataFrame n_row
        axis = 1 pd.DataFrame n_column

    """
    total_row_count = X.shape[0]
    total_feature_count = X.shape[1]

    if verbose:
        logger.info("Calculate_NA_ratio")

    if axis == 1:
        features = X.columns
        column_mvr = pd.DataFrame(columns=features)
        for feature in features:
            r = 1 - X[feature].count() / total_row_count
            column_mvr.at[0, feature] = r

    if inplace:
        if axis == 0:
            X[Cleaners.col_name_NaN_ratio] = 1 - X.count(axis=1) / total_feature_count
            return X
        elif axis == 1:
            return column_mvr
    else:
        if axis == 0:
            y = copy.deepcopy(X)
            return y
        if axis == 1:
            return column_mvr


############## 3
@register_DataFrame_method
def paso_delete_NA_Features(*args, **kwargs) -> pd.DataFrame:
    return delete_NA_Features(*args, **kwargs)


def delete_NA_Features(
    X: pd.DataFrame,
    threshold: List = [],
    axis: int = 0,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:

    if inplace:
        y = X
    else:
        y = X.copy()

    if axis == 0:  # r_mvr, drop rows
        y = calculate_NaN_ratio(y, inplace=inplace, verbose=verbose, axis=axis)
        return y[y[Cleaners.col_name_NaN_ratio] < threshold]

    elif axis == 1:  # drop features/columns
        column_mvr = calculate_NaN_ratio(y, inplace=inplace, verbose=verbose, axis=axis)
        for feature in y.columns:
            if column_mvr.at[0, feature] >= threshold:
                y.drop([feature], inplace=True, axis=1)
        return y

    else:
        raise_PasoError(
            "{} axis can be 0 ot 1 was:{}".format("column_missing_value_ratio", axis)
        )


########## 4
@register_DataFrame_method
def paso_delete_Duplicate_Features(*args, **kwargs) -> pd.DataFrame:
    return delete_Duplicate_Features(*args, **kwargs)


def delete_Duplicate_Features(
    X: pd.DataFrame,
    features: List[str] = [],
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    If a feature has the same values by index as another feature
    then one of those features should be deleted. The duplicate
    feature is redundant and will have no predictive power. 

    Duplicate features are quite common as an enterprise's
    database or data lake ages and different data sources are added.

    Parameters:
        X: dataset

    Keywords:
       features:  (default: None)
            None: do all NOT features (columns) in X
            [a,b....z] do only features listed

        inplace: 
            True: mutate X, return X
            False: do no change X, return X.copy()
            type: any to be laced with np.NaN

        verbose:
            True: output
            False: silent

    Returns: X or X.copy - see inplace

    """
    _Check_No_NA_Values(X)

    equal = {}

    for nth, f in enumerate(X.columns[0:-1]):
        if f in features:
            break
        # the twisted logic below is for speed. most values are ne
        for mth, g in enumerate(X.columns[nth + 1 :]):
            if (X[f].values != X[g].values).any():
                pass
            elif g not in equal:
                equal[g] = 1

    drop_list = list(equal.keys())
    if len(drop_list) > 0:
        y = X.drop(columns=drop_list, inplace=True, index=1)
        if verbose:
            logger.info("Duplicate_Features_Removed: {}".format(str(drop_list)))

    if inplace:
        return X
    else:
        return y


########## 5


@register_DataFrame_method
def paso_delete_Features_with_Single_Unique_Value(*args, **kwargs) -> pd.DataFrame:
    return delete_Features_with_Single_Unique_Value(*args, **kwargs)


def delete_Features_with_Single_Unique_Value(
    X: pd.DataFrame,
    features: List[str] = [],
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    This method finds all the features which have only one unique value.
    The variation between values is zero. All these features are removed from
    the DataFrame as they have no predictive ability.

    Parameters:
        X: dataset

    Keywords:
       features: (default: None)
            None: do all NOT features (columns) in X
            [a,b....z] do only features listed

        inplace: ,
            True: mutate X, return X
            False: do no change X, return X.copy()
            type: any to be laced with np.NaN

        verbose:
            True: output
            False: silent

    Returns: X or X.copy - see inplace

    """

    _Check_No_NA_Values(X)

    efs = []
    n = 0
    for f in X.columns:
        if f in features:
            pass
        # weirdness where df has
        # 2 or more features with same name
        #               len(df[f].squeeze().shape) == 1) and
        elif X[f].nunique() == 1:
            efs.append(f)
            n += 1
    if n > 0:
        if verbose:
            logger.info("Eliminate_Single_Unique_Value_Features {}".format(str(efs)))
        for f in efs:
            y = X.drop(f, inplace=inplace, axis=1)

    if inplace:
        return X
    else:
        return y


# 5nb.
@register_DataFrame_method
def paso_delete_Features_with_All_Unique_Values(*args, **kwargs) -> pd.DataFrame:
    return delete_Features_with_All_Unique_Values(*args, **kwargs)


def delete_Features_with_All_Unique_Values(
    X: pd.DataFrame,
    features: List[str] = [],
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:

    """
    This method finds all the features where every value is s
    unique value, whose unique count equals value count.
    All these features are removed from
    the DataFrame as they have no predictive ability.

    Parameters:
        X: dataset

    Keywords:
       features: (default: None)
            None: do all NOT features (columns) in X
            [a,b....z] do only features listed

        inplace:
            True: mutate X, return X
            False: do no change X, return X.copy()
            type: any to be laced with np.NaN

        verbose:
            True: output
            False: silent

    Returns: X or X.copy - see inplace

    Note: All NaN imputed or removed.
    """

    _Check_No_NA_Values(X)

    efs = []
    n = 0
    for f in X.columns:
        if f in features:
            pass
        # weirdness where df has
        # 2 or more features with same name
        #               len(df[f].squeeze().shape) == 1) and
        elif X[f].nunique() == X[f].count():
            efs.append(f)
            n += 1
    if n > 0:
        if verbose:
            logger.info("delete_Features_with_All_Unique_Values {}".format(str(efs)))
        for f in efs:
            y = X.drop(f, inplace=inplace, axis=1)

    if inplace:
        return X
    else:
        return y


########## 6.a
@register_DataFrame_method
def paso_statistics(*args, **kwargs) -> pd.DataFrame:
    return statistics(*args, **kwargs)


def statistics() -> List[str]:
    """
        List of all statistic terms available. The term "all" triggers
        calculation of all statistics.

        Returns: List[str]
    """
    return [k for k in Cleaners._statistics]


############ 6.b
@register_DataFrame_method
def paso_feature_Statistics(*args, **kwargs) -> pd.DataFrame:
    return feature_Statistics(*args, **kwargs)


def feature_Statistics(
    X: pd.DataFrame,
    statistics: str = "all",
    concat: bool = True,
    axis: int = 0,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    # todo row statistics
    """
        Calculate the statistics of each feature and returns a DataFrame
        where each row is that statistic.

        This method can be used as an diagnostic tool (concat = False)
        to decide if the sd or any other statistic  is too small
        and thus will have low predictive power.  It can also added onto the dataset
        (concat=True).

        When the target feature removed, each statistic helps desrcibe
        the distribution ot the feature (or row).
        This will only make sense for those features
        whose values are numeric.

        Parameters:
            X: dataset

        Keywords:
            statistics:  Must be 'all'or a list of symbols in  Cleaners._statistics

            concat:
                if concat is True, ten the statistics is concated to right (or bottom)
                of the DataFrame is returned

           axis : int {0 or ‘rows’, 1 or ‘columns’)
                Whether to calculate statistics for the rows (0)
                or columns (1).

            inplace:
                True: mutate X, return X
                False: do no change X, return df-stats

            verbose:
                True: output
                False: silent

        Returns:
            axis = 0 pd.DataFrame n_row (concat on right)
            axis = 1 pd.DataFrame n_column (concat on bottom)

        Note:
            All NaN values should be imputed or removed.
            Including target will cause leakage of ground truth into test.

            Statistics only make sense when all vales are scaled.

    """
    # work around automated review complaint
    if inplace:
        y = X
    else:
        y = X.copy()

    _Check_No_NA_Values(X)

    _must_be_list_tuple_int(statistics)
    if "all" in statistics:
        statistics = Cleaners._statistics[:-1]

    tmp_statistics = []
    for stat in statistics:
        if stat in Cleaners._statistics[:-1]:
            tmp_statistics.append(stat)
        else:
            raise_PasoError(
                "\n One of {}\n is unknown statistic from the list of accepted statistics\n{}".format(
                    statistics, Cleaners._statistics[:-1]
                )
            )
    statistics = tmp_statistics
    if verbose:
        logger.info("Features_Statistics {}".format(statistics))

    # by column stats
    if axis == 1:
        column_stats = pd.DataFrame()
        for stat in statistics:
            m = getattr(X,stat)(axis=1)
#            m = eval("X." + stat + "(axis=1)")
            column_stats = pd.concat([column_stats, m], axis=1)
        column_stats.columns = statistics
        if concat:
            return pd.concat([y, column_stats], axis=1)
        else:
            return column_stats

    # by row stats
    elif axis == 0:
        s = pd.DataFrame()
        for stat in statistics:
            m = getattr(X,stat)(axis=0)
#            m = pd.DataFrame(eval("X." + stat + "(axis=1)"))
            s = pd.concat([s, m], axis=0)
        if concat:
            return pd.concat([y, s], axis=0)
        else:
            return s


# 7
@register_DataFrame_method
def paso_boolean_to_integer(*args, **kwargs) -> pd.DataFrame:
    return boolean_to_integer(*args, **kwargs)


def boolean_to_integer(
    X: pd.DataFrame, inplace: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """
    transform spurious   features and values from dataset. Encoding and scaling
    and other data-set preprocessing should not be done here.

         Parameters:
            X: dataset

        Keywords:
            inplace:
                True: mutate X, return X
                False: do no change X, return df-stats

            verbose:
                True: output
                False: silent

        Returns:
            axis = 0 pd.DataFrame n_row (concat on right)
            axis = 1 pd.DataFrame n_column (concat on bottom)

        Note:
            All NaN values should be imputed or removed.
    """
    if not inplace:
        inplace = True
    # change boolean from whatever to 0,1
    X.replace(False, int(0), inplace=inplace)
    X.replace(True, int(1), inplace=inplace)

    if verbose:
        logger.info("\nboolean_to_integer features: {}".format(X.dtypes))

    return X


########## 8
@register_DataFrame_method
def paso_feature_Feature_Correlation(*args, **kwargs) -> pd.DataFrame:
    return feature_Feature_Correlation(*args, **kwargs)


def feature_Feature_Correlation(
    X: pd.DataFrame, method: str = "pearson", verbose: bool = True
) -> pd.DataFrame:
    """
    If any given Feature has an absolute high correlation coefficient
    with another feature (open interval -1.0,1.0) then is very likely
    the second one of them will have low predictive power as it is
    redundant with the other.

    Usually the Pearson correlation coefficient is used, which is
    sensitive only to a linear relationship between two variables
    (which may be present even when one variable is a nonlinear
    function of the other). A Pearson correlation coefficient
    of -0.97 is a strong negative correlation while a correlation
    of 0.10 would be a weak positive correlation.

    Spearman's rank correlation coefficient is a measure of how well
    the relationship between two variables can be described by a
    monotonic function. Kendall's rank correlation coefficient is
    statistic used to measure the ordinal association between two
    measured quantities. Spearman's rank correlation coefficient is
    the more widely used rank correlation coefficient. However,
    Kendall's is easier to understand.

    In most of the situations, the interpretations of Kendall and
    Spearman rank correlation coefficient are very similar to
    Pearson correlation coefficient and thus usually lead to the
    same diagnosis. The paso class calculates Peason's,or Spearman's,
    or Kendall's correlation co-efficients for all feature pairs of
    the dataset.

    One of the features of the feature-pair should be removed for
    its negligible effect on the prediction. Again, this class is
    a diagnostic that indicates if one feature of will have low
    predictive power. Care should be used before eliminating any
    feature to look at the **SHAP** value, (sd/mean) and correlation
    co-efficient in order to reach a decision to remove a feature.

    Parameters:
         Parameters:
            X: dataset

        Keywords:
            method : {‘pearson’, ‘kendall’, ‘spearman’} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarrays

            verbose:
                True: output
                False: silent

        Returns:
            axis = 0 pd.DataFrame n_row (concat on right)
            axis = 1 pd.DataFrame n_column (concat on bottom)

        Note:
            All NaN values should be imputed or removed.
    Returns:
            Correlation of X DataFrame

    Note: All NaN imputed or removed.

    """
    if verbose:
        logger.info("Correlation method: {}", method)
    corr = X.corr(method=method)
    return corr


@register_DataFrame_method
def paso_plot_corr(*args, **kwargs) -> pd.DataFrame:
    return plot_corr(*args, **kwargs)


def plot_corr(
    X: pd.DataFrame,
    kind: str = "numeric",
    mirror: bool = False,
    xsize: float = 10,
    ysize: float = 10,
) -> None:
    """"
    Plot of correlation matrix.

    Parameters:
        X: `dataset

        kind : {‘numeric’, ‘visual’}

        mirror:
            If True,show opposite side  (mirror) half of matrix.

        xsize:

        ysize:

    Note: All NaN imputed or removed.

    """

    # todo put in EDA module

    def plot_corr_numeric(corr):
        if mirror:
            sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
            plt.xticks(range(len(corr.columns)), corr.columns)
            plt.yticks(range(len(corr.columns)), corr.columns)
        else:
            dropSelf = np.zeros_like(corr)
            dropSelf[np.triu_indices_from(dropSelf)] = True
            colormap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
            plt.xticks(range(len(corr.columns)), corr.columns)
            plt.yticks(range(len(corr.columns)), corr.columns)

    def heatmap(x, y, **kwargs):
        if "color" in kwargs:
            color = kwargs["color"]
        else:
            color = [1] * len(x)

        if "palette" in kwargs:
            palette = kwargs["palette"]
            n_colors = len(palette)
        else:
            n_colors = 256  # Use 256 colors for the diverging color palette
            palette = sns.color_palette("Blues", n_colors)

        if "color_range" in kwargs:
            color_min, color_max = kwargs["color_range"]
        else:
            color_min, color_max = (
                min(color),
                max(color),
            )  # Range of values that will be mapped to the palette, i.e. min and max possible correlation

        def value_to_color(val):
            if color_min == color_max:
                return palette[-1]
            else:
                val_position = float((val - color_min)) / (
                    color_max - color_min
                )  # position of value in the input range, relative to the length of the input range
                val_position = min(
                    max(val_position, 0), 1
                )  # bound the position betwen 0 and 1
                ind = int(
                    val_position * (n_colors - 1)
                )  # target index in the color palette
                return palette[ind]

        if "size" in kwargs:
            size = kwargs["size"]
        else:
            size = [1] * len(x)

        if "size_range" in kwargs:
            size_min, size_max = kwargs["size_range"][0], kwargs["size_range"][1]
        else:
            size_min, size_max = min(size), max(size)

        size_scale = kwargs.get("size_scale", 500)

        def value_to_size(val):
            if size_min == size_max:
                return 1 * size_scale
            else:
                val_position = (val - size_min) * 0.99 / (
                    size_max - size_min
                ) + 0.01  # position of value in the input range, relative to the length of the input range
                val_position = min(
                    max(val_position, 0), 1
                )  # bound the position betwen 0 and 1
                return val_position * size_scale

        if "x_order" in kwargs:
            x_names = [t for t in kwargs["x_order"]]
        else:
            x_names = [t for t in sorted(set([v for v in x]))]
        x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

        if "y_order" in kwargs:
            y_names = [t for t in kwargs["y_order"]]
        else:
            y_names = [t for t in sorted(set([v for v in y]))]
        y_to_num = {p[1]: p[0] for p in enumerate(y_names)}

        plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x10 grid
        ax = plt.subplot(
            plot_grid[:, :-1]
        )  # Use the left 14/15ths of the grid for the main plot

        marker = kwargs.get("marker", "s")

        kwargs_pass_on = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "color",
                "palette",
                "color_range",
                "size",
                "size_range",
                "size_scale",
                "marker",
                "x_order",
                "y_order",
            ]
        }

        ax.scatter(
            x=[x_to_num[v] for v in x],
            y=[y_to_num[v] for v in y],
            marker=marker,
            s=[value_to_size(v) for v in size],
            c=[value_to_color(v) for v in color],
            **kwargs_pass_on,
        )
        ax.set_xticks([v for k, v in x_to_num.items()])
        ax.set_xticklabels(
            [k for k in x_to_num], rotation=45, horizontalalignment="right"
        )
        ax.set_yticks([v for k, v in y_to_num.items()])
        ax.set_yticklabels([k for k in y_to_num])

        ax.grid(False, "major")
        ax.grid(True, "minor")
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

        ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
        ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
        ax.set_facecolor("#F1F1F1")

        # Add color legend on the right side of the plot
        if color_min < color_max:
            ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

            col_x = [0] * len(palette)  # Fixed x coordinate for the bars
            bar_y = np.linspace(
                color_min, color_max, n_colors
            )  # y coordinates for each of the n_colors bars

            bar_height = bar_y[1] - bar_y[0]
            ax.barh(
                y=bar_y,
                width=[5] * len(palette),  # Make bars 5 units wide
                left=col_x,  # Make bars start at 0
                height=bar_height,
                color=palette,
                linewidth=0,
            )
            ax.set_xlim(
                1, 2
            )  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
            ax.grid(False)  # Hide grid
            ax.set_facecolor("white")  # Make background white
            ax.set_xticks([])  # Remove horizontal ticks
            ax.set_yticks(
                np.linspace(min(bar_y), max(bar_y), 3)
            )  # Show vertical ticks for min, middle and max
            ax.yaxis.tick_right()  # Show vertical ticks on the right

    def plot_corr_visual(data, size_scale=500, marker="s"):
        corr = pd.melt(data.reset_index(), id_vars="index")
        corr.columns = ["x", "y", "value"]
        heatmap(
            corr["x"],
            corr["y"],
            color=corr["value"],
            color_range=[-1, 1],
            palette=sns.diverging_palette(20, 220, n=256),
            size=corr["value"].abs(),
            size_range=[0, 1],
            marker=marker,
            x_order=data.columns,
            y_order=data.columns[::-1],
            size_scale=size_scale,
        )

    _Check_No_NA_Values(X)
    corr = X.corr()
    _, _ = plt.subplots(figsize=(xsize, ysize))
    _ = sns.diverging_palette(220, 10, as_cmap=True)
    if kind == "numeric":
        plot_corr_numeric(corr)
    elif kind == "visual":
        plot_corr_visual(corr)
    else:
        raise_PasoError("plot_corr, unknown kind:{}".format(kind))

    plt.show()


########## 8
@register_DataFrame_method
def paso_delete_Features(*args, **kwargs) -> pd.DataFrame:
    return delete_Features(*args, **kwargs)


def delete_Features(
    X: pd.DataFrame,
    features: List[str] = [],
    verbose: bool = True,
    inplace: bool = True,
) -> pd.DataFrame:
    """
        This class finds all the features which have only one unique value.
        The variation between values is zero. All these features are removed from
        the DataFrame as they have no predictive ability.

        Parameters:
            X: dataset

            features:
                The features (column names) to eliminate.

            inplace:
                True: mutate X, return X
                False: do no change X, return df-stats

            verbose:
                True: output
                False: silent

        Returns:
            X:
                transformed X DataFrame

        Note:
            The end goal of any feature elimination is to increase speed and perhaps
            decrease the loss. are should be used before eliminating
            any feature and a 2nd opinion of the **SHAP** value should be used in order
            to reach a decision to remove a feature.
    """

    y = X.drop(features, axis=1, inplace=inplace)

    if verbose:
        logger.info("delete_Features {}".format(features))

    if inplace:
        return X
    else:
        return y


########## 9
@register_DataFrame_method
def paso_delete_Features_not_in_train_or_test(*args, **kwargs) -> pd.DataFrame:
    return delete_Features_not_in_train_or_test(*args, **kwargs)


def delete_Features_not_in_train_or_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str] = [],
    verbose: bool = True,
    inplace: bool = True,
) -> pd.DataFrame:

    """
    If the train or test datasets have features the other
    does not, then those features will have no predictive
    power and should be removed from both datasets.
    The exception being the target feature that is present
    in the training dataset of a supervised problem. 
    Duplicate features are quite common as an enterprise's
    database or datalake ages and different data sources are added.

    Parameters:
        train: dataset

        test: dataset

        features:
            The features (column names) not to eliminate.
            Usually this keywod argument is used for the target feature
            that is maybe present in the training dataset.

        inplace:
            True: mutate X, return X
            False: do no change X, return df-stats

        verbose:
            True: output
            False: silent

    Returns:
        X: transformed train DataFrame
        Y: transformed test DataFrame

    Note: All NaN imputed or removed.

    """
    if inplace:
        X = train
        y = test
    else:
        X = train.copy()
        y = test.copy()

    _Check_No_NA_Values(X)
    _Check_No_NA_Values(y)

    rem = set(X.columns).difference(set(y.columns))
    x_features_cleaned_efs = []
    x_features_cleaned = 0
    if len(rem) >= 1:
        for f in rem:
            if f not in ignore:
                X.drop(f, inplace=True, axis=1)
                x_features_cleaned_efs.append(f)
                x_features_cleaned += 1
    if x_features_cleaned > 0:
        if verbose:
            logger.info("Clean_Features_in_X: {}".format(str(x_features_cleaned_efs)))

    rem = set(y.columns).difference(set(X.columns))
    y_features_cleaned_efs = []
    y_features_cleaned = 0
    if len(rem) >= 1:
        for f in rem:
            if f not in features:
                y.drop(f, inplace=True, axis=1)
                y_features_cleaned_efs.append(f)
                y_features_cleaned += 1
    if y_features_cleaned > 0:
        if verbose:
            logger.info(
                "Clean_Features_not_in_y {}".format(str(y_features_cleaned_efs))
            )

    if len(X.columns) == 0:
        logger.error(
            "Clean_Features_not_in_X_or_test:transform:X and Y are orthogonal."
        )
        raise PasoError()

    if inplace:
        return None
    else:
        return X, y


############## 10
# code refactored into paso_pkg from pyjanitor
@register_DataFrame_method
def paso_standardize_column_names(*args, **kwargs) -> pd.DataFrame:
    return standardize_column_names(*args, **kwargs)


def _change_case(col: str, case_type: str) -> str:
    """Change case of a column name."""
    case_types = ["preserve", "upper", "lower", "snake"]
    if case_type.lower() not in case_types:
        raise_PasoError(f"case_type must be one of: {case_types}")

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
    return "".join(item for item in str(col_name) if item.isalnum() or "_" in item)


_underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
_underscorer2 = re.compile("([a-z0-9])([A-Z])")


def _camel2snake(col_name: str) -> str:
    """
    Convert camelcase names to snake case.
    Implementation taken from: https://gist.github.com/jaytaylor/3660565
    by @jtaylor
    """

    subbed = _underscorer1.sub(r"\1_\2", col_name)
    return _underscorer2.sub(r"\1_\2", subbed).lower()


FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", "")]


def _normalize_1(col_name: Hashable) -> str:
    result = str(col_name)
    for search, replace in FIXES:
        result = re.sub(search, replace, result)
    return result


def _strip_accents(col_name: str) -> str:
    """
    Removes accents from a DataFrame column name.
    .. _StackOverflow: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
    """  # noqa: E501
    return "".join(
        l
        for l in unicodedata.normalize("NFD", col_name)
        if not unicodedata.combining(l)
    )


def _strip_underscores(
    df: pd.DataFrame, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
    """
    Strip underscores from DataFrames column names.
    Underscores can be stripped from the beginning, end or both.
    .. code-block:: python
        df = _strip_underscores(df, strip_underscores='left')
    df: The pandas DataFrame object.
    strip_underscores: (optional) Removes the outer underscores from all
        column names. Default None keeps outer underscores. Values can be
        either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
        and True.
    :returns: A pandas DataFrame with underscores removed.
    """
    df = df.rename(columns=lambda x: _strip_underscores_func(x, strip_underscores))
    return df


def _strip_underscores_func(
    col: str, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
    """Strip underscores from a string."""
    underscore_options = [None, "left", "right", "both", "l", "r", True]
    if strip_underscores not in underscore_options:
        raise_PasoError(f"strip_underscores must be one of: {underscore_options}")

    if strip_underscores in ["left", "l"]:
        col = col.lstrip("_")
    elif strip_underscores in ["right", "r"]:
        col = col.rstrip("_")
    elif strip_underscores == "both" or strip_underscores is True:
        col = col.strip("_")
    return col


@register_DataFrame_method
def paso_standardize_column_names(
    X: pd.DataFrame,
    strip_underscores: str = None,
    case_type: str = "lower",
    remove_special: bool = True,
    strip_accents: bool = True,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Standardize to pandas column names.

    Takes all column names, converts them to lowercase, then replaces all
    spaces with underscores.

    Parameters:
        X:  dataset

    Keywords:
        inplace:
            True: mutate X, return X
            False: do no change X, return df-stats

        verbose:
            True: output
            False: silent

        strip_underscores:  (optional)
            True: Removes the outer underscores from all
            column names. Default None keeps outer underscores. Values can be
            either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
            and True.

        case_type:  (optional)
            True: Whether to make columns lower or uppercase.
            Current case may be preserved with 'preserve',
            while snake case conversion (from CamelCase or camelCase only)
            can be turned on using "snake".
            Default 'lower' makes all characters lowercase.

        remove_special:  (optional)
            True: Remove special characters from columns.
            Only letters, numbers and underscores are preserved.


    Returns: DataFrame
        X: DataFrame
    """
    _fun_name = paso_standardize_column_names.__name__

    if inplace:
        nX = X
    else:
        nX = X.copy()

    original_column_names = list(nX.columns)

    nX = nX.rename(columns=lambda x: _change_case(x, case_type))

    nX = nX.rename(columns=_normalize_1)

    if remove_special:
        nX = nX.rename(columns=_remove_special)

    if strip_accents:
        nX = nX.rename(columns=_strip_accents)

    nX = nX.rename(columns=lambda x: re.sub("_+", "_", x))
    nX = _strip_underscores(nX, strip_underscores)

    if verbose:
        logger.info("{} done.".format(_fun_name))

    return nX


class Balancers(pasoFunction):
    """
        **paso** supports (currenly) only the imbalanced-learn
        package. This package is very comprehensive with
        examples on how to transform (clean) different types of
        imbalanced class data. Balancing only works continuous data.

      description_file:

      project: HPKinetics/paso #[optional]
        verbose: True  #[optional]
        inplace: True #[optional]
        kind:
          <strategy>:
            description: "SMOTE overbalance all classes"
            genus: Balancer                             #[optional]
            type: sklearn
            kwargs:
                <kwarg-1>

       strategy: (list)
          __Balancers__: each balancer uses and underweight or overweight strategy.
          class global

        Warning:
            Only **SMOTEC** can balance datasets with categorical features. All
            others will accept a dataset only with continuous features.

    """

    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.over_sampling import BorderlineSMOTE, SMOTENC, SVMSMOTE

    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.under_sampling import ClusterCentroids
    from imblearn.under_sampling import (
        NearMiss,
        EditedNearestNeighbours,
        RepeatedEditedNearestNeighbours,
    )
    from imblearn.under_sampling import CondensedNearestNeighbour, OneSidedSelection

    __Balancers__ = {
        "RanOverSample": RandomOverSampler,
        "SMOTE": SMOTE,
        "ADASYN": ADASYN,
        "BorderLineSMOTE": BorderlineSMOTE,
        "SVMSMOTE": SVMSMOTE,
        "SMOTENC": SMOTENC,
        "RandomUnderSample": RandomUnderSampler,
        "ClusterCentroids": ClusterCentroids,
        "NearMiss": NearMiss,
        "EditedNearestNeighbour": EditedNearestNeighbours,
        "RepeatedEditedNearestNeighbours": RepeatedEditedNearestNeighbours,
        "CondensedNearestNeighbour": CondensedNearestNeighbour,
        "OneSidedSelection": OneSidedSelection,
    }

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):
        """

        """

        super().__init__()

    @staticmethod
    def balancers() -> List[str]:
        """
        Parameters:
            None

        Returns: List
            List of available class Balancers ames.
        """
        return list(Balancers.__Balancers__.keys())

    @pasoDecorators.TTWrapXy(array=False)
    def transform(self, X, y, verbose=True, inplace=True):
        """
        Parameters:
            X:  column(s) are independent features of dataset
            y: target or dependent feature of dataset.

        Keywords:
            inplace:
                True: mutate X, return X
                False: do no change X, return df-stats

            verbose:
                True: output
                False: silent

        Returns: ( Balanced)
            X: (DataFrame\) column(s) are independent features of dataset
            y: (numpy vector )  target or dependent feature of dataset.
        """

        # create instance of this particular learner
        # checks for non-optional keyword
        if self.kind_name not in Balancers.__Balancers__:
            raise_PasoError(
                "transform; no Balancer named: {} not in Balancer.__Balancers__: {}".format(
                    self.kind_name, Balancers.__Balancers__.keys()
                )
            )
        else:
            self.model = Balancers.__Balancers__[self.kind_name](
                **self.kind_name_kwargs
            )

        # workaround in case all classes are equal

        uniqueValues, _, occurCount = np.unique(
            y, return_index=True, return_counts=True
        )

        if np.max(occurCount) == np.min(occurCount):
            return X, y

        # workaround in case all classes are equal

        uniqueValues, _, occurCount = np.unique(
            y, return_index=True, return_counts=True
        )

        self.n_class = len(uniqueValues)
        self.class_names = _array_to_string(uniqueValues)

        X_result, y_result = self.model.fit_resample(X.to_numpy(), y)

        X = pd.DataFrame(X_result, columns=X.columns)

        if verbose:
            logger.info("Balancer done")
        self.balanced = True

        return X, y_result


@register_DataFrame_method
def paso_balance(
    X: pd.DataFrame,
    class_instance: Balancers,
    target: str,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    method to transform dataset by imputing NaN values. Encoding and scaling
    and other data-set preprocessing should not be done here.

    Parameters:
        X: dataset

    Keywords:
       class_instance:

       target:

        inplace:
            True: mutate X, return X
            False: do no change X, return X.copy()
            type: any to be laced with np.NaN

        verbose:
            True: output
            False: silent

    Returns: X or X.copy - see inplace

    """
    if inplace:
        nX = X
    else:
        nX = X.copy()

    nX, y = DataFrame_to_Xy(nX, target)

    nX, y = class_instance.transform(nX, y, inplace=True, verbose=verbose)

    return Xy_to_DataFrame(nX, y)


class Augmenters(pasoFunction):
    """
    Currently, **paso** supports class structured data.

    Note:
    """

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):
        """
            Parameters:
                name:
                description_filepath:

            Returns:
                self
        """
        super().__init__()

    @pasoDecorators.TTWrapXy(array=False)
    def transform(
        self, X: pd.DataFrame, y: np.array, **kwargs
    ) -> Union[pd.DataFrame, np.array]:
        """
        Argument data by ratio.
            1. the dataset is class balanced first.
            1. then ratio dataset stub is applied to first class by sampling
            1. this ratio stub is kept as it must be removed later
            1. the stub ia added and the dataset rebalanced
            1. the stub is subtracted
            1. and the dataset is rebalanced.

            the result is a dataset augmented by ratio artificial data for each class,

        Parameters:
            X:  dataset

            ratio: List
                The features (column names) to eliminate.

            inplace: ,
                True: mutate X, return X
                False: do no change X, return df-stats

            verbose:
                True: output
                False: silent

        Returns:
            X: DataFrame
                transformed X DataFrame

        Note:
            Because of integer roundoff. the ratio increased may not be exact.
            Also, ratio of 0.0 or less indicates balance only no augmentation.

        """

        kwa = "ratio"
        self.ratio = _dict_value(kwargs, kwa, 1.0)
        _check_non_optional_kw(
            self.ratio,
            "ratio keyword pair not specified in Balancer:.ratio {}".format(kwargs),
        )
        kwa = "verbose"
        self.verbose = _dict_value(kwargs, kwa, True)
        kwa = "inplace"
        self.inplace = _dict_value(kwargs, kwa, True)

        # create instance of this particular learner
        # checks for non-optional keyword
        if self.kind_name not in Balancers.__Balancers__:
            raise_PasoError(
                "transform; no Augmenter named: {} not in Balancer.__Balancers__: {}".format(
                    self.kind_name, Balancers.__Balancers__.keys()
                )
            )
        else:
            self.model = Balancers.__Balancers__[self.kind_name](
                **self.kind_name_kwargs
            )

        # balance before augmentation
        X, y = Balancers(description_filepath=self.description_filepath).transform(
            X, y, verbose=False
        )
        # 0.0or less indicates balance only no augmentation
        if self.ratio < 0.0:
            logger.warning("ratio lt 0. just returning X,y balanced")
            return X, y
        elif self.ratio > 1.0:
            raise_PasoError("Ratio<= 1.0 was: {}".format(self.ratio))

        # 3
        # calcuate before balancer creates augment data (pusedo data)
        # ratio is the multiplier of max sized class
        # augment will first balance so that 1.0 will result
        # in a size
        target = "TaRgEtx403856789"
        Xy_to_DataFrame(X, y, target)

        self.class_names = _array_to_string(X[target].unique())
        each_class_count = X.groupby([target]).count()
        max_class_sizes_s = each_class_count.iloc[:, 0]
        max_class_size = max_class_sizes_s.max()
        stub_size = int(max_class_size)
        highest_class_arg = max_class_sizes_s.argmax()

        # only sample as ratio can not be bigger than 1.0
        stub = X[X[target] == highest_class_arg].sample(stub_size)
        # 4
        X = X.append(stub)
        X, y = DataFrame_to_Xy(X, target)
        X, y = Balancers(description_filepath=self.description_filepath).transform(
            X, y, verbose=False
        )

        # 5the stub is subtracted
        Xy_to_DataFrame(X, y, target)
        X.drop(X.index[stub.index], axis=0, inplace=True)
        # 6 and the dataset is rebalanced.
        X, y = DataFrame_to_Xy(X, target)
        X, y = Balancers(description_filepath=self.description_filepath).transform(
            X, y, verbose=False
        )
        if self.verbose:
            logger.info("Augmenter ratio: {}".format(self.ratio))
        self.augmented = True
        return X, y


@register_DataFrame_method
def paso_augment(
    X: pd.DataFrame,
    class_instance: Augmenters,
    target: str,
    ratio: float = 1.0,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    method to transform dataset by imputing NaN values. Encoding and scaling
    and other data-set preprocessing should not be done here.

    Parameters:
        X: dataset

    Keywords:
       class_instance:

       target:

        inplace:
            True: mutate X, return X
            False: do no change X, return X.copy()
            type: any to be laced with np.NaN

        verbose:
            True: output
            False: silent

    Returns: X or X.copy - see inplace

    """
    if inplace:
        nX = X
    else:
        nX = X.copy()

    nX, y = DataFrame_to_Xy(nX, target)

    nX, y = class_instance.transform(nX, y, ratio=ratio, inplace=True, verbose=verbose)

    return Xy_to_DataFrame(nX, y, target)


