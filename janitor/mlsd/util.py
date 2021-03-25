#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
__coverage__ = 0.98

# todo:  enable checkpoint
# todo: support RAPID in parm file
# todo: port to swift


from typing import Dict, List
from loguru import logger

import numpy as np
import pandas as pd
from pandas.core.dtypes.generic import ABCDataFrame
from numba import jit

#
import warnings

warnings.filterwarnings("ignore")


class janitor_Error(Exception):
    pass


def raise_janitor_Error(msg):
    logger.error(msg)
    raise Photonai_Error(msg)


####  for interna use


def _add_target_to_df(
    X: pd.DataFrame, y: np.ndarray, target: str = "target"
) -> pd.DataFrame:
    """
        Tranform X(DataFrame and y (numpy vector) into one DataFame X

        Note: Memory is saved as y added to X dataFrame in place.

    Parameters:

        X:
        y:
        target:

    Return: X DataFrame
    """
    X[target] = y
    return X


def _strip_off_target(
    X: pd.DataFrame, target: str = "target"
) -> (pd.DataFrame, np.ndarray):
    """
    Strips off column 'target'as  y as a numpy array from dataframe X. X remains dataframe
    Parameters:
        X:
        target:

    Return: list pair X,Y
    """
    y = X[target].values
    X = X[X.columns.difference([target])]
    return X, y


def _must_be_list_tuple_int_float_str(attribute):
    if isinstance(attribute, (tuple, list)):
        return attribute
    elif isinstance(attribute, (str, int, float)):
        return attribute
    else:
        raise_Photonai_Error(
            "{} must be tiple,list, str, float, or int. unsupported type: {}".format(
                attribute, type(attribute)
            )
        )


def isSeries(X) -> bool:
    """
    Parameters:
        X (any type)

    Returns:
        True (boolean) if DataFrame Series type.
        False (boolean) otherwise
    """
    return isinstance(X, pd.core.series.Series)


def isDataFrame(X) -> bool:
    """
    Parameters:
        X (any type)

    Returns:
        True (boolean) if DataFrame type.
        False (boolean) otherwise
    """
    return isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series))


# must be dataFrame or series
def _Check_is_DataFrame(X) -> List:
    if isDataFrame(X):
        return True
    else:
        raise_Photonai_Error(
            "TransformWrap:Xarg must be if type DataFrame. Was type:{}".format(
                str(type(X))
            )
        )


def _Check_No_NA_F_Values(df: pd.DataFrame, feature: str) -> bool:
    if not df[feature].isna().any():
        return True
    else:
        raise_Photonai_Error("Passed dataset, DataFrame, contained NA")


def _Check_No_NA_Series_Values(ds: pd.DataFrame):
    if not ds.isna().any():
        return True
    else:
        raise_Photonai_Error("Passed dataset, DataFrame, contained NA")


def _Check_No_NA_Values(df: pd.DataFrame):
    for feature in df.columns:
        if _Check_No_NA_F_Values(df, feature):
            pass


@jit
def _float_range(start, stop, step):
    istop = int((stop - start) / step)
    edges = []
    for i in range(int(istop) + 1):
        edges.append(start + i * step)
    return edges


# @jit  CANT DO IT, X IS DATAFRAME
def _fixed_width_labels(X, nbins, miny, maxy):
    # preparation of fixed-width bins

    edges = _float_range(miny, maxy, (maxy - miny) / nbins)
    lf = 0 if miny == 0 else round(abs(math.log10(abs(miny))))
    loglf = 0 if maxy == 0 else math.log10(abs(maxy / nbins))
    hf = round(abs(loglf))
    if loglf > 0:
        fs = "(%2." + str(0) + "f, %2." + str(0) + "f]"
    else:
        ff = lf + 1 if (lf > hf) else hf + 1
        fs = "(%2." + str(ff) + "f, %2." + str(ff) + "f]"

    lbl = np.array(
        [fs % (edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    )
    return lbl


### pasoDecorators class
# adapted from pandas-flavor 11/13/2019
from pandas.api.extensions import register_dataframe_accessor
from functools import wraps


def register_DataFrame_method(method):
    """Register a function as a method attached to the Pandas DataFrame.
    Example
    -------
    for a function
        @pf.register_dataframe_method
        def row_by_value(df, col, value):
        return df[df[col] == value].squeeze()

    for a class method
        @pf.register_dataframe_accessor('Aclass')
        class Aclass(object):

        def __init__(self, data):
        self._data

        def row_by_value(self, col, value):
            return self._data[self._data[col] == value].squeeze()
    """

    def inner(*args, **kwargs):
        class AccessorMethod(object):
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        register_dataframe_accessor(method.__name__)(AccessorMethod)
        return method

    return inner()


#####no tests


@jit
def _divide_dict(d: Dict, n: np.int_) -> Dict:
    return {k: d[k] / n for k in d}


# @jit
def _mean_arrays_in_dict(d: Dict) -> Dict:
    return {k: np.mean(d[k]) for k in d}


# @jit
def _median_arrays_in_dict(d: Dict) -> Dict:
    return {k: np.median(d[k]) for k in d}


# @jit
def _std_arrays_in_dict(d: Dict) -> Dict:
    return {k: np.std(d[k]) for k in d}


# @jit
def _var_arrays_in_dict(d: Dict) -> Dict:
    return {k: np.var(d[k]) for k in d}


# @jit
def _stat_arrays_in_dict(d: Dict) -> Dict:
    """
    Determines statistics, mean, median, std, var of an dict of arrays.

    parameters:
        d: Dict ,where k is str, v is array of values

    :return:
        d:dict statistics of eah key'array
    """
    return {
        "mean": _mean_arrays_in_dict(d),
        "median": _median_arrays_in_dict(d),
        "std": _std_arrays_in_dict(d),
        "var": _var_arrays_in_dict(d),
    }


def _merge_dicts(d1: Dict, d2: Dict) -> Dict:
    return {**d2, **d1}


def _add_dicts(d1: Dict, d2: Dict) -> Dict:
    for k in d1.keys():
        if k in d2.keys():
            d1[k] = d1[k] + d2[k]
    return {**d2, **d1}


def _array_to_string(array: np.ndarray) -> List:
    return [str(item) for item in array]


def _new_feature_names(X: pd.DataFrame, names: List) -> pd.DataFrame:
    """
    Change feature nanmes of a dataframe to names.

    Parameters:
        X: Dataframe
        names: list,str

    Returns: X

    """
    # X is inplace because only changing column names
    if names == []:
        return X
    c = list(X.columns)
    if isinstance(names, (list, tuple)):
        c[0 : len(labels)] = labels
    else:
        c[0:1] = [labels]
    X.columns = c
    return X


def set_modelDict_value(v, at):
    """
    replaced by _dict_value
    """
    if at not in object_.modelDict.keys():
        object_.modelDict[at] = v


def _exists_as_dict_value(dictnary: Dict, key: str) -> Dict:
    """
    used to variable to dict-value or default.
    if key in dict return key-value
    else return default.

    """
    if key in dictnary:
        if isinstance(dictnary[key], dict):
            return dictnary[key]
        else:
            raise_Photonai_Error("{} is NOt of type dictionary".format(key))
    else:
        logger.warning("{} is no in dictionary:({})".format(key, dictnary))
        return {}


def _dict_value(dictnary: Dict, key: str, default):
    """
    used to variable to dict-value or default.
    if key in dict return key-value
    else return default.

    """
    if key in dictnary:
        return dictnary[key]
    else:
        return default


def _dict_value2(fdictnary: Dict, dictnary: Dict, key: str, default):
    """
    used to variable to dict or fdict (2nd dict) value or default.
    if key in dict or fdict return key-value
    else return default.

    if in both, fdict is given precedent

    """
    result = default
    if key in dictnary:
        result = dictnary[key]
    if key in fdictnary:
        result = fdictnary[key]  # precedence to ontoloical file
    return result


def _exists_Attribute(object_, attribute_string) -> str:
    return hasattr(object_, attribute_string)


def _exists_key(dictnary: Dict, key: str, error: bool = True):
    if key in dictnary:
        return dictnary[key]
    else:
        if error:
            raise_Photonai_Error(
                "{} not specified through keyword call or description file for: {}".format(
                    attribute, self.name
                )
            )
        else:
            logger.warning(
                "{} not specified. very likely NOT error): {}".format(
                    attribute, self.name
                )
            )
            return False
