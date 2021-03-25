# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
__coverage__ = 0.53

import pandas as pd
import numpy as np
import pytest
import scipy

import warnings

warnings.filterwarnings("ignore")

from pandas.core.dtypes.generic import ABCDataFrame, ABCIndexClass, ABCSeries

from tqdm import tqdm

#

from janitor.mlsd.util import raise_Photonai_Error, Photonai_Error
from janitor.mlsd.util import _add_target_to_df, _strip_off_target
from janitor.mlsd.util import _must_be_list_tuple_int_float_str
from janitor.mlsd.util import _Check_No_NA_F_Values, isDataFrame, isSeries
#

#1

@pytest.mark.mlsd
def test_Class_PhotonError():
    with pytest.raises(Photonai_Error):
        raise_Photonai_Error('bad_error_test')
#2
@pytest.mark.mlsd
def test_add_target_to_df(City: pd.DataFrame)-> None:
    y = City["MEDV"].values
    newCity = _add_target_to_df(City, y)
    assert newCity['target'].unique().all() ==  City["MEDV"].unique().all()
#3
@pytest.mark.mlsd
def test_strip_off_target(City: pd.DataFrame)-> None:
    yp = City["MEDV"].values
    Cityp = City.drop(columns='MEDV', inplace=False)
    newCity, y = _strip_off_target(City, 'MEDV')
    assert set(newCity.columns) == set(Cityp.columns) and (y.any() == yp.any())
#4 list
@pytest.mark.mlsd
def test_must_be_list()-> None:
    x = [1,2]
    assert _must_be_list_tuple_int_float_str(x) == x

#5 set
@pytest.mark.mlsd
def test_must_be_lis_bad2():
    with pytest.raises(TypeError):
        x = set[1, 2]
        assert _must_be_list_tuple_int_float_str(x) == x
#6 tuple
@pytest.mark.mlsd
def test_must_be_tuple()-> None:
    x = (1,2)
    assert _must_be_list_tuple_int_float_str(x) == x
#7 int
@pytest.mark.mlsd
def test_must_be_int_float_str3()-> None:
    x = 999
    assert _must_be_list_tuple_int_float_str(x) == x
#8 float
@pytest.mark.mlsd
def test_must_be_float()-> None:
    x = 999.9
    assert _must_be_list_tuple_int_float_str(x) == x

#9 str
@pytest.mark.mlsd
def test_must_be_str()-> None:
    x = 'astring'
    assert _must_be_list_tuple_int_float_str(x) == x
#10 DataFrame error
@pytest.mark.mlsd
def test_must_be_BadArg()-> None:
    with pytest.raises(Photonai_Error):
        x = pd.DataFrame()
        assert _must_be_list_tuple_int_float_str(x) == x
#11 nparray error
@pytest.mark.mlsd
def test_must_be_BadArg2()-> None:
    with pytest.raises(Photonai_Error):
        x = np.ndarray([2,2])
        assert _must_be_list_tuple_int_float_str(x) == x
#12
@pytest.mark.mlsd
def test_Check_No_NA_F_Values(City: pd.DataFrame)-> None:
    assert _Check_No_NA_F_Values(City,'MEDV')
#13
@pytest.mark.mlsd
def test_Check_No_NA_F_Values_bad(City: pd.DataFrame)-> None:
    with pytest.raises(KeyError):
        assert _Check_No_NA_F_Values(City,'fred')
#14
@pytest.mark.mlsd
def test_Check_No_NA_F_Values_bad2(City: pd.DataFrame)-> None:
    with pytest.raises(Photonai_Error):
        City.loc[1,'MEDV']= None
        assert _Check_No_NA_F_Values(City,'MEDV')
#15
@pytest.mark.mlsd
def test_isDataFrame(City: pd.DataFrame)-> None:
    assert isDataFrame(City)

#16
@pytest.mark.mlsd
def test_isSeries_bad(City: pd.DataFrame)-> None:
    with pytest.raises(AssertionError):
        assert isSeries(City)