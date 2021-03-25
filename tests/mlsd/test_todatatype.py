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

# photonai imports
from janitor.mlsd.util import raise_janitor_Error

from janitor.mlsd.util import janitor_Error

from janitor.mlsd.todatatype import toDataFrame, toCategory
from janitor.mlsd.todatatype import toContinuousCategory
from janitor.mlsd.todatatype import toDatetimeComponents


@pytest.mark.mlsd
def test_Class_init_BadArg():
    with pytest.raises(TypeError):
        toDataFrame(fred=1)


# 2


@pytest.mark.mlsd
def test_2d_numpy_columns_(X, cn):
    assert toDataFrame(X, columns=cn).columns[1] == "k"


# 2b


@pytest.mark.mlsd
def test_2d_numpy_columns_method_error(X, cn):
    with pytest.raises(AttributeError):
        assert X.toDataFrame(columns=cn).columns[1] == "k"


# 2c kinky


@pytest.mark.mlsd
@pytest.mark.mlsd
def test_2d_numpy_columns_method_(X, cn):
    assert toDataFrame(X, columns=cn).toDataFrame().columns[1] == "k"


# 3
@pytest.mark.mlsd
def test_2d_numpy_columns_default(X, cn):
    assert toDataFrame(X).columns[1] == "c_1"


# 4
@pytest.mark.mlsd
def test_1d_numpy_bad_arg(y, cn):
    with pytest.raises(janitor_Error):
        toDataFrame(y)


# 5
@pytest.mark.mlsd
def test_2d_named_too_many_args(X, cnv):
    with pytest.raises(TypeError):
        toDataFrame(X, cnv, cnv, X, 5, 6).columns[2] == cnv[2]


# 6b
@pytest.mark.mlsd
def test_2d_named_kinky(X, cnv):
    assert toDataFrame(X, columns=cnv).toDataFrame().columns[2] == cnv[2]


# edge tests
@pytest.mark.mlsd
def test_Blank_Arg_Transform():
    with pytest.raises(TypeError):
        toDataFrame()


# 8
@pytest.mark.mlsd
def test_Empty_list_Arg_Transform():
    with pytest.raises(janitor_Error):
        toDataFrame([])


# 9
@pytest.mark.mlsd
def test_3dArray_numpy(X):
    with pytest.raises(ValueError):
        toDataFrame(np.reshape(X, (-1, -1, 1)))


# 10
@pytest.mark.mlsd
def test_df_columns_set(X, cnv):
    assert (
        toDataFrame(pd.DataFrame(X, columns=cnv), columns=cnv).columns == cnv
    ).all()


# 11
@pytest.mark.mlsd
def test_toDataFrame_save_not_implemented():
    with pytest.raises(AttributeError):
        toDataFrame.save("nofilepath")


# 12
@pytest.mark.mlsd
def test_toDataFrame_not_class_no_load():
    with pytest.raises(AttributeError):
        toDataFrame.load("a foobar")


# 13
@pytest.mark.mlsd
def test_toDataFrame_write_error():
    with pytest.raises(AttributeError):
        toDataFrame.write("nofilepath")


# 14
@pytest.mark.mlsd
def test_toDataFrame_list():
    assert len(toDataFrame([1, 2, 3], columns=[]).columns) == 1


# 15
@pytest.mark.mlsd
def test_toDataFrame_list_1d():
    assert len(toDataFrame([[1, 2, 3]]).columns) == 3


# 16
@pytest.mark.mlsd
def test_toDataFrame_list_2d():
    assert toDataFrame([[1, 2, 3], [4, 5, 5]]).shape == (2, 3)


# 17
@pytest.mark.mlsd
def test_toDataFrame_array():
    r = 100
    c = 10
    arr = np.ndarray(shape=(r, c), dtype=float, order="F")
    assert toDataFrame(arr).shape == (r, c)


# 18
@pytest.mark.mlsd
def test_toDataFrame_array3d_error():
    r = 100
    c = 10
    arr = np.ndarray(shape=(2, r, c), dtype=float, order="F")
    with pytest.raises(janitor_Error):
        assert toDataFrame(arr, columns=[]).shape == (r, c)


# 19
@pytest.mark.mlsd
def test_toDataFrame_series():
    r = 100
    c = 1
    arr = np.ndarray(shape=(r), dtype=float, order="F")
    s = pd.Series(arr)
    assert toDataFrame(s, columns=[]).shape == (r, c)


# 20
@pytest.mark.mlsd
def test_toDataFrame_csr():
    r = 3
    c = 3
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    csr = scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(3, 3)
    ).todense()
    assert toDataFrame(csr, columns=[]).shape == (r, c)


# 21


@pytest.mark.mlsd
def test_toDataFrame_csr_kinky():
    r = 3
    c = 3
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    csr = scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(3, 3)
    ).todense()
    assert toDataFrame(csr, columns=[]).toDataFrame().shape == (r, c)


#####

# 1
@pytest.mark.mlsd
def test_toCategory_no_passed_arg_type_error_none():
    with pytest.raises(TypeError):
        toCategory()


# 2
@pytest.mark.mlsd
def test_toCategory_passed_arg_type_error0():
    with pytest.raises(AttributeError):
        toCategory(0)


# 3
@pytest.mark.mlsd
def test_toCategory_City_ignore_float(City):
    assert (toCategory(City) == City).all().all()


# 3b
@pytest.mark.mlsd
def test_toCategory_City_ignore_float_method(City):
    assert (City.toCategory() == City).all().all()


# 4
@pytest.mark.mlsd
def test_toCategory_City_ignore_integer(City):
    dfi = City.astype("int")
    assert (
        (toCategory(dfi, int_=False, bool_=False, object_=False) == dfi)
        .all()
        .all()
    )


# 4b
@pytest.mark.mlsd
def test_toCategory_City_ignore_integer_meyhto(City):
    dfi = City.astype("int")
    assert (
        (dfi.toCategory(int_=False, bool_=False, object_=False) == dfi)
        .all()
        .all()
    )


# 5
@pytest.mark.mlsd
def test_toCategory__City_int_default(City):
    dfi = City.astype("int")
    assert (toCategory(dfi) == dfi.astype("category")).all().all()


# 5b
@pytest.mark.mlsd
def test_toCategory_City_default_int_method(City):
    dfi = City.astype("int")
    assert (dfi.toCategory() == dfi.astype("category")).all().all()


# 6
@pytest.mark.mlsd
def test_toCategory_City_integer_column_names(City):
    dfi = City.astype("int")
    assert (toCategory(dfi).columns == dfi.astype("category").columns).all()


# 6b
@pytest.mark.mlsd
def test_toCategory_City_integer_column_names_method(City):
    dfi = City.astype("int")
    assert (dfi.toCategory().columns == dfi.astype("category").columns).all()


# 7
@pytest.mark.mlsd
def test_toCategory_Xobect_ignore_object(Xobject):
    assert (
        (toCategory(Xobject, int_=True, bool_=True, object_=False) == Xobject)
        .all()
        .all()
    )


# 7 b
@pytest.mark.mlsd
def test_toCategory_Xobect_ignore_object_method(Xobject):
    assert (
        (Xobject.toCategory(int_=True, bool_=True, object_=False) == Xobject)
        .all()
        .all()
    )


# 8
@pytest.mark.mlsd
def test_toCategory_Xobject_not_datetime_format(Xobject):
    assert (toCategory(Xobject) == Xobject.astype("category")).all().all()


# 9
@pytest.mark.mlsd
def test_toCategory_Xboolean_ignore_bool(Xboolean):
    assert (
        (
            toCategory(Xboolean, int_=False, bool_=False, object_=False)
            == Xboolean
        )
        .all()
        .all()
    )


# 10
@pytest.mark.mlsd
def test_toCategory_Xboolean(Xboolean):
    assert (toCategory(Xboolean) == Xboolean.astype("category")).all().all()


# 11
@pytest.mark.mlsd
def test_toCategory_df_internet_traffic_ignore_object_and_integer(
    df_internet_traffic,
):
    assert (
        (
            toCategory(
                df_internet_traffic, int_=False, bool_=True, object_=False
            )
            == df_internet_traffic
        )
        .all()
        .all()
    )


# 12
@pytest.mark.mlsd
def test_toCategory_df_internet_traffic_datetime_and_integer(
    df_internet_traffic,
):
    assert (
        (
            toCategory(df_internet_traffic)
            == df_internet_traffic.astype("category")
        )
        .all()
        .all()
    )


# 13
@pytest.mark.mlsd
def test_toCategory_df_internet_traffic_ignore_datetime_and_integer(
    df_internet_traffic,
):
    assert (
        (
            toCategory(
                df_internet_traffic, int_=False, bool_=False, object_=False
            )
            == df_internet_traffic
        )
        .all()
        .all()
    )


# 14
@pytest.mark.mlsd
def test_toCategory_passed_no_save(City):
    with pytest.raises(AttributeError):
        toCategory.save("foo")


# 15
@pytest.mark.mlsd
def test_toCategory_passed_no_load(City):
    with pytest.raises(AttributeError):
        toCategory.load("foo")


# 16
@pytest.mark.mlsd
def test_toCategory_Xobject_not_datetime_formats_method(Xobject):
    assert (Xobject.toCategory() == Xobject.astype("category")).all().all()


# 17
@pytest.mark.mlsd
def test_toCategory_Xboolean_ignore_bools_method(Xboolean):
    assert (
        (
            Xboolean.toCategory(int_=False, bool_=False, object_=False)
            == Xboolean
        )
        .all()
        .all()
    )


# 18
@pytest.mark.mlsd
def test_toCategory_Xbooleans_method(Xboolean):
    assert (Xboolean.toCategory() == Xboolean.astype("category")).all().all()


# 19
@pytest.mark.mlsd
def test_toCategory_df_internet_traffic_ignore_object_and_integers_method(
    df_internet_traffic,
):
    assert (
        (
            df_internet_traffic.toCategory(
                int_=False, bool_=True, object_=False
            )
            == df_internet_traffic
        )
        .all()
        .all()
    )


# 20
@pytest.mark.mlsd
def test_toCategory_df_internet_traffic_datetime_and_integers_method(
    df_internet_traffic,
):
    assert (
        (
            df_internet_traffic.toCategory()
            == df_internet_traffic.astype("category")
        )
        .all()
        .all()
    )


# 21
@pytest.mark.mlsd
def test_toCategory_df_internet_traffic_ignore_datetime_and_integers_method(
    df_internet_traffic,
):
    assert (
        (
            df_internet_traffic.toCategory(
                int_=False, bool_=False, object_=False
            )
            == df_internet_traffic
        )
        .all()
        .all()
    )


#######

# 1
@pytest.mark.mlsd
def test_df_toDatetimeComponents_WrongArgType():
    with pytest.raises(janitor_Error):
        toDatetimeComponents([1, 2, 3])


# 2
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_All_Componentss_drop_Date_True_Default(
    df_small_no_NA, df_small_NFeatures, NComponentFeatures
):
    dt = df_small_no_NA.copy()
    assert (
        toDatetimeComponents(dt).shape[1]
        == (df_small_NFeatures - 4) * NComponentFeatures
        + df_small_NFeatures
        - 5
    )


# 3
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_All_Components_drop_Date_False(
    df_small_no_NA, df_small_NFeatures, NComponentFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        toDatetimeComponents(dt, drop=False).shape[1]
        == (df_small_NFeatures - 4) * NComponentFeatures + df_small_NFeatures
    )


# 4
@pytest.mark.mlsd
def test_toDatetimeComponents_with_NA_ValueError(df_small_NA):

    with pytest.raises(janitor_Error):
        toDatetimeComponents(df_small_NA)


# 5
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_Year_Component_drop_Date_True_Default(
    df_small, df_small_NFeatures
):

    dt = df_small.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    with pytest.raises(janitor_Error):
        toDatetimeComponents(dt, components=["Year"])


# 6
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_Year_Component_drop_Date_True_Elapsed_Default(
    df_small_no_NA, df_small_NFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        toDatetimeComponents(
            dt, drop=True, components=["Year", "Elapsed"]
        ).shape[1]
        == (df_small_NFeatures - 4) * 2 + df_small_NFeatures - 5
    )


# 7
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_Year_Component_drop_Date_False(
    df_small_no_NA, df_small_NFeatures
):
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )

    assert (
        toDatetimeComponents(dt, drop=False, components=["Year"]).shape[1]
        == (df_small_NFeatures - 4) * 1 + df_small_NFeatures
    )


# 8
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_DoY_Elapsed_IME_Components_drop_Date_True(
    df_small_no_NA, df_small_NFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        toDatetimeComponents(
            dt, components=["Dayofyear", "Elapsed", "Is_month_end"]
        ).shape[1]
        == (df_small_NFeatures - 4) * 3 + df_small_NFeatures - 5
    )


# 9
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_DoY_Elapsed_IME_Components_drop_Date_False(
    df_small_no_NA, df_small_NFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        toDatetimeComponents(
            dt, drop=False, components=["DayofYear", "Elapsed", "Is_month_end"]
        ).shape[1]
        == (df_small_NFeatures - 4) * 3 + df_small_NFeatures
    )


# 10
@pytest.mark.mlsd
def test_toDatetimeComponents_nternet_traffic_datetime(df_internet_traffic):
    dt = df_internet_traffic.copy()
    dt["date"] = pd.to_datetime(
        df_internet_traffic["date"],
        exact=True,
        errors="ignore",
        infer_datetime_format=True,
    )
    assert (
        toDatetimeComponents(dt, drop=True).columns
        == [
            "bit",
            "byte",
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Elapsed",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
    )


# 11
@pytest.mark.mlsd
def test_toDatetimeComponents_internet_traffic_datetime_no_drop(
    df_internet_traffic,
):
    dt = df_internet_traffic.copy()
    dt["date"] = pd.to_datetime(
        df_internet_traffic["date"],
        exact=True,
        errors="ignore",
        infer_datetime_format=True,
    )

    assert (
        toDatetimeComponents(dt, drop=False).columns
        == [
            "date",
            "bit",
            "byte",
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Elapsed",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
    )


# 12
@pytest.mark.mlsd
def test_df_toDatetimeComponents_WrongArgType_method():
    with pytest.raises(AttributeError):
        [1, 2, 3].toDatetimeComponents()


# 13
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_All_Componentss_drop_Date_True_Default_method(
    df_small_no_NA, df_small_NFeatures, NComponentFeatures
):
    dt = df_small_no_NA.copy()
    assert (
        dt.toDatetimeComponents().shape[1]
        == (df_small_NFeatures - 4) * NComponentFeatures
        + df_small_NFeatures
        - 5
    )


# 14
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_All_Components_drop_Date_False_method(
    df_small_no_NA, df_small_NFeatures, NComponentFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        dt.toDatetimeComponents(drop=False).shape[1]
        == (df_small_NFeatures - 4) * NComponentFeatures + df_small_NFeatures
    )


# 15
@pytest.mark.mlsd
def test_toDatetimeComponents_with_NA_ValueError_method(df_small_NA):

    with pytest.raises(janitor_Error):
        df_small_NA.toDatetimeComponents()


# 16
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_Year_Component_drop_Date_True_Default_method(
    df_small, df_small_NFeatures
):

    dt = df_small.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    with pytest.raises(janitor_Error):
        dt.toDatetimeComponents(components=["Year"])


# 17
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_Year_Component_drop_Date_True_Elapsed_Default_method(
    df_small_no_NA, df_small_NFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        dt.toDatetimeComponents(
            drop=True, components=["Year", "Elapsed"]
        ).shape[1]
        == (df_small_NFeatures - 4) * 2 + df_small_NFeatures - 5
    )


# 18
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_Year_Component_drop_Date_False_method(
    df_small_no_NA, df_small_NFeatures
):
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )

    assert (
        dt.toDatetimeComponents(drop=False, components=["Year"]).shape[1]
        == (df_small_NFeatures - 4) * 1 + df_small_NFeatures
    )


# 19
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_DoY_Elapsed_IME_Components_drop_Date_True_method(
    df_small_no_NA, df_small_NFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        toDatetimeComponents(
            dt, components=["Dayofyear", "Elapsed", "Is_month_end"]
        ).shape[1]
        == (df_small_NFeatures - 4) * 3 + df_small_NFeatures - 5
    )


# 20
@pytest.mark.mlsd
def test_toDatetimeComponents_passed_DoY_Elapsed_IME_Components_drop_Date_False_method(
    df_small_no_NA, df_small_NFeatures
):

    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature],
            exact=True,
            errors="ignore",
            infer_datetime_format=True,
        )
    assert (
        dt.toDatetimeComponents(
            drop=False, components=["DayofYear", "Elapsed", "Is_month_end"]
        ).shape[1]
        == (df_small_NFeatures - 4) * 3 + df_small_NFeatures
    )


# 21
@pytest.mark.mlsd
def test_toDatetimeComponents_nternet_traffic_datetime_method(
    df_internet_traffic,
):
    dt = df_internet_traffic.copy()
    dt["date"] = pd.to_datetime(
        df_internet_traffic["date"],
        exact=True,
        errors="ignore",
        infer_datetime_format=True,
    )
    assert (
        dt.toDatetimeComponents(drop=True).columns
        == [
            "bit",
            "byte",
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Elapsed",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
    )


# 22
@pytest.mark.mlsd
def test_toDatetimeComponents_internet_traffic_datetime_no_drop_method(
    df_internet_traffic,
):
    dt = df_internet_traffic.copy()
    dt["date"] = pd.to_datetime(
        df_internet_traffic["date"],
        exact=True,
        errors="ignore",
        infer_datetime_format=True,
    )

    assert (
        dt.toDatetimeComponents(drop=False).columns
        == [
            "date",
            "bit",
            "byte",
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Elapsed",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
    )


# @#
# 22
@pytest.mark.mlsd
def test_toDatetimeComponents_make_4feature():
    from datetime import datetime, timedelta

    darr = np.arange(
        datetime(1751, 7, 1), datetime(2015, 7, 1), timedelta(days=1)
    ).reshape(24106, 4)
    rc = 24106
    cc = 52
    assert toDataFrame(darr, columns=[]).toDatetimeComponents(
        inplace=False
    ).shape == (
        rc,
        cc,
    )


# 23


@pytest.mark.mlsd
def test_toDatetimeComponents_make_1feature():
    from datetime import datetime, timedelta

    rc = 94598
    cc = 1
    acc = 13
    darr = np.arange(
        datetime(1751, 7, 1), datetime(2010, 7, 1), timedelta(days=1)
    ).reshape(rc, cc)
    assert toDataFrame(darr, columns=[]).toDatetimeComponents(
        inplace=False
    ).shape == (
        rc,
        acc,
    )


###############

# 1
@pytest.mark.mlsd
def test_ContinuoustoCategory_no_passed_arg_type_error(City):
    with pytest.raises(TypeError):
        toContinuousCategory()


# 2
@pytest.mark.mlsd
def test_ContinuoustoCategory_passed_arg_type_error(City):
    with pytest.raises(AttributeError):
        toContinuousCategory(0)


# 3
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_float(City):
    CityC = City.copy()
    beforelen = 2 * len(CityC.columns)
    assert (
        len(CityC.toContinuousCategory(drop=False, inplace=True).columns)
        == beforelen
    )


# 4
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_ignore_integer(City):
    dfi = City.astype("int")
    assert (
        dfi.toContinuousCategory(drop=False, int_=False).columns == dfi.columns
    ).all()


# 4b
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_ignore_integer_inplace_True(City):
    dfi = City.astype("int")
    assert (
        (dfi.toContinuousCategory(int_=False, inplace=True) == dfi).all().all()
    )


# 4c
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_ignore_integer_inplace_Drop_True(City):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        (dfi.toContinuousCategory(int_=False, inplace=True, drop=True) == dfi)
        .all()
        .all()
    )


# 5
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_integer(City):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        (
            dfi.toContinuousCategory(inplace=True, drop=True)
            == dfi.astype("category")
        )
        .all()
        .all()
    )


# 5b
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_integer_inplace_True(City):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        (
            dfi.toContinuousCategory(inplace=True, drop=False)
            == dfi.astype("category")
        )
        .all()
        .all()
    )


# 5c
@pytest.mark.mlsd
def test_ContinuoustoCategory__City_integer_inplace_Trueno_NOX_scale(City):
    dfi = City.copy().astype("int")
    assert (
        len(dfi.toContinuousCategory(inplace=True, drop=False).columns) == 28
    )


# 5d
@pytest.mark.mlsd
def test_ContinuoustoCategory__City_integer_inplace_Trueno_NOX_scale(City):
    dfi = City.copy().astype("int")
    assert len(dfi.toContinuousCategory(inplace=True, drop=True).columns) == 14


# 6
@pytest.mark.mlsd
def test_ContinuoustoCategory_tCity_integer_column_names(City):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        dfi.toContinuousCategory(float_=False, inplace=True).columns
        == dfi.astype("category").columns
    ).all()


# 7
@pytest.mark.mlsd
def test_ContinuoustoCategory_tXobect_ignore_object(Xobject):
    assert (
        (
            Xobject.toContinuousCategory(float_=True, int_=False, inplace=True)
            == Xobject
        )
        .all()
        .all()
    )


# 8
@pytest.mark.mlsd
def test_ContinuoustoCategory_Xobject_not_datetime_format(Xobject):
    # #    print(Xobject)
    # #    print('\n')
    # #    print(Xobject.toContinuousCategory(inplace=True,drop=False))
    # #    print('\n')
    assert len(Xobject.toContinuousCategory(inplace=True).columns) == 0


# 8b
@pytest.mark.mlsd
def test_ContinuoustoCategory_Xobject_not_datetime_format(Xobject):
    assert (
        len(Xobject.toContinuousCategory(inplace=True, drop=False).columns)
        == 3
    )


# 9
@pytest.mark.mlsd
def test_ContinuoustoCategory_Xboolean_drop(Xboolean):
    assert (
        len(Xboolean.toContinuousCategory(inplace=True, drop=True).columns)
        == 0
    )


# 10
@pytest.mark.mlsd
def test_ContinuoustoCategory_Xboolean(Xboolean):
    assert (
        len(Xboolean.toContinuousCategory(inplace=True, drop=False).columns)
        == 3
    )


# 10b
@pytest.mark.mlsd
def test_ContinuoustoCategory_tXboolea_inplace_false(Xboolean):
    assert (
        len(Xboolean.toContinuousCategory(inplace=False, drop=False).columns)
        == 3
    )


# 11
@pytest.mark.mlsd
def test_ContinuoustoCategory_tdf_internet_traffic_ignore_object_and_integer(
    df_internet_traffic,
):
    assert (
        (
            df_internet_traffic.toContinuousCategory(inplace=True, drop=False)
            == df_internet_traffic
        )
        .all()
        .all()
    )


# 12
@pytest.mark.mlsd
def test_ContinuoustoCategory_df_internet_traffic(df_internet_traffic):
    with pytest.raises(ValueError):
        assert df_internet_traffic.toContinuousCategory(
            inplace=True, drop=True
        ) == df_internet_traffic.astype("category")


# 12b
@pytest.mark.mlsd
def test_ContinuoustoCategory_internet_traffic_drop(df_internet_traffic):
    assert (
        len(
            df_internet_traffic.toContinuousCategory(
                inplace=True, drop=True
            ).columns
        )
        == 2
    )


# 13
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_ignore_integer(City):
    assert (
        (City.toContinuousCategory(int_=False, inplace=True) == City)
        .all()
        .all()
    )


# 14
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_integer_column_names_not_inplace_drop_false(
    City,
):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        len(dfi.toContinuousCategory(inplace=False, drop=False).columns) == 28
    )


# 15
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_integer_column_names_not_inplace_drop_True(
    City,
):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        len(dfi.toContinuousCategory(inplace=False, drop=True).columns) == 14
    )


# 16
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_integer_column_names_drop_def_True(City):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert len(dfi.toContinuousCategory(inplace=False).columns) == len(
        dfi.astype("category").columns
    )


# 17
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_integer_column_names_inplace_def_true(City):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        dfi.toContinuousCategory(inplace=True).columns
        == dfi.astype("category").columns
    ).all()


# 18
@pytest.mark.mlsd
def test_ContinuoustoCategory_chained_float_unique_1_ERROR():
    farr = np.ndarray(shape=(1000000, 4), dtype=float, order="F")
    assert (
        len(
            toDataFrame(farr, columns=[])
            .toContinuousCategory(drop=False)
            .columns
        )
        == 8
    )


@pytest.mark.mlsd
def test_ContinuoustoCategory_chained_float_quantile():
    nc = 6
    nr = 1000000
    delta = 0.1
    farr = np.arange(0, (nr * delta * nc), delta, dtype=float).reshape(nr, nc)
    toDataFrame(farr, columns=[]).toContinuousCategory(drop=False)


# 20
@pytest.mark.mlsd
def test_ContinuoustoCategory_chained_float_fixed():
    nc = 6
    nr = 1000000
    delta = 0.1
    farr = np.arange(0, (nr * delta * nc), delta, dtype=float).reshape(nr, nc)
    toDataFrame(farr, columns=[]).toContinuousCategory(
        quantile=False, nbin=3, drop=False
    )


# 21
@pytest.mark.mlsd
def test_ContinuoustoCategory_City_integer_column_names_inplace_def_true(City):
    CityC = City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert dfi.toContinuousCategory(inplace=False).iloc[0, 13] == pd.Interval(
        22.0, 24.0, closed="right"
    )


########

# 98
data_dict = {
    "really_long_name_for_a_column": range(10),
    "another_really_long_name_for_a_column": [2 * item for item in range(10)],
    "another_really_longer_name_for_a_column": list("lllongname"),
    "this_is_getting_out_of_hand": list("longername"),
}


@pytest.mark.mlsd
def test_toColumnNamesFixedLen_700_inplace_null_ec():
    example_dataframe = pd.DataFrame(data_dict)
    assert (
        (
            example_dataframe.toColumnNamesFixedLen(
                column_length=700, inplace=True
            )
            == example_dataframe
        )
        .any()
        .any()
    )


# 99
@pytest.mark.mlsd
def test_toColumnNamesFixedLen_7():
    example_dataframe = pd.DataFrame(data_dict)
    assert (
        example_dataframe.toColumnNamesFixedLen(
            column_length=7, inplace=True
        ).columns
        == ["really", "another", "another_1", "this_is"]
    ).any()


# 100
@pytest.mark.mlsd
def test_toColumnNamesFixedLen_700_inplace_false_ec():
    example_dataframe = pd.DataFrame(data_dict)
    assert len(
        example_dataframe.toColumnNamesFixedLen(
            column_length=3, inplace=False
        ).columns[0]
    ) != len(example_dataframe.columns[0])


# 101
@pytest.mark.mlsd
def test_toColumnNamesFixedLen_3():
    example_dataframe = pd.DataFrame(data_dict)
    assert (
        example_dataframe.toColumnNamesFixedLen(column_length=7).columns
        == ["really", "another", "another_1", "this_is"]
    ).any()


# 102
# toUnixDatedatetime


# 103


# 104

# 101
