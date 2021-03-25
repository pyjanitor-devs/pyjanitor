# !/usr/bin/env python
# -*- coding: utf-8 -*-
__coverage__ = 0.86
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License" ""
import warnings

warnings.filterwarnings("ignore")
import pytest, os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# paso imports
from paso.base import PasoError, DataFrame_to_Xy
from paso.pre.cleaners import Balancers, Augmenters, Imputers
from paso.pre.cleaners import values_to_nan, delete_NA_Features
from paso.pre.cleaners import calculate_NaN_ratio, delete_Duplicate_Features
from paso.pre.cleaners import statistics, feature_Statistics
from paso.pre.cleaners import delete_Features_with_Single_Unique_Value
from paso.pre.cleaners import delete_Features_with_All_Unique_Values

# from paso.pre.cleaners import feature_Feature_Correlation, plot_corr
from paso.pre.cleaners import delete_Features
from paso.pre.cleaners import boolean_to_integer
from paso.pre.cleaners import delete_Features_not_in_train_or_test
from paso.pre.inputers import Inputers

os.chdir('../../')

def create_dataset(
    n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3, class_sep=0.8, n_clusters=1
):
    return make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        weights=list(weights),
        class_sep=class_sep,
        random_state=0,
    )


x_pd, y_pg = create_dataset(n_samples=1000, weights=(0.1, 0.2, 0.7))


# 1
def test_Values_to_nan_list0_0(City):
    assert (
        City.paso_values_to_nan(inplace=True, values=[0.0]).isnull().sum().sum() == 843
    )


# 2
def test_f_Values_to_nan_none(City):
    assert City.paso_values_to_nan(inplace=True, values=[]).isnull().sum().sum() == 0


# 3
def test_Values_to_nan_none(City):
    assert values_to_nan(City, inplace=True, values=[]).isnull().sum().sum() == 0


# 4
def test_Values_to_nan_list0_0(City):
    assert values_to_nan(City, inplace=True, values=[0.0]).isnull().sum().sum() == 843


# 5
def test_Values_to_nan_0_0(City):
    assert values_to_nan(City, inplace=True, values=0.0).isnull().sum().sum() == 843


# 6
def test_Values_to_nan_0_0_inplace(City):
    assert City.paso_values_to_nan(inplace=True, values=0.0).isnull().sum().sum() == 843


# 7
def test_Values_to_nan_0_0_not_inplace(City):
    assert (City.paso_values_to_nan(inplace=False, values=0.0) != City).any().any()


# 8
def test_delete_NA_features_to_999_2(City):
    City["bf"] = 999
    City["bf2"] = 2
    values_to_nan(City, inplace=True, values=[2, 999])
    assert delete_NA_Features(City, inplace=True, threshold=0.8, axis=1).shape[1] == 14


# 9
def test_delete_NA_features_to_999_2_chain(City):
    City["bf"] = 999
    City["bf2"] = 2
    assert (
        City.paso_values_to_nan(inplace=True, values=[2, 999])
        .paso_delete_NA_Features(inplace=True, threshold=0.8, axis=1)
        .shape[1]
        == 14
    )


# 10


def test_delete_NA_Features_to_2(City):
    City["bf"] = 999
    City["bf2"] = 2
    z = City.shape[1] - 1
    values_to_nan(City, inplace=True, values=[2])
    assert delete_NA_Features(City, inplace=True, threshold=0.8, axis=1).shape[1] == z


# 11
def test_delete_NA_Features_to_2_chain(City):
    City["bf"] = 999
    City["bf2"] = 2
    z = City.shape[1] - 1

    assert (
        values_to_nan(City, inplace=True, values=[2])
        .paso_delete_NA_Features(inplace=True, threshold=0.8, axis=1)
        .shape[1]
        == z
    )


# 12
def test_delete_NA_Features_to1_threshold(City):
    City["bf"] = 999
    City["bf2"] = 2
    z = City.shape[1] - 0
    values_to_nan(City, inplace=True, values=[-1])
    assert delete_NA_Features(City, inplace=True, threshold=1.0).shape[1] == z + 1


# 1
def test_delete_NA_Features_to_big_threshold(City):
    City["bf"] = 999
    City["bf2"] = 2
    z = City.shape[1] - 0
    values_to_nan(City, inplace=True, values=[2])
    with pytest.raises(PasoError):
        assert (
            delete_NA_Features(City, inplace=True, axis=2, threshold=1.1).shape[1]
            == z + 1
        )


# 1
def test_calculate_NaN_ratio_err(City):
    City["bf"] = 999
    City["bf2"] = 2
    z = City.columns
    values_to_nan(City, inplace=True, values=[2, 999])
    c = calculate_NaN_ratio(City, inplace=True, axis=1).columns
    with pytest.raises(AssertionError):
        assert len(c) == len(z) + 1


# 15
def test_calculate_NaN_ratio(City):
    City["bf"] = 999
    City["bf2"] = 2
    z = City.columns
    values_to_nan(City, inplace=True, values=[2, 999])
    c = calculate_NaN_ratio(City, inplace=True, axis=0).columns
    assert len(c) == len(z) + 1


# 1
def test_Calculate_NA_has_NaN_ratio(City):
    City["bf"] = 999
    City["bf2"] = 2
    z = City.columns
    values_to_nan(City, inplace=True, values=[2, 999])
    c = calculate_NaN_ratio(City, inplace=True, axis=0).columns
    assert c.shape == (17,)


# 1
def test_Calculate_NA_has_NaN_ratio_count(City):
    City["bf"] = 999
    City["bf2"] = 2
    values_to_nan(City, inplace=True, values=[2, 999])
    assert calculate_NaN_ratio(City, inplace=True, axis=0).count().count() == 17


def test_Calculate_NA_has_NaN_ratio_c_err(City):
    City["bf"] = 999
    City["bf2"] = 2
    values_to_nan(City, inplace=True, values=[2, 999])
    calculate_NaN_ratio(City, inplace=True, axis=1).shape == [512, 18]


def test_Calculate_NA_has_NaN_ratio_c_err(City):
    City["bf"] = 999
    City["bf2"] = 2
    values_to_nan(City, inplace=True, values=[2, 999])
    calculate_NaN_ratio(City, inplace=True, axis=1).count().count() == 512 * 16


# 20
def test_delete_Duplicate_Featuress(City):
    City["bf"] = 999
    City["bf2"] = City["bf"]
    assert delete_Duplicate_Features(City, inplace=True).shape[1] == 15


#
def test_delete_Duplicate_Featuress4(City):
    City["bf"] = 999
    City["bf4"] = City["bf3"] = City["bf2"] = City["bf"]
    assert delete_Duplicate_Features(City, inplace=True).shape[1] == 15


#
def test_cleaner_statistics(City):
    assert statistics() == [
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


#
def test_feature_Statistics_bad_arg(City):
    with pytest.raises(PasoError):
        City.paso_feature_Statistics(
            concat=False, statistics=[" sum"], inplace=True, verbose=True
        )


def test_feature_Statistics_3_stats_rows(City):
    c = feature_Statistics(
        City,
        concat=False,
        axis=0,
        statistics=["sum", "mean", "kurt"],
        inplace=True,
        verbose=True,
    )
    assert c.shape == (3*14, 1)


# 25
def test_feature_Statistics_meancolumns(City):
    c = feature_Statistics(
        City, concat=False, axis=1, statistics=["mean"], inplace=True, verbose=True
    )
    assert c.shape == (506, 1)


def test_feature_Statistics_meancolumns_concat(City):
    c = feature_Statistics(
        City, concat=True, axis=1, statistics=["mean"], inplace=True, verbose=True
    )
    assert c.shape == (506, 15)


# 27
def test_feature_Statistics_all(City):
    c = feature_Statistics(
        City, axis=1, concat=False, statistics=["all"], inplace=True, verbose=True
    )
    assert c.shape == (506, 12)


# 2
def test_feature_Statistics_3_stats_concat(City):
    c = feature_Statistics(
        City,
        concat=True,
        statistics=["sum", "mean", "kurt"],
        axis=1,
        inplace=True,
        verbose=True,
    )
    assert c.shape == (506, 17)


# 29
def test_delete_Features_with_Single_Unique_Value3(City):
    City["bf"] = 999
    City["bf2"] = 2
    City["bf3"] = 1
    assert delete_Features_with_Single_Unique_Value(City, inplace=True).shape[1] == 14


# 30
def test_delete_Features_with_Single_Unique_Value_ignore(City):
    City["bf"] = 999
    City["bf2"] = 2
    City["bf3"] = 1
    assert (
        delete_Features_with_Single_Unique_Value(
            City, features=["bf3"], inplace=True
        ).shape[1]
        == 15
    )


def test_delete_Features_with_Single_All_Value_features(City):
    City["bf"] = np.linspace(0, 505, num=506)

    assert (
        delete_Features_with_All_Unique_Values(
            City, features=["bf3"], inplace=True
        ).shape[1]
        == 14
    )


# 32
def test_delete_Features_with_Single_All_Value_ffeatures(City):
    City["bf"] = np.linspace(0, 505, num=506)

    assert (
        City.paso_delete_Features_with_All_Unique_Values(
            features=["bf3"], inplace=True
        ).shape[1]
        == 14
    )


# 33
def test_Feature_Correlation(City):
    City.paso_plot_corr()
    assert City.paso_feature_Feature_Correlation(verbose=True).shape == (14, 14)


# 34
def test_Feature_Correlation_flowers(flower):

    flower.paso_plot_corr(kind="visual")
    assert flower.paso_feature_Feature_Correlation(verbose=True).shape == (5, 5)


# 35
def test_delete_features_null(flower):

    assert flower.paso_delete_Features(verbose=True, inplace=True).shape == (150, 5)


#
def test_delete_features_all(flower):

    assert delete_Features(
        flower, features=flower.columns, verbose=True, inplace=True
    ).shape == (150, 0)


#
def test_delete_features_but12(City):

    assert delete_Features(
        City, features=City.columns[3:5], verbose=True, inplace=True
    ).shape == (506, 12)


#
def test_delete_features_not_in_train_or_test(City):

    train = City.copy()
    City["bf"] = 999
    City["bf2"] = 2
    City["bf3"] = 1
    test = City.copy()
    delete_Features_not_in_train_or_test(train, test, features=["bf3"], inplace=True)
    assert train.shape[1] == (test.shape[1] - 1)


# 39
def test_Balancer_classBalancer():
    assert Balancers().balancers() == [
        "RanOverSample",
        "SMOTE",
        "ADASYN",
        "BorderLineSMOTE",
        "SVMSMOTE",
        "SMOTENC",
        "RandomUnderSample",
        "ClusterCentroids",
        "NearMiss",
        "EditedNearestNeighbour",
        "RepeatedEditedNearestNeighbours",
        "CondensedNearestNeighbour",
        "OneSidedSelection",
    ]


# 40
def test_Balancer_read_parameters_file_not_exists(City):
    o = Balancers()
    with pytest.raises(PasoError):
        o.transform(City[City.columns.difference(["MEDV"])], City["MEDV"]).shape == []


#
def test_Balancer_Smote_Flowers(flower):
    o = Balancers(description_filepath="descriptions/pre/cleaners/SMOTE.yaml")
    X, y = DataFrame_to_Xy(flower, "TypeOf")
    X, y = o.transform(X, y)
    assert X.shape == (150, 4) and y.shape == (150,)


# 42
def test_Augmenter_no_Ratio(flower):
    o = Augmenters(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    X, y = DataFrame_to_Xy(flower, "TypeOf")
    with pytest.raises(PasoError):
        assert o.transform(X, y) == 0


# 43
def test_Augmenter_Smote_Flower_ratio_1_default(flower):
    o = Augmenters(description_filepath="descriptions/pre/cleaners/SMOTE.yaml")
    X, y = DataFrame_to_Xy(flower, "TypeOf")
    X, y = o.transform(X, y, ratio=0.0)
    assert X.shape == (300, 4) and y.shape == (300,)


# 44
def test_Augmenter_Smote_Flower_ratio_1(flower):
    o = Augmenters(description_filepath="descriptions/pre/cleaners/SMOTE.yaml")
    X, y = DataFrame_to_Xy(flower, "TypeOf")
    X, y = o.transform(X, y, ratio=1.0)
    assert X.shape == (300, 4) and y.shape == (300,)


# 44b
def test_Augmenter_Smote_Flower_ratio_2(flower):
    o = Augmenters(description_filepath="descriptions/pre/cleaners/SMOTE.yaml")
    X = flower.paso_augment(o, "TypeOf", ratio=1.0)
    assert X.shape == (300, 5)


# 34
def test_Augmenter_bad_ratio(flower):
    o = Augmenters(description_filepath="descriptions/pre/cleaners/SMOTE.yaml")
    X, y = DataFrame_to_Xy(flower, "TypeOf")
    with pytest.raises(PasoError):
        o.transform(X, y, ratio=2.0)


# 45
def test_imputer_given():
    assert len(Imputers().imputers()) == 6


# 46
def test_imputer_features_no_given():
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    assert imp.transform(train, inplace=True, verbose=True).shape == (1484, 9)


# 47
def test_imputer_features_not_found():
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    with pytest.raises(PasoError):
        train = imp.transform(
            train, features=["bad", "badder", "Alm"], inplace=True, verbose=True
        )


# 48
def test_imputer_features_not_found():
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")
    train.loc[0, "Alm"] = np.NaN
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    assert (
        imp.transform(train, features=["Alm"], inplace=True, verbose=True)
        .isnull()
        .all()
        .all()
    ) == False


# 49
def test_imputer_features_allnans():
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")
    train["Alm"] = np.NaN
    train.loc[0, "Alm"] = 0
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    with pytest.raises(PasoError):
        assert (
            imp.transform(train, features=["Alm"], inplace=True, verbose=True)
            .isnull()
            .all()
            .all()
        ) == False


# 50
def test_imputer_features_nans_found():
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")
    train["bad"] = np.NaN
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    assert (
        imp.transform(train, features=["Alm"], inplace=True, verbose=True)
        .isnull()
        .any()
        .any()
    ) == True


# 51
def test_imputer_features_nans_found():
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")
    train["bad"] = np.NaN
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    assert (
        train.paso_impute(imp, features=["Alm"], inplace=True, verbose=True)
        .isnull()
        .any()
        .any()
    ) == True


# 52
def test_boo_lnone(City):

    train = City.copy()
    boolean_to_integer(train, inplace=True)
    assert (train == City).all().all()


# 53
def test_boo_lnone_c(City):

    train = City.copy()
    train.paso_boolean_to_integer(inplace=True)
    assert (train == City).all().all()


# 54
def test_booL_some(City):
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")

    train["Alm"] = False
    train.loc[0, "Alm"] = True
    assert boolean_to_integer(train, inplace=True).loc[0, "Alm"] == 1


# 55
def test_booL_some_not_true(City):
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")

    train["Mcg"] = True
    train.loc[0, "Mcg"] = False
    train["Alm"] = False
    train.loc[0, "Alm"] = True
    assert (
        boolean_to_integer(train, inplace=True).loc[0, "Alm"] == 1
        and train.loc[0, "Mcg"] == 0
    )


# 56
def test_Statistics_3_column_stats(City):

    column_stats = feature_Statistics(
        City,
        concat=False,
        axis=1,
        statistics=["sum", "mean", "kurt"],
        inplace=False,
        verbose=True,
    )
    assert column_stats.shape == (506, 3)


# 57
def test_imputer_feature_boston():
    o = Inputers(description_filepath="descriptions/pre/inputers/boston.yaml")
    train = o.transform(dataset="train")
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    assert imp.transform(train, inplace=True, verbose=True).shape == (506, 14)


# 58
def test_imputer_feature_boston_err():
    o = Inputers(description_filepath="descriptions/pre/inputers/boston.yaml")
    train = o.transform(dataset="train")
    train["ggoo"] = np.nan
    imp = Imputers(
        description_filepath="descriptions/pre/cleaners/most_frequent_impute.yaml"
    )
    with pytest.raises(PasoError):
        assert imp.transform(train, inplace=True, verbose=True).shape == (506, 14)


# 52
def test_paso_standardize_column_names_snake(City):

    City.paso_standardize_column_names(case_type="snake")
    assert City.shape == (506, 14)


# 53
def test_paso_standardize_column_names_(City):
    c = City.paso_standardize_column_names(inplace=False, case_type="upper")
    assert City.shape == (506, 14)
