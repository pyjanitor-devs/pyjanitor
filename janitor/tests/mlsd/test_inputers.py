#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

__coverage__ = 0.86

import pandas as pd

import pytest, os

# paso imports

from paso.base import Paso, Param, PasoError
from paso.pre.inputers import Inputers, Splitters

os.chdir('../../')


fp = "parameters/base.yaml"
session = Paso(parameters_filepath=fp).startup()

# 0
def test_paso_param_file():
    assert session.parameters_filepath == fp


# 1
def test_paso_param_file_globa():
    assert session.parameters_filepath == Param.gfilepath


# 2
def test_parm_global():
    assert Param.parameters_D["project"] == "HPKinetics/paso"


# 3
def test_inputer_imputer():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    assert inputer.inputers() == [
        "exec",
        "cvs",
        "xls",
        "xlsm",
        "text",
        "image2D",
        "image3D",
    ]


# 3b
def test_inputer_formats():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    assert inputer.formats() == ["csv", "zip", "data", "sklearn.datasets", "yaml"]


# 3c
def test_inputer_datasets():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    assert inputer.datasets() == [
        "train",
        "valid",
        "test",
        "sampleSubmission",
        "directory_path",
    ]


# 4
def test_inputer_transform_exec(flower):  # descriptions
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    assert (inputer.transform() == flower).any().any()


# 5
def test_inputer_transform_cvs_url():
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    names = [
        "Class",
        "Alcohol",
        "Malic-acid",
        "Ash",
        "Alcalinity-ash",
        "Magnesium",
        "phenols",
        "Flavanoids",
        "Nonflavanoid-phenols",
        "Proanthocyanins",
        "Color-intensity",
        "Hue",
        "OD280-OD315-diluted-wines",
        "Proline",
    ]

    winmeo = pd.read_csv(link, names=names).head()
    inputer = Inputers(description_filepath="descriptions/pre/inputers/wine.yaml")
    assert (inputer.transform().columns == winmeo.columns).any()


# 6
def test_inputer_transform_splitter_onto_wrong_place():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    Flower = inputer.transform()
    splitter = Splitters()
    with pytest.raises(IndexError):
        train, valid = splitter.transform(
            Flower,
            target=inputer.target,
            description_filepath="descriptions/pre/inputers/test_size_30.yaml",
        )


# 7
def test_inputer_transform_splitter_X_train():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    train = inputer.transform()
    y = train[inputer.target].values
    X = train[train.columns.difference([inputer.target])]
    splitter = Splitters(
        description_filepath="descriptions/pre/inputers/split_30_stratify.yaml"
    )
    train, valid, y_train, y_valid = splitter.transform(X, y)
    assert train.shape == (105, 4) and y_train.shape == (105,)


# 8
def test_inputer_transform_splitter_X_valid():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    train = inputer.transform()
    y = train[inputer.target].values
    X = train[train.columns.difference([inputer.target])]
    splitter = Splitters(
        description_filepath="descriptions/pre/inputers/split_30_stratify.yaml"
    )
    train, valid, y_train, y_valid = splitter.transform(X, y)
    assert valid.shape == (45, 4) and y_valid.shape == (45,)


# 9
def test_inputer_transform_odescription_arg_error(flower):
    o = Inputers()
    with pytest.raises(PasoError):
        Flower = o.transform(
            flower, description_filepath="descriptions/inputers/iris.yaml"
        )


# 10
def test_inputer_transform_ontological_bad_description_filepath(flower):
    o = Inputers(description_filepath="descriptions/inputers/XXXX.yaml")
    with pytest.raises(PasoError):
        Flower = o.transform()


# 11
def test_inputer_transform_flower(flower):
    o = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    Flower = o.transform()
    assert Flower.shape == flower.shape


# 12
def test_inputer_transform_wine():
    o = Inputers(description_filepath="descriptions/pre/inputers/wine.yaml")
    Wine = o.transform()
    assert Wine.shape == (178, 14)


# 13
def test_inputer_transform_otto_group():
    o = Inputers(description_filepath="descriptions/pre/inputers/otto_group.yaml")
    otto_group = o.transform()
    assert otto_group.shape == (61878, 95)


# 14
def test_spitter_transform_s_wine():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/wine.yaml")
    train = inputer.transform()
    y = train[inputer.target].values
    X = train[train.columns.difference([inputer.target])]
    splitter = Splitters(
        description_filepath="descriptions/pre/inputers/split_30_stratify.yaml"
    )
    train, valid, y_train, y_valid = splitter.transform(X, y)
    assert (
        train.shape == (124, 13)
        and valid.shape == (54, 13)
        and y_train.shape == (124,)
        and y_valid.shape == (54,)
    )


# 15
def test_inputer_train_arg_error():
    o = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    with pytest.raises(AttributeError):
        flower = o.traih()


# 16
def test_inputer_train_bad_file(flower):
    o = Inputers(description_filepath="descriptions/pre/inputers/bad.yaml")
    with pytest.raises(PasoError):
        _ = o.transform()


# 17
def test_inputer_pima_diabetes_train():
    o = Inputers(
        description_filepath="descriptions/pre/inputers/pima-diabetes.yaml"
    )
    test = o.transform(dataset="train")
    assert test.shape == (768, 9)


# 18
def test_inputer_otto_group_test():
    o = Inputers(description_filepath="descriptions/pre/inputers/otto_group.yaml")
    test = o.transform(dataset="test")
    assert test.shape == (144368, 94)


# 19
def test_inputer_otto_groupsample_Submission():
    o = Inputers(description_filepath="descriptions/pre/inputers/otto_group.yaml")
    sampleSubmission = o.transform(dataset="sampleSubmission")
    assert sampleSubmission.shape == (144368, 10)


# 20
def test_spitter_transform_otto_group():
    inputer = Inputers(
        description_filepath="descriptions/pre/inputers/otto_group.yaml"
    )
    train = inputer.transform()
    y = train[inputer.target].values
    X = train[train.columns.difference([inputer.target])]
    splitter = Splitters(
        description_filepath="descriptions/pre/inputers/split_20_stratify.yaml"
    )
    train, valid, y_train, y_valid = splitter.transform(X, y)
    assert (
        train.shape == (49502, 94)
        and valid.shape == (12376, 94)
        and y_train.shape == (49502,)
        and y_valid.shape == (12376,)
    )


# 21
def test_splitter_transform__onto_group():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    train = inputer.transform()
    y = train[inputer.target].values
    X = train[train.columns.difference([inputer.target])]
    splitter = Splitters(
        description_filepath="descriptions/pre/inputers/split_20_stratify.yaml"
    )
    train, valid, y_train, y_valid = splitter.transform(X, y)
    assert (
        train.shape == (120, 4)
        and valid.shape == (30, 4)
        and y_train.shape == (120,)
        and y_valid.shape == (30,)
    )


# 22
def test_inputer_transform_dataset_setting_bad(flower):
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    with pytest.raises(PasoError):
        assert (inputer.transform(dataset="test") == flower).any().any()


# 23
def test_inputer_datasets():
    inputer = Inputers(description_filepath="descriptions/pre/inputers/iris.yaml")
    assert inputer.datasets() == [
        "train",
        "valid",
        "test",
        "sampleSubmission",
        "directory_path",
    ]


# 24cvs-zip
def test_inputer_creditcard_url_cvs_zip():
    o = Inputers(description_filepath="descriptions/pre/inputers/creditcard.yaml")
    train = o.transform(dataset="train")
    assert train.shape == (284807, 31)


# 25
#
def test_spltter_transform_creditcard_url_30__cvs_zip():
    inputer = Inputers(
        description_filepath="descriptions/pre/inputers/yeast3.yaml"
    )
    train = inputer.transform(dataset="train")
    y = train[inputer.target].values
    X = train[train.columns.difference([inputer.target])]
    splitter = Splitters(
        description_filepath="descriptions/pre/inputers/split_20_stratify.yaml"
    )
    train, valid, y_train, y_valid = splitter.transform(X, y)
    assert (
        train.shape == (1187, 8)
        and valid.shape == (297, 8)
        and y_train.shape == (1187,)
        and y_valid.shape == (297,)
    )


# 26
def test_inputer_bad_keyword(flower):  # descriptions
    inputer = Inputers(
        description_filepatsh="descriptions/pre/inputers/iris.yaml"
    )
    with pytest.raises(PasoError):
        assert (inputer.transform() == flower).any().any()


# 27
def test_inputer_yeast3_cvs_zip():
    o = Inputers(description_filepath="descriptions/pre/inputers/yeast3.yaml")
    train = o.transform(dataset="train")
    assert train.shape == (1484, 9)


# 28
def test_inputer_create_data():
    o = Inputers(
        description_filepath="descriptions/pre/inputers/create-data.yaml"
    )
    train = o.transform(dataset="train")
    assert train.shape == (1000, 3)


# 29
def test_inputer_pima():
    o = Inputers(
        description_filepath="descriptions/pre/inputers/pima-diabetes.yaml"
    )
    train = o.transform(dataset="train")
    assert train.shape == (768, 9)


# 30
def test_spltter_transform_creditcard_20__url_cvs_zip():
    inputer = Inputers(
        description_filepath="descriptions/pre/inputers/creditcard.yaml"
    )
    train = inputer.transform(dataset="train")
    y = train[inputer.target].values
    X = train[train.columns.difference([inputer.target])]
    splitter = Splitters(
        description_filepath="descriptions/pre/inputers/split_20_stratify.yaml"
    )
    train, valid, _, _ = splitter.transform(X, y)
    assert train.shape == (227845, 30) and valid.shape == (56962, 30)
#31
def test_inputer_bad_train():  # descriptions
    inputer = Inputers(
        description_filepath="descriptions/pre/inputers/otto_group_bad.yaml"
    )
    with pytest.raises(PasoError):
        assert (inputer.transform()  == 1)

#32
def test_inputer_bad_test():  # descriptions
    inputer = Inputers(
        description_filepath="descriptions/pre/inputers/otto_group_bad2.yaml"
    )
    with pytest.raises(PasoError):
        assert (inputer.transform()  == 1)

#33
def test_inputer_bad_kind():  # descriptions
    inputer = Inputers(
        description_filepath="descriptions/pre/inputers/otto_group_bad4.yaml"
    )
    with pytest.raises(PasoError):
        assert (inputer.transform()  == 1)