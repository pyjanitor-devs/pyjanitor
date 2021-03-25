from pathlib import Path

import pytest

# paso imports

from paso.base import Paso, PasoError
from paso.pre.scalers import Scaler

session = Paso(parameters_filepath="parameters/lesson.1.yaml").startup()
# 0
def test_df_Class_init_NoArg():
    with pytest.raises(TypeError):
        g = Scaler()


# 1
def test_df_Class_init_WrongScaler():
    with pytest.raises(PasoError):
        g = Scaler("GORG")


# BoxCoxScaler unit tests
# 2
def test_df_BoxCox_City_negative_error(df_City):
    g = Scaler("BoxCoxScaler")
    with pytest.raises(ValueError):
        g.train(df_City)


# 2b
def test_df_BoxCox_City_NA_error(df_small):
    g = Scaler("BoxCoxScaler")
    with pytest.raises(PasoError):
        g.train(df_small)


# 3
def test_df_BoxCox_df_type(df_type):
    g = Scaler("BoxCoxScaler")
    assert g.train(df_type) == g


# 4
def test_df_BoxCox_numpy_1d_error(y):
    g = Scaler("BoxCoxScaler")
    with pytest.raises(PasoError):
        g.train(y)


# 5
def test_predict_df_BoxCox_df_type(df_type):
    g = Scaler("BoxCoxScaler")
    g.train(df_type)
    assert g.predict(df_type).shape == df_type.shape


# 6
def test_df_BoxCox_type_error(ystr):
    g = Scaler("BoxCoxScaler")
    with pytest.raises(TypeError):
        g.train(ystr())


# LambertScaler unit tests
# 7
def test_df_Lambert_train(df_type):
    g = Scaler("LambertScaler")
    assert g.train(df_type) == g


# 8
def test_df_Lambert_predict(df_type):
    g = Scaler("LambertScaler")
    g.train(df_type)
    assert g.predict(df_type).shape == df_type.shape


# 9
def test_df_Lambert_numpy_1d_error(y):
    g = Scaler("LambertScaler")
    with pytest.raises(PasoError):
        g.train(y)


# 10

# 11
def test_df_Lambert_type_error(ystr):
    g = Scaler("LambertScaler")
    with pytest.raises(PasoError):
        g.train(ystr)


# 12

# 13
def test_df_Lambert_no_fit(yn):
    g = Scaler("LambertScaler")
    with pytest.raises(AttributeError):
        g.fit()


# 14
def test_ScalerList(X):
    g = Scaler("BoxCoxScaler")
    assert g.scalers() == [
        "StandardScaler",
        "MinMaxScaler",
        "Normalizer",
        "MaxAbsScaler",
        "RobustScaler",
        "QuantileTransformer",
        "BoxCoxScaler",
        "LambertScaler",
    ]


# 15
def test_df_BoxCox_inverse(df_type):
    g = Scaler("BoxCoxScaler")
    g.train(df_type, inplace=False)
    assert g.inverse_predict(g.predict(df_type) == df_type).any().any()


# 16
def test_df_BoxCox_write(df_type):
    g = Scaler("BoxCoxScaler").cacheOn()
    g.train(df_type).predict(df_type)
    fp: str = "tmp/df"
    g.write(fp)
    assert (
        g.trained
        and g.predicted
        and g.cache
        and g.persisted
        and (g.save_file_name == fp)
    ) == True


def test_df_MinMax_write(df_type):
    g = Scaler("MinMaxScaler").cacheOn()
    g.train(df_type).predict(df_type)
    fp: str = "tmp/df"
    g.write(fp)
    assert (
        g.trained
        and g.predicted
        and g.cache
        and g.persisted
        and (g.save_file_name == fp)
    ) == True


def test_df_Lambert_write(df_type):
    g = Scaler("LambertScaler").cacheOn()
    g.train(df_type).predict(df_type)
    fp: str = "tmp/df"
    g.write(fp)
    assert (
        g.trained
        and g.predicted
        and g.cache
        and g.persisted
        and (g.save_file_name == fp)
    ) == True


# 17
def test_df_Lambert_wo(df_City):
    g = Scaler("LambertScaler").cacheOn()
    g.train(df_City).predict(df_City)
    fp: str = "tmp/df_write"
    g.write(fp)
    assert (
        g.trained
        and g.predicted
        and g.cache
        and g.persisted
        and (g.save_file_name == fp)
    ) == True


# 18
def test_df_BoxCox_read(df_type):
    g = Scaler("BoxCoxScaler").cacheOn()
    g.train(df_type).predict(df_type)
    fp: str = "tmp/df_write"
    g.write(fp)
    g.read(fp)
    assert (
        g.trained
        and g.predicted
        and g.cache
        and g.persisted
        and (g.save_file_name == fp)
    ) == True


def test_df_MinMax_read(df_City):
    g = Scaler("MinMaxScaler").cacheOn()
    g.train(df_City).predict(df_City)
    fp: str = "tmp/df_write"
    g.write(fp)
    g.read(fp)
    assert (
        g.trained
        and g.predicted
        and g.cache
        and g.persisted
        and (g.save_file_name == fp)
    ) == True


def test_df_Lambert_read(df_City):
    g = Scaler("LambertScaler").cacheOn()
    g.train(df_City).predict(df_City)
    fp: str = "tmp/df_write"
    g.write(fp)
    g.read(fp)
    assert (
        g.trained
        and g.predicted
        and g.cache
        and g.persisted
        and (g.save_file_name == fp)
    ) == True


# 18

# 19
def test_bad_scale_name():
    with pytest.raises(PasoError):
        g = Scaler("fred")
