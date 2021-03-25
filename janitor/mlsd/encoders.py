# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")

from typing import  List  #,Dict
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.binary import BinaryEncoder

# from category_encoders.count import CountEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.polynomial import PolynomialEncoder
from category_encoders.basen import BaseNEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.cat_boost import CatBoostEncoder


# paso imports
from paso.base import pasoModel, PasoError
from paso.base import  pasoDecorators
from paso.toutil import toDataFrame
from loguru import logger


########

class Encoders(pasoModel):
    """
    Parameters:
        encoderKey: (str) One of Encoder.encoders()

        verbose: (str) (default) True, logging off (``verbose=False``)

    Note:
        **Encode**
    """
    _category_encoders__version_ = "2.0.0"

    _encoders_ = {}

    _category_encoders_ = [
        "BackwardDifferenceEncoder",
        "BinaryEncoder",
        "HashingEncoder",
        "HelmertEncoder",
        "OneHotEncoder",
        "OrdinalEncoder",
        "SumEncoder",
        "PolynomialEncoder",
        "BaseNEncoder",
        "LeaveOneOutEncoder",
        "TargetEncoder",
        "WOEEncoder",
        "MEstimateEncoder",
        "JamesSteinEncoder",
        "CatBoostEncoder",
        "EmbeddingEncoder",
    ]

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):

        """
        Parameters:
            modelKey: (str) On
            verbose: (str) (default) True, logging off (``verbose=False``)

        Note:

        """
        super().__init__()
        self.debug = True
        self.model = None
        self.model_name = None
        self.trained = False
        self.predicted = False


    def encoders(self) -> List:
        """
        Parameters:
            None

        Returns:
            List of available encoders names.
        """
        return [k for k in Encoders._category_encoders_]

    @pasoDecorators.TTWrap(array=True)
    def train(self, X: pd.DataFrame, verbose: bool = True, **kwargs):
        """
        Parameters:
            X:  pandas dataFrame, that encompasses all values encoded

            verbose:

        Returns:
            self

        Note: so complete dataset has same .fit instance, tran,valid, test and
        any other subsets should be merged and passed as X.

        Note: Do not include target in dataset as it must not be encoded.

        Note Encoding not required for tree-based learners  (RF, xgboost ,etc,).
        In some cases, encoding might make tree-base learner training
        and prediction worse.
        """

        if self.kind == {}:
            raise_PasoError(
                "keyword kind must be present at top level:{}:".format(
                    self.kind
                )
            )

        if self.kind_name not in Encoders._encoders_:
            raise_PasoError(
                "Train; no operation named: {} in encoder;: {}".format(
                    self.kind_name, Encoders._encoders_.keys()
                )
            )
        else:
            self.model_name = self.kind_name
            self.model = Encoders._encoders_[self.kind_name](
                **self.kind_name_kwargs
            )
            self.model.fit(X)

        self.trained = True

        return self

    @pasoDecorators.PredictWrap(array=False)
    def predict(self, X: pd.DataFrame, inplace:bool=False, **kwargs) -> pd.DataFrame:

        """
        Parameters:
            X:  pandas dataFrame #Todo:Dask numpy

            inplace: (CURRENTLY IGNORED)
                    False (boolean), replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False
            
        Returns:
            (DataFrame): transform X

        Note: once .train is called on entire dataset subsets(train,valid, test)
        predict can be called separately for each subset.

        Note: Do not include target in dataset as it must not be encoded.

        """

        self.predicted = True

        return self.model.transform(X, **kwargs)

    def inverse_predict(self, X: pd.DataFrame, inplace=False, **kwargs)-> pd.DataFrame:
        """
        Args:
            Xarg (array-like): Predictions of different models for the labels.

        Returns:
            (DataFrame): inverse of Xarg
        """
        X = Xarg.values
        if self.trained and self.predicted:
            X = self.model.inverse_transform(X)
            if self.verbose:
                logger.info("Scaler:inverse_transform:{}".format(self.encoderKey))
            return toDataFrame().transform(X, labels=Xarg.columns, inplace=False)
        else:
            raise pasoError(
                "scale:inverse_transform: must call train and predict before inverse"
            )

    def load(self, filepath=None):
        """
       Can not load an encoder model.

        Parameters:
            filepath: (str)
                ignored

        Raises:
                raise PasoError(" Can not load an encoder model.")
        """

        logger.error("Can not load an encoder model.")
        raise PasoError()

    def save(self, filepath=None):
        """
        Can not save an encoder model.

        Parameters:
            filepath: (str)
                ignored

        Raises:
                raise PasoError(" Can not save an encoder model.")

        """

        logger.error("Can not save an encoder model.")
        raise PasoError()


class EmbeddingEncoder(BaseEstimator, TransformerMixin):
    """

    Attributes:
        None
    """

    def __init__(self):
        self.coefs_ = []  # Store tau for each transformed variable

    #        self.verbose = verbose

    def _reset(self):
        self.coefs_ = []  # Store tau for each transformed variable

    def fit(self, X, **kwargs):
        return self.train(X, **kwargs)

    def train(self, X, inplace=False, **kwargs):
        """
        Args:
            X (dataframe):
                Calculates BoxCox coefficients for each column in X.

        Returns:
            self (model instance)

        Raises:
            ValueError will result of not 1-D or 2-D numpy array, list or Pandas Dataframe.

            ValueError will result if has any negative value elements.

            TypeError will result if not float or integer type.

        """

        for x_i in X.T:
            self.coefs_.append(boxcox(x_i)[1])
        return self

    def transform(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def predict(self, X, inplace=False, **kwargs):
        # todo: transform dataframe to numpy and back?
        """
        Transform data using a previous ``.fit`` that calulates BoxCox coefficents for data.

        Args:
            X (np.array):
                Transform  numpy 1-D or 2-D array distribution by BoxCoxScaler.

        Returns:
            X (np.array):
                tranformed by BoxCox

        """

        return np.array(
            [boxcox(x_i, lmbda=lmbda_i) for x_i, lmbda_i in zip(X.T, self.coefs_)]
        ).T

    # is there a way to specify with step or does does step need enhancing?
    def inverse_transform(self, y):
        """
        Recover original data from BoxCox transformation.
        """

        return np.array(
            [
                (1.0 + lmbda_i * y_i) ** (1.0 / lmbda_i)
                for y_i, lmbda_i in zip(y.T, self.coefs_)
            ]
        ).T

    def inverse_predict(self, y, inplace=False):
        """
        Args:
            y: dataframe

        Returns:
            Dataframe: Recover original data from BoxCox transformation.
        """
        return self.inverse_transform(y)

########

# initilize wth all encoders
for encoder in Encoders._category_encoders_:
    Encoders._encoders_[encoder] = eval(encoder)


# add new encoders
#Encoders._encoders_["EmbeddingEncoder"] = EmbeddingEncoder
