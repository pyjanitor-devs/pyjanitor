import pandas as pd
from pandas.util._validators import validate_bool_kwarg
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from numba import jit
import joblib

# sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.utils import get_obj_cols, convert_input
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import QuantileTransformer
import sklearn.preprocessing.data as skpr

# scipy imports
from scipy.special import lambertw
from scipy.stats import kurtosis, boxcox
from scipy.optimize import fmin

# paso imports
from paso.base import pasoModel, raise_PasoError
from paso.base import pasoDecorators
from paso.base import Paso
from paso.toutil import toDataFrame
from loguru import logger
import sys

# !/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#

__ScalerDict__ = {}
__ScalerDict__["StandardScaler"] = StandardScaler
__ScalerDict__["MinMaxScaler"] = MinMaxScaler
__ScalerDict__["Normalizer"] = Normalizer
__ScalerDict__["MaxAbsScaler"] = MaxAbsScaler
__ScalerDict__["RobustScaler"] = RobustScaler
__ScalerDict__["QuantileTransformer"] = QuantileTransformer
# __ScalerDict__['self'] = skpr.__all__,
#######
Paso()
# Coefficent Functions
@jit
def _w_d(z, delta):
    # Eq. 9
    if delta < 1e-6:
        return z
    return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)


@jit
def _w_t(y, tau):
    # Eq. 8
    return tau[0] + tau[1] * _w_d((y - tau[0]) / tau[1], tau[2])


@jit
def _inverse(x, tau):
    # Eq. 6
    u = (x - tau[0]) / tau[1]
    return tau[0] + tau[1] * (u * np.exp(u * u * (tau[2] * 0.5)))


def _igmm(y, tol=1.22e-4, max_iter=100):
    # Infer mu, sigma, delta using IGMM in Alg.2, Appendix C
    if np.std(y) < 1e-4:
        return np.mean(y), np.std(y).clip(1e-4), 0
    delta0 = _delta_init(y)
    tau1 = (np.median(y), np.std(y) * (1.0 - 2.0 * delta0) ** 0.75, delta0)
    for k in range(max_iter):
        tau0 = tau1
        z = (y - tau1[0]) / tau1[1]
        delta1 = _delta_gmm(z)
        x = tau0[0] + tau1[1] * _w_d(z, delta1)
        mu1, sigma1 = np.mean(x), np.std(x)
        tau1 = (mu1, sigma1, delta1)

        if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
            break
        else:
            if k == max_iter - 1:
                raise ValueError(
                    "Warning: No convergence after %d iterations. Increase max_iter."
                    % max_iter
                )
    return tau1


def _delta_gmm(z):
    # Alg. 1, Appendix C
    delta0 = _delta_init(z)

    def func(q):
        u = _w_d(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.0
        else:
            k = kurtosis(u, fisher=True, bias=False) ** 2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            else:
                return k

    res = fmin(func, np.log(delta0), disp=0)
    return np.around(np.exp(res[-1]), 6)


def _delta_init(z):
    gamma = kurtosis(z, fisher=False, bias=False)
    with np.errstate(all="ignore"):
        delta0 = np.clip(1.0 / 66 * (np.sqrt(66 * gamma - 162.0) - 6.0), 0.01, 0.48)
    if not np.isfinite(delta0):
        delta0 = 0.01
    return delta0


########
# BoxCoxScaler
class BoxCoxScaler(BaseEstimator, TransformerMixin):
    """
    BoxCoxScaler method to Gaussianize heavy-tailed data. The Box-Cox transformation is to be used to:
        - stabilize variance
        - remove right tail skewness
    
    There are two major limitations of this approach:
        - only applies to positive data 
        - transforms into normal gaussian form only data with a Gausssian heavy right-hand tail.

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

    def train(self, X, inplace=True, **kwargs):
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


__ScalerDict__["BoxCoxScaler"] = BoxCoxScaler
########
# LambertScaler


class LambertScaler(BaseEstimator, TransformerMixin):
    """
    LambertScaler method to Gaussianize heavy-tailed data.
    - a one-parameter (``w_t``) family based on Lambert's W function that removes heavy tails from data.
    - the ``LambertScaler`` has no difficulties with negative values.

    Args:
        tol(optional): (default) 0.01, ``w_t`` is calulated internally to ``tol`` erance.

        max_iter(optional): (default) 1000. ``max_iter`` is maximum number of
        iterations that will be used to determine ``w_t`` to ``tol``.

    Returns: none

    References:
    :Title: The Lambert Way to Gaussianize heavy tailed data with the inverse of Tukey's h as a special case
    :Author: Georg M. Goerg
    :URL: https://arxiv.org/pdf/1010.2265.pdf

    """

    def __init__(self, tol=0.001, max_iter=1000):
        self.coefs_ = []  # Store tau for each transformed variable
        self.tol = tol
        self.max_iter = max_iter

    def _reset(self):
        self.coefs_ = []  # Store tau for each transformed variable
        return self

    def fit(self, X, **kwargs):
        self._reset()
        return self.train(X, **kwargs)

    def train(self, X, inplace=False, **kwargs):
        """
        Fit a one-parameter (``w_t``) family based on Lambert's W function that removes heavy tails from data. ``w_t`` will be determined iteratively (``max_iter``) to tolerance (``tol``) for each feature(column). Because of the iterative calulation of ``w_t`` ``LambertScaler`` can be slower by a factor of 10 than other scalers. 
                        
        Args:
            X (array-like):   Determine ``w_t`` of X, a 1-d or 2d numpy array. Shape assumed for array is (n-row) or (n-row,n-column).
        
        Raises:
            ValueError will result of not 1-D or 2-D numpy array, list or Pandas Dataframe.
            
            TypeError will result if not float or integer type.
            
        Returns:
            self
        """
        self._reset()

        for x_i in X.T:
            self.coefs_.append(_igmm(x_i, tol=self.tol, max_iter=self.max_iter))

        return self

    def transform(self, X, **kwargs):
        return self.predict(X, **kwargs)

    def predict(self, X, inplace=False, **kwargs):

        """
        Transform data using a previously learned Lambert coefficents for data.
        
        Args:
            X (array-like):  1-d or 2d numpy array. 

        Returns: 
            X: transformed by  Lambert's W function and returns 2-d numpy array.
        """

        return np.array([_w_t(x_i, tau_i) for x_i, tau_i in zip(X.T, self.coefs_)]).T

    def inverse_transform(self, y, inplace=False):
        """
        Reverse transformation of data from Lambert transformation.

        Args:
            y (DataFrame):  1-d or 2d numpy array.

        Returns: y, transformed back from Lambert's W function.
        """

        return self.inverse_predict(X)

    def inverse_predict(self, y, inplace=False):
        """
        Reverse transformation of data from Lambert transformation.
                
        Args:
            y (DataFrame):  1-d or 2d numpy array.

        Returns: y, transformed back from Lambert's W function.
        """

        return np.array(
            [_inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.coefs_)]
        ).T


__ScalerDict__["LambertScaler"] = LambertScaler

# paso Scaler to __ScalerDict__
class Scalers(pasoModel):
    """
    Parameters:
        encoderKey: (str) One of 'StandardScaler', 'MinMaxScaler', 'Normalizer', 'MaxAbsScaler', 'RobustScaler'
        , 'QuantileTransformer', 'BoxCoxScaler', 'LambertScaler'

        verbose: (str) (default) True, logging off (``verbose=False``)

    Note:
        **Scaling**

        Scaling means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1). If a feature set consists of only a single feature, then scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, then feature scaling provides the following benefits:
            - Helps gradient descent converge more quickly. (i.e. deep learning sic.)
            - Helps avoid the "NaN trap," in which one number in the model becomes a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.
            - Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much attention to the features having a wider range.
            - You don't have to give every floating-point feature exactly the same scale. Nothing terrible will happen if Feature A is scaled from -1 to +1 (Min/Max) while Feature B is scaled from -3 to +3 (normalizion, or Z-scaling). However, your model will react poorly if Feature B is scaled from 5000 to 100000.

        **Gaussian Scaling**

        This family of methods applys smooth, invertible transformations to some univariate data so that the distribution of the transformed data is as Gaussian as possible. This would/could be a pre-processing step for feature(s) upstream of futher data mangaling.
            - A standard pre-processing step is to "whiten" data by subtracting the mean and scaling it to have standard deviation 1. Gaussianized data has these properties and more.
            - Robust statistics / reduce effect of outliers. Lots of real world data exhibits long tails. For machine learning, the small number of examples in the tails of the distribution can have a large effect on results. Gaussianized data will "squeeze" the tails in towards the center.
            - Gaussian distributions are very well studied with many unique properties (because it is a well behaved function...( that dates back to Gauss :)
            - Many statistical models assume Gaussianity (trees being the exception)


    """

    def __init__(self, encoderKey, verbose=False, *args, **kwargs):
        super().__init__()
        if encoderKey in __ScalerDict__:
            Encoder = __ScalerDict__[encoderKey](*args)
        else:
            raise raise_PasoError(
                "paso:scale: No scaler named: {} found.".format(encoderKey)
            )
        self.encoderKey = encoderKey
        self.model = Encoder
        validate_bool_kwarg(verbose, "verbose")
        self.verbose = verbose

    def scalers(self):
        """
        Parameters:
            None

        Returns:
            List of available scaler names.
        """
        return list(__ScalerDict__.keys())

    @pasoDecorators.TrainWrap(array=True)
    def train(self, X, inplace=False, **kwargs):
        """

        Parameters:
            Xarg:  pandas dataFrame #Todo:Dask numpy

            inplace: (CURRENTLY IGNORED)
                    False (boolean), replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False
            
        Returns:
            self
        """

        self.model.fit(X, **kwargs)

        return self

    #
    @pasoDecorators.PredictWrap(array=True)
    def predict(self, X, inplace=False, **kwargs):
        """
        Parameters:
            Xarg:  pandas dataFrame #Todo:Dask numpy

            inplace: (CURRENTLY IGNORED)
                    False (boolean), replace 1st argument with resulting dataframe
                    True:  (boolean) ALWAYS False
            
        Returns:
            (DataFrame): transform X
        """

        self.f_x = self.model.transform(X, **kwargs)
        return self.f_x

    def inverse_predict(self, Xarg, inplace=False, **kwargs):
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
            raise raise_PasoError(
                "scale:inverse_transform: must call train and predict before inverse"
            )


########
