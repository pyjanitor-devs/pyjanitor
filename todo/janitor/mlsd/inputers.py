# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import urllib.request, os, glob

# from tqdm import tqdm
# from pandas.util._validators import validate_bool_kwarg
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# paso imports
from paso.base import pasoFunction, raise_PasoError, _array_to_string
from paso.base import pasoDecorators, _check_non_optional_kw, _dict_value

# from loguru import logger
import sys, os.path

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
#


def _formats_supported(path):
    for format in Inputers._formats_.keys():
        if path.endswith(format):
            return format
    raise raise_PasoError("format of this file not supported: {}".format(path))


def _url_path_exists(url):
    """
    Checks that a given URL is reachable.
    :Parameters:
        url: (str) url
    Returns: (bool)
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: "HEAD"

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def _inputer_exec(self, **kwargs):

    # must always be data = ' train' or no dataset =
    if self.dataset != "train" and ("train" not in kwargs):
        raise_PasoError(
            "dataset='{}'  not recognized: in  {} ".format(self.dataset, kwargs)
        )

    key = ["pre", "post"]
    if key[0] in kwargs and kwargs[key[0]] != None:
        for stmt in kwargs[key[0]]:
            exec(stmt)

    dfkey = "create-df"
    if dfkey in kwargs and kwargs[dfkey] != None:
        result = eval(kwargs[dfkey])

    if key[1] in kwargs and kwargs[key[1]] != "None":
        for stmt in kwargs[key[1]]:
            exec(stmt)

    return result


def _create_path(kw, dictnary, directory_path, default):

    if kw in dictnary:
        return directory_path + _dict_value(dictnary, kw, default)
    else:
        return default


# todo refactor this mess
def _inputer_cvs(self, **kwargs):
    kw = "names"
    self.names = _dict_value(kwargs, kw, [])

    kw = "directory_path"
    self.directory_path = _dict_value(kwargs, kw, "")

    kw = "train"
    if kw in kwargs:
        zzz = os.getcwd()
        print(zzz)
        xxx= glob.glob('*')
        self.train_path = self.directory_path + kwargs[kw]
        if os.path.exists(self.train_path) or _url_path_exists(self.train_path):
            if self.names != []:
                train = pd.read_csv(self.train_path, names=self.names)
            elif self.names == []:
                train = pd.read_csv(self.train_path)
        else:
            raise_PasoError(
                "Inputer train dataset path does not exist: {} or there might not be a directory_path:{}".format(
                    self.train_path, self.directory_path
                )
            )

    kw = "test"
    if kw in kwargs:
        self.test_path = self.directory_path + kwargs[kw]
        if os.path.exists(self.test_path):
            if self.names != []:
                test = pd.read_csv(self.test_path, names=self.names)
            elif self.names == []:
                test = pd.read_csv(self.test_path)
        else:
            raise_PasoError(
                "Inputer test dataset path does not exist: {}".format(self.test_path)
            )

    kw = "sampleSubmission"
    if kw in kwargs:
        self.sampleSubmission_path = self.directory_path + kwargs[kw]
        if os.path.exists(self.sampleSubmission_path):
            if self.names != []:
                sampleSubmission = pd.read_csv(
                    self.sampleSubmission_path, names=self.names
                )
            elif self.names == []:
                sampleSubmission = pd.read_csv(self.sampleSubmission_path)
        else:
            raise_PasoError(
                "Inputer sampleSubmission dataset path does not exist: {}".format(
                    self.test_path
                )
            )

    # no case in python
    if self.dataset == "train":
        return train
    elif self.dataset == "valid":
        return valid
    elif self.dataset == "test":
        return test
    elif self.dataset == "sampleSubmission":
        return sampleSubmission
    else:
        raise_PasoError("dataset not recognized: {} ".format(self.dataset))


def _inputer_xls(self, **kwargs):
    return None


def _inputer_xlsm(self, **kwargs):
    return None


def _inputer_text(self, **kwargs):
    return None


def _inputer_image2d(self, **kwargs):
    return None


def _inputer_image3d(self, **kwargs):
    return None


### Inputer
class Inputers(pasoFunction):

    """
    class to input file or url that is cvs or zip(cvs)
    or an error will be raised.

    parameters: None

    keywords:
        input_path: (str) the data source source path name.
            The path can be url or local. Format must be csv or csv/zip.

        target: the dependent feature name of this data_set.

        drop: (list) list of feature names to drop from
            dataset, X,y are then extracted from dataset.

    attributes set:
        self.target: (str)
        self.input_path = input_path

    returns:
        dataset: (DataFrame) complete dataset input from data source.
    """

    _formats_ = {
        "csv": True,
        "zip": True,
        "data": True,
        "sklearn.datasets": True,
        "yaml": True,
    }

    _inputer_ = {
        "exec": _inputer_exec,
        "cvs": _inputer_cvs,
        "xls": _inputer_xls,
        "xlsm": _inputer_xlsm,
        "text": _inputer_text,
        "image2D": _inputer_image2d,
        "image3D": _inputer_image3d,
    }

    _datasets_available_ = [
        "train",
        "valid",
        "test",
        "sampleSubmission",
        "directory_path",
    ]

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):

        """
        Parameters:
            filepath: (string)
            verbose: (boolean) (optiona) can be set. Default:True

        Note:

        """
        super().__init__()
        self.input_data_set = False

    @staticmethod
    def inputers():
        """
        Parameters:
            None

        Returns:
            List of available inputer names.
        """
        return [k for k in Inputers._inputer_.keys()]

    @staticmethod
    def formats():
        """
        Parameters:
            None

        Returns:
            List of available inputer names.
        """
        return [k for k in Inputers._formats_.keys()]

    @staticmethod
    def datasets():
        """
        List type of files available

        Parameters: None

        Returns: lists of datasets

        """
        return Inputers._datasets_available_

    @pasoDecorators.TTWrapNoArg(array=False)
    def transform(self, *args, **kwargs):
        # Todo:Rapids numpy
        """"
        main method to input file or url,
        or an error will be raised.

        parameters: None

        keywords:
            input_path: (str) the data source source path name.
                The path can be url or local. Format must be csv or csv/zip.

            target: the dependent feature name of this data_set.

            drop: (list) list of feature names to drop from
                dataset, X,y are then extracted from dataset.

        attributes set:
            self.target: (str)
            self.input_path = input_path

        returns:
            dataset: (DataFrame) complete dataset input from data source.
        """

        # currently support only one inputer, very brittle parser
        kwa = "target"
        self.target = _dict_value(self.kind_name_kwargs, kwa, None)
        _check_non_optional_kw(
            kwa, "Inputer: needs target keyword. probably not set in ontological file."
        )
        # currently just  can only be in inputer/transformkwarg
        kwa = "dataset"
        self.dataset = _dict_value(kwargs, kwa, "train")

        # create instance of this particular learner
        # checks for non-optional keyword
        if self.kind_name not in Inputers._inputer_:
            raise_PasoError(
                "transform; no format named: {} not in Inputers;: {}".format(
                    self.kind_name, Inputers._inputer_.keys()
                )
            )

        if _formats_supported(self.description_filepath):
            self.input_data_set = True
            return Inputers._inputer_[self.kind_name](self, **self.kind_name_kwargs)


### Splitters
class Splitters(pasoFunction):
    """
    Input returns dataset.
    Tne metadata is the instance attibutesof Inputer prperties.

    Note:

    Warning:

    """

    @pasoDecorators.InitWrap()
    def __init__(self, **kwargs):

        """
        Parameters:
            filepath: (string)
            verbose: (boolean) (optiona) can be set. Default:True

        Note:

        """
        super().__init__()
        self.inplace = False
        return self

    @pasoDecorators.TTWrapXy(array=False)
    def transform(self, X, y, **kwargs):
        # Todo:Rapids numpy
        """
        Parameters:
            target: dependent feature which is "target" of trainer

            Returns:
                    [train DF , test DF] SPLIT FROM X
            Raises:

            Note:
        """
        # note arrays
        # stratify =True them reset to y
        if "stratify" in self.kind_name_kwargs:
            self.kind_name_kwargs["stratify"] = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **self.kind_name_kwargs
        )

        return X_train, X_test, y_train, y_test


###
