"""Miscellaneous internal PyJanitor helper functions."""

import functools
import os
import sys
import warnings
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rfn

from .errors import JanitorError


def check(varname: str, value, expected_types: list):
    """
    One-liner syntactic sugar for checking types.

    Should be used like this::

        check('x', x, [int, float])

    :param varname: The name of the variable.
    :param value: The value of the varname.
    :param expected_types: The types we expect the item to be.
    :returns: TypeError if data is not the expected type.
    """
    is_expected_type = False
    for t in expected_types:
        if isinstance(value, t):
            is_expected_type = True
            break

    if not is_expected_type:
        raise TypeError("{varname} should be one of {expected_types}".format(
            varname=varname, expected_types=expected_types))


def _clean_accounting_column(x: str) -> float:
    """
    Perform the logic for the `cleaning_style == "accounting"` attribute.

    This is a private function, not intended to be used outside of
    ``currency_column_to_numeric``.

    It is intended to be used in a pandas `apply` method.

    :returns: An object with a cleaned column.
    """
    y = x.strip()
    y = y.replace(",", "")
    y = y.replace(")", "")
    y = y.replace("(", "-")
    if y == "-":
        return 0.00
    return float(y)


def _currency_column_to_numeric(x, cast_non_numeric=None) -> str:
    """
    Perform logic for changing cell values.

    This is a private function intended to be used only in
    ``currency_column_to_numeric``.

    It is intended to be used in a pandas `apply` method, after being passed
    through `partial`.
    """
    acceptable_currency_characters = {
        "-",
        ".",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    }
    if len(x) == 0:
        return "ORIGINAL_NA"

    if cast_non_numeric:
        if x in cast_non_numeric.keys():
            check(
                "{%r: %r}" % (x, str(cast_non_numeric[x])),
                cast_non_numeric[x],
                [int, float],
            )
            return cast_non_numeric[x]
        else:
            return "".join(i for i in x if i in acceptable_currency_characters)
    else:
        return "".join(i for i in x if i in acceptable_currency_characters)


def _replace_empty_string_with_none(column_series):
    column_series.loc[column_series == ""] = None
    return column_series


def _replace_original_empty_string_with_none(column_series):
    column_series.loc[column_series == "ORIGINAL_NA"] = None
    return column_series


def _strip_underscores(
        df: pd.DataFrame,
        strip_underscores: Union[str, bool] = None) -> pd.DataFrame:
    """
    Strip underscores from DataFrames column names.

    Underscores can be stripped from the beginning, end or both.

    .. code-block:: python

        df = _strip_underscores(df, strip_underscores='left')

    :param df: The pandas DataFrame object.
    :param strip_underscores: (optional) Removes the outer underscores from all
        column names. Default None keeps outer underscores. Values can be
        either 'left', 'right' or 'both' or the respective shorthand 'l', 'r'
        and True.
    :returns: A pandas DataFrame with underscores removed.
    """
    df = df.rename(
        columns=lambda x: _strip_underscores_func(x, strip_underscores))
    return df


def _strip_underscores_func(col: str,
                            strip_underscores: Union[str, bool] = None
                            ) -> pd.DataFrame:
    """Strip underscores from a string."""
    underscore_options = [None, "left", "right", "both", "l", "r", True]
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}")

    if strip_underscores in ["left", "l"]:
        col = col.lstrip("_")
    elif strip_underscores in ["right", "r"]:
        col = col.rstrip("_")
    elif strip_underscores == "both" or strip_underscores is True:
        col = col.strip("_")
    return col


def import_message(
    submodule: str,
    package: str,
    conda_channel: str = None,
    pip_install: bool = False,
):
    """
    Return warning if package is not found.

    Generic message for indicating to the user when a function relies on an
    optional module / package that is not currently installed. Includes
    installation instructions. Used in `chemistry.py` and `biology.py`.

    :param submodule: pyjanitor submodule that needs an external dependency.
    :param package: External package this submodule relies on.
    :param conda_channel: Conda channel package can be installed from,
        if at all.
    :param pip_install: Whether package can be installed via pip.
    """
    is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    installable = True
    if is_conda:
        if conda_channel is None:
            installable = False
            installation = f"{package} cannot be installed via conda"
        else:
            installation = f"conda install -c {conda_channel} {package}"
    else:
        if pip_install:
            installation = f"pip install {package}"
        else:
            installable = False
            installation = f"{package} cannot be installed via pip"

    print(f"To use the janitor submodule {submodule}, you need to install "
          f"{package}.")
    print()
    if installable:
        print("To do so, use the following command:")
        print()
        print(f"    {installation}")
    else:
        print(f"{installation}")


def idempotent(func: Callable, df: pd.DataFrame, *args, **kwargs):
    """
    Raises error if a function operating on a `DataFrame` is not idempotent,
    that is, `func(func(df)) = func(df)` is not true for all `df`.

    :param func: A python method.
    :param df: A pandas `DataFrame`.
    :param args: Positional arguments supplied to the method.
    :param kwargs: Keyword arguments supplied to the method.
    :raises ValueError: If `func` is found to not be idempotent for the given
        `DataFrame` `df`.
    """
    if not func(df, *args, **kwargs) == func(func(df, *args, **kwargs), *args,
                                             **kwargs):
        raise ValueError("Supplied function is not idempotent for the given "
                         "DataFrame.")


def deprecated_alias(**aliases) -> Callable:
    """
    Used as a decorator when deprecating old function argument names, while
    keeping backwards compatibility.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

    Functional usage example:

    .. code-block:: python

        @deprecated_alias(a='alpha', b='beta')
        def simple_sum(alpha, beta):
            return alpha + beta

    :param aliases: Dictionary of aliases for a function's arguments.
    :return: Your original function wrapped with the kwarg redirection
        function.
    """  # noqa: E501

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def refactored_function(message: str) -> Callable:
    """Used as a decorator when refactoring functions

    Implementation is inspired from `Hacker Noon`_.

    .. Hacker Noon: https://hackernoon.com/why-refactoring-how-to-restructure-python-package-51b89aa91987

    Functional usage example:

    .. code-block:: python

        @refactored_function(
            message="simple_sum() has been refactored. Use hard_sum() instead."
        )
        def simple_sum(alpha, beta):
            return alpha + beta

    :param message: Message to use in warning user about refactoring.
    :return: Your original function wrapped with the kwarg redirection
        function.
    """  # noqa: E501

    def decorator(func):
        def emit_warning(*args, **kwargs):
            warnings.warn(message, FutureWarning)
            return func(*args, **kwargs)

        return emit_warning

    return decorator


def rename_kwargs(func_name: str, kwargs: Dict, aliases: Dict):
    """
    Used to update deprecated argument names with new names. Throws a
    TypeError if both arguments are provided, and warns if old alias is used.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

    :param func_name: name of decorated function.
    :param kwargs: Arguments supplied to the method.
    :param aliases: Dictionary of aliases for a function's arguments.
    :return: Nothing; the passed `kwargs` are modified directly.
    """  # noqa: E501
    for old_alias, new_alias in aliases.items():
        if old_alias in kwargs:
            if new_alias in kwargs:
                raise TypeError(
                    f"{func_name} received both {old_alias} and {new_alias}")
            warnings.warn(
                f"{old_alias} is deprecated; use {new_alias}",
                DeprecationWarning,
            )
            kwargs[new_alias] = kwargs.pop(old_alias)


def check_column(df: pd.DataFrame,
                 old_column_names: List,
                 present: bool = True):
    """
    One-liner syntactic sugar for checking the presence or absence of a column.

    Should be used like this::

        check(df, ['a', 'b'], present=True)

    :param df: The name of the variable.
    :param old_column_names: A list of column names we want to check to see if
        present (or absent) in df.
    :param present: If True (default), checks to see if all of old_column_names
        are in df.columns. If False, checks that none of old_column_names are
        in df.columns.
    :returns: ValueError if data is not the expected type.
    """
    for column_name in old_column_names:
        if present:
            if column_name not in df.columns:
                raise ValueError(
                    f"{column_name} not present in dataframe columns!")
        else:  # Tests for exclusion
            if column_name in df.columns:
                raise ValueError(
                    f"{column_name} already present in dataframe columns!")


def skipna(f: Callable) -> Callable:
    """
    Decorator for escaping np.nan and None in a function

    Should be used like this::

        df[column].apply(skipna(transform))

    or::

        @skipna
        def transform(x):
            pass

    :param f: the function to be wrapped
    :returns: _wrapped, the wrapped function
    """
    def _wrapped(x, *args, **kwargs):
        if (type(x) is float and np.isnan(x)) or x is None:
            return np.nan
        else:
            return f(x, *args, **kwargs)

    return _wrapped


def skiperror(f: Callable,
              return_x: bool = False,
              return_val=np.nan) -> Callable:
    """
    Decorator for escaping errors in a function

    Should be used like this::

        df[column].apply(
            skiperror(transform, return_val=3, return_x=False))

    or::

        @skiperror(return_val=3, return_x=False)
        def transform(x):
            pass

    :param f: the function to be wrapped
    :param return_x: whether or not the original value that caused error
        should be returned
    :param return_val: the value to be returned when an error hits.
        Ignored if return_x is True
    :returns: _wrapped, the wrapped function
    """
    def _wrapped(x, *args, **kwargs):
        try:
            return f(x, *args, **kwargs)
        except Exception:
            if return_x:
                return x
            return return_val

    return _wrapped


def _check_instance(entry: Dict):
    """
    Function to check instances in the expand_grid function.

    This checks if entry is a dictionary,
    checks the instance of value in key:value pairs in entry,
    and makes changes to other types as deemed necessary.

    Additionally, type-specific errors are raised
    if unsupported data types are passed in as values
    in the entry dictionary.

    How each type is handled, and their associated exceptions,
    are pretty clear from the code.
    """
    # dictionary should not be empty
    if not entry:
        raise ValueError("passed dictionary cannot be empty")
    # If it is a NoneType, number, Boolean, or string,
    # then wrap in a list
    entry = {
        key:
        [value] if isinstance(value,
                              (type(None), int, float, bool, str)) else value
        for key, value in entry.items()
    }

    # Convert to list if value is a set|tuple|range
    entry = {
        key: list(value) if isinstance(value, (set, tuple, range)) else value
        for key, value in entry.items()
    }

    # collect dataframes here
    dfs = []

    # collect non dataframes here, proper dicts
    dicts = {}

    for key, value in entry.items():

        # exclude dicts:
        if isinstance(value, dict):
            raise TypeError("Nested dictionaries are not allowed")

        # process arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("array cannot be empty")
            if value.ndim == 1:
                dfs.append(pd.DataFrame(value, columns=[key]))
            elif value.ndim == 2:
                dfs.append(pd.DataFrame(value).add_prefix(f"{key}_"))
            else:
                raise TypeError(
                    "`expand_grid` works with only vector and matrix arrays")
        # process series
        if isinstance(value, pd.Series):
            if value.empty:
                raise ValueError("passed Series cannot be empty")
            if not isinstance(value.index, pd.MultiIndex):
                # this section checks if the Series has a name or not
                # and uses that information to create a new column name
                # for the resulting Dataframe
                if value.name:
                    value = value.to_frame(name=f"{key}_{value.name}")
                    dfs.append(value)
                else:
                    value = value.to_frame(name=f"{key}")
                    dfs.append(value)
            else:
                raise TypeError(
                    "`expand_grid` does not work with pd.MultiIndex")
        # process dataframe
        if isinstance(value, pd.DataFrame):
            if value.empty:
                raise ValueError("passed DataFrame cannot be empty")
            if not (isinstance(value.index, pd.MultiIndex)
                    or isinstance(value.columns, pd.MultiIndex)):
                # add key to dataframe columns
                value = value.add_prefix(f"{key}_")
                dfs.append(value)
            else:
                raise TypeError(
                    "`expand_grid` does not work with pd.MultiIndex")
        # process lists
        if isinstance(value, list):
            if not value:
                raise ValueError("passed Sequence cannot be empty")
            if np.array(value).ndim == 1:
                checklist = (type(None), str, int, float, bool)
                instance_check_type = (isinstance(internal, checklist)
                                       for internal in value)
                if all(instance_check_type):
                    dicts.update({key: value})
                else:
                    raise ValueError("values in iterable must be scalar")
            elif np.array(value).ndim == 2:
                value = pd.DataFrame(value).add_prefix(f"{key}_")
                dfs.append(value)
            else:
                raise ValueError("Sequence's dimension should be 1d or 2d")

    return dfs, dicts


def _grid_computation_dict(dicts: Dict) -> pd.DataFrame:
    """
    Function used within the expand_grid function,
    to compute dataframe from values that are not dataframes/arrays/series.
    These values are collected into a dictionary,
    and processed with numpy meshgrid.

    Numpy's meshgrid is faster than itertools' product,
    and when converting to a dataframe, is fast as well.
    Structured arrays are used here, to ensure the datatypes are preserved.
    """
    # if there is only name value pair in the dictionary
    if len(dicts) == 1:
        key = list(dicts.keys())[0]
        value = list(dicts.values())[0]
        final = pd.DataFrame(value, columns=[key])
    # if there are more than one name value pair
    else:
        res = np.meshgrid(*dicts.values())
        # create structured array
        # keeps data type of each value in the dict
        outcome = np.core.records.fromarrays(res, names=",".join(dicts))
        # reshape into a 1 column array
        # using the size of any of the arrays obtained
        # from the meshgrid computation
        outcome = np.reshape(outcome, (np.size(res[0]), 1))
        # flatten structured array into 1d array
        outcome = np.concatenate(outcome)
        # sort array
        outcome.sort(axis=0, order=list(dicts))
        # create dataframe
        # structed array already has names,
        # this gets transferred as column names
        final = pd.DataFrame.from_records(outcome)
    return final


def _compute_two_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the cartesian product of two Dataframes.

    Used by the expand_grid function.

    Numpy is employed here, to get faster computations,
    compared to running a many-to-many join with pandas merge.

    Structured arrays are employed, to preserve data type.
    """
    # get lengths of dataframes(number of rows) and swap
    # essentially we'll pair one dataframe with the other's length:
    lengths = reversed([ent.index.size for ent in (df1, df2)])

    # grab the maximum string length
    string_cols = [
        frame.select_dtypes(include="object").columns for frame in (df1, df2)
    ]

    # pair max string length with col
    # will be passed into frame.to_records,
    # to get dtype in numpy recarray
    string_cols = [{col: f"<U{frame[col].str.len().max()}"
                    for col in ent}
                   for ent, frame in zip(string_cols, (df1, df2))]

    # pair length, column data type and dataframe
    (len_first, col_dtypes,
     first), (len_last, col_dtypes,
              last) = list(zip(lengths, string_cols, (df1, df2)))

    # export to numpy as recarray,
    # ensuring that the column data types are captured
    # this is particularly relevant to object data type
    first = first.to_records(column_dtypes=col_dtypes, index=False)

    # tile first with len_first
    # remember, len_first is the length of the other dataframe
    first = np.tile(first, (len_first, 1))

    # get a 1d array
    first = np.concatenate(first)

    # sorting here ensures we get each row of the first
    # with the entire rows of the other dataframe
    np.recarray.sort(first, order=first.dtype.names[0])

    # same process as first, except there'll be no sorting
    last = last.to_records(column_dtypes=col_dtypes, index=False)
    last = np.tile(last, (len_last, 1))
    last = np.concatenate(last)

    # merge first and last
    # and return a dataframe
    result = rfn.merge_arrays((first, last), flatten=True, asrecarray=True)
    return pd.DataFrame.from_records(result)


def _grid_computation_list(dfs: List):
    """
    Computes cartesian product of Dataframes in the expand_grid function.

    This builds on _compute_two_dfs function,
    by applying it to more two or more Dataframes.
    """
    return functools.reduce(_compute_two_dfs, dfs)


def _grid_computation(dfs: List, dicts: Dict) -> pd.DataFrame:
    """
    Return the final output of the expand_grid function.
    """
    if not dicts:
        result = _grid_computation_list(dfs)
    elif not dfs:
        result = _grid_computation_dict(dicts)
    else:
        dfs.append(_grid_computation_dict(dicts))
        result = _grid_computation_list(dfs)
    return result
