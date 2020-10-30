"""Miscellaneous internal PyJanitor helper functions."""

import functools
import os
import sys
import warnings
from itertools import chain, product
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Union

import numpy as np
import pandas as pd

from .errors import JanitorError


def check(varname: str, value, expected_types: list):
    """
    One-liner syntactic sugar for checking types.

    Should be used like this::

        check('x', x, [int, float])

    :param varname: The name of the variable.
    :param value: The value of the varname.
    :param expected_types: The types we expect the item to be.
    :raises TypeError: if data is not the expected type.
    """
    is_expected_type = False
    for t in expected_types:
        if isinstance(value, t):
            is_expected_type = True
            break

    if not is_expected_type:
        raise TypeError(
            "{varname} should be one of {expected_types}".format(
                varname=varname, expected_types=expected_types
            )
        )


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
    df: pd.DataFrame, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
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
        columns=lambda x: _strip_underscores_func(x, strip_underscores)
    )
    return df


def _strip_underscores_func(
    col: str, strip_underscores: Union[str, bool] = None
) -> pd.DataFrame:
    """Strip underscores from a string."""
    underscore_options = [None, "left", "right", "both", "l", "r", True]
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )

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

    print(
        f"To use the janitor submodule {submodule}, you need to install "
        f"{package}."
    )
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
    if not func(df, *args, **kwargs) == func(
        func(df, *args, **kwargs), *args, **kwargs
    ):
        raise ValueError(
            "Supplied function is not idempotent for the given " "DataFrame."
        )


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
    Nothing is returned as the passed ``kwargs`` are modified directly.

    Implementation is inspired from `StackOverflow`_.

    .. _StackOverflow: https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias

    :param func_name: name of decorated function.
    :param kwargs: Arguments supplied to the method.
    :param aliases: Dictionary of aliases for a function's arguments.
    :raises TypeError: if both arguments are provided.
    """  # noqa: E501
    for old_alias, new_alias in aliases.items():
        if old_alias in kwargs:
            if new_alias in kwargs:
                raise TypeError(
                    f"{func_name} received both {old_alias} and {new_alias}"
                )
            warnings.warn(
                f"{old_alias} is deprecated; use {new_alias}",
                DeprecationWarning,
            )
            kwargs[new_alias] = kwargs.pop(old_alias)


def check_column(
    df: pd.DataFrame, old_column_names: List, present: bool = True
):
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
    :raises ValueError: if data is not the expected type.
    """
    for column_name in old_column_names:
        if present:
            if column_name not in df.columns:
                raise ValueError(
                    f"{column_name} not present in dataframe columns!"
                )
        else:  # Tests for exclusion
            if column_name in df.columns:
                raise ValueError(
                    f"{column_name} already present in dataframe columns!"
                )


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


def skiperror(
    f: Callable, return_x: bool = False, return_val=np.nan
) -> Callable:
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
    Additionally, ValueErrors are raised if empty containers are
    passed in as values into the dictionary.
    How each type is handled, and their associated exceptions,
    are pretty clear from the code.
    """
    # dictionary should not be empty
    if not entry:
        raise ValueError("passed dictionary cannot be empty")

    # couple of checks that should cause the program to fail early
    # if conditions are not met
    for _, value in entry.items():

        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("array cannot be empty")
            if value.ndim > 2:
                raise ValueError(
                    "expand_grid works only on 1D and 2D structures."
                )

        if isinstance(value, (pd.DataFrame, pd.Series)):
            if value.empty:
                raise ValueError("passed DataFrame cannot be empty")

        if isinstance(value, (list, tuple, set, dict)):
            if not value:
                raise ValueError("passed data cannot be empty")

    entry = {
        # If it is a scalar value, then wrap in a list
        # this is necessary, as we will use the itertools.product function
        # which works only on iterables.
        key: [value]
        if isinstance(value, (type(None), int, float, bool, str, np.generic))
        else value
        for key, value in entry.items()
    }

    return entry


def _grid_computation(entry: Dict) -> pd.DataFrame:
    """
    Return the final output of the expand_grid function as a dataframe.
     This kicks in after the ``_check_instance`` function is completed,
     and essentially creates a cross join of the values in the `entry`
     dictionary. If the `entry` dictionary is a collection of lists/tuples,
     then `itertools.product` will be used for the cross join, before a
    dataframe is created; if however, the `entry` contains a pandas dataframe
    or a pandas series or a numpy array, then identical indices are created for
    each entry and `pandas DataFrame join` is called to create the cross join.
    """

    # checks if the dictionary does not have any of
    # (pd.Dataframe, pd.Series, numpy) values and uses itertools.product.
    # numpy meshgrid is faster, but requires homogenous data to appreciate
    # the speed, and also to keep the data type for each column created.
    # As an example, if we have a mix in the dictionary of strings and numbers,
    # numpy will convert it to an object data type. Itertools product is
    # efficient and does not lose the data type.

    if not any(
        isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
        for key, value in entry.items()
    ):
        df_expand_grid = (value for key, value in entry.items())
        df_expand_grid = product(*df_expand_grid)
        return pd.DataFrame(df_expand_grid, columns=entry)

    # dictionary is a mix of different types - dataframe/series/numpy/...
    # so we check for each data type- if it is a pandas dataframe, then convert
    # to numpy and add to `df_expand_grid`; the other data types are added to
    # `df_expand_grid` as is. For each of the data types, new column names are
    # created if they do not have, and modified if names already exist. These
    # names are built through the for loop below and added to `df_columns`
    df_columns = []
    df_expand_grid = []
    for key, value in entry.items():
        if isinstance(value, pd.DataFrame):
            df_expand_grid.append(value.to_numpy())
            if isinstance(value.columns, pd.MultiIndex):
                df_columns.extend(
                    [f"{key}_{ind}" for ind, col in enumerate(value.columns)]
                )
            else:
                df_columns.extend([f"{key}_{col}" for col in value])
        elif isinstance(value, pd.Series):
            df_expand_grid.append(np.array(value))
            if value.name:
                df_columns.append(f"{key}_{value.name}")
            else:
                df_columns.append(str(key))
        elif isinstance(value, np.ndarray):
            df_expand_grid.append(value)
            if value.ndim == 1:
                df_columns.append(f"{key}_0")
            else:
                df_columns.extend(
                    [f"{key}_{ind}" for ind in range(value.shape[-1])]
                )
        else:
            df_expand_grid.append(value)
            df_columns.append(key)

        # here we run the product function from itertools only if there is
        # more than one item in the list; if only one item, we simply
        # create a dataframe with the new column names from `df_columns`
    if len(df_expand_grid) > 1:
        df_expand_grid = product(*df_expand_grid)
        df_expand_grid = (
            chain.from_iterable(
                [val]
                if not isinstance(val, (pd.DataFrame, pd.Series, np.ndarray))
                else val
                for val in value
            )
            for value in df_expand_grid
        )
        return pd.DataFrame(df_expand_grid, columns=df_columns)
    return pd.DataFrame(*df_expand_grid, columns=df_columns)


def _complete_groupings(df, list_of_columns):
    # this collects all the columns as individual labels, which will be
    # used to set the index of the dataframe
    index_columns = []
    # this will collect all the values associated with the respective
    # columns, and used to reindex the dataframe, to get the complete
    # pairings
    reindex_columns = []
    for item in list_of_columns:
        if not isinstance(item, (str, dict, list, tuple)):
            raise ValueError(
                """Value must either be a column label, a list/tuple of columns or a
                    dictionary where the keys are columns in the dataframe."""
            )
        if not item:
            raise ValueError("grouping cannot be empty")
        if isinstance(item, str):
            reindex_columns.append(set(df[item].array))
            index_columns.append(item)
        else:
            # this comes into play if we wish to input values that
            # do not exist in the data, say years, or alphabets, or
            # range of numbers
            if isinstance(item, dict):
                if len(item) > 1:
                    index_columns.extend(item.keys())
                else:
                    index_columns.append(*item.keys())
                    item_contents = [
                        # convert scalars to iterables; this is necessary
                        # when creating combinations with itertools' product
                        [value]
                        if isinstance(value, (int, float, str, bool))
                        else value
                        for key, value in item.items()
                    ]
                    reindex_columns.extend(item_contents)
            else:
                index_columns.extend(item)
                # TODO : change this to read as a numpy instead
                # instead of a list comprehension
                # it should be faster
                item = (df[sub_column].array for sub_column in item)
                item = set(zip(*item))
                reindex_columns.append(item)

    reindex_columns = product(*reindex_columns)
    # A list comprehension, coupled with itertools chain.from_iterable
    # would likely be faster; I fear that it may hamper readability with
    # nested list comprehensions; as such, I chose the for loop method.
    new_reindex_columns = []
    for row in reindex_columns:
        new_row = []
        for cell in row:
            if isinstance(cell, tuple):
                new_row.extend(cell)
            else:
                new_row.append(cell)
        new_reindex_columns.append(tuple(new_row))

    df = df.set_index(index_columns)

    return df, new_reindex_columns


def _data_checks_pivot_longer(
    df,
    index,
    column_names,
    names_to,
    values_to,
    names_sep,
    names_pattern,
    dtypes,
):

    """
    This function raises errors or warnings if the arguments have the wrong
    python type, or if an unneeded argument is provided. It also raises an
    error message if `names_pattern` is a list/tuple of regular expressions,
    and `names_to` is not a list/tuple, and the lengths do not match.
    This function is executed before proceeding to the computation phase.

    Type annotations are not provided because this function is where type
    checking happens.
    """

    if any(
        (
            isinstance(df.index, pd.MultiIndex),
            isinstance(df.columns, pd.MultiIndex),
        ),
    ):
        raise ValueError(
            """pivot_longer is designed for single index dataframes;
               for MultiIndex , kindly use pandas.melt."""
        )

    if index is not None:
        if isinstance(index, str):
            index = [index]
        check("index", index, [list, tuple, Pattern])

    if column_names is not None:
        if isinstance(column_names, str):
            column_names = [column_names]
        check("column_names", column_names, [list, tuple, Pattern])

    check("names_to", names_to, [list, tuple, str])

    if isinstance(names_to, str):
        names_to = [names_to]

    if isinstance(names_to, (list, tuple)):
        if not all(isinstance(word, str) for word in names_to):
            raise TypeError(
                "All entries in `names_to` argument must be strings."
            )

        if len(names_to) > 1:
            if all((names_pattern, names_sep)):
                raise ValueError(
                    """
                    Only one of names_pattern or names_sep
                    should be provided.
                    """
                )

            if all(
                (names_pattern is None, names_sep is None)
            ):  # write test for this
                raise ValueError(
                    """
                    If `names_to` is a list/tuple, then either
                    `names_sep` or `names_pattern` must be supplied.
                    """
                )

            if ".value" in names_to:
                if names_to.count(".value") > 1:
                    raise ValueError(
                        "Column name `.value` must not be duplicated."
                    )
        if len(names_to) == 1:
            # names_sep creates more than one column
            # whereas regex with names_pattern can be limited to one column
            if names_sep is not None:
                raise ValueError(
                    """
                    For a single names_to value,
                    names_sep is not required.
                    """
                )
    if names_pattern is not None:
        check("names_pattern", names_pattern, [str, Pattern, List, Tuple])
        if isinstance(names_pattern, (list, tuple)):
            if not all(
                isinstance(word, (str, Pattern)) for word in names_pattern
            ):
                raise TypeError(
                    """
                    All entries in ``names_pattern`` argument
                    must be regular expressions.
                    """
                )

            if len(names_pattern) != len(names_to):
                raise ValueError(
                    """
                    Length of ``names_to`` does not match
                    number of patterns.
                    """
                )

            if ".value" in names_to:
                raise ValueError(
                    """
                    ``.value`` not accepted if ``names_pattern``
                    is a list/tuple.
                    """
                )

    if names_sep is not None:
        check("names_sep", names_sep, [str, Pattern])

    if dtypes is not None:
        check("dtypes", dtypes, [dict])

    check("values_to", values_to, [str])

    return (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        dtypes,
    )


def _pivot_longer_pattern_match(
    df: pd.DataFrame,
    index: Optional[Union[str, Pattern]] = None,
    column_names: Optional[Union[str, Pattern]] = None,
) -> Tuple:
    """
    This checks if a pattern (regular expression) is supplied
    to index or columns and extracts the names that match the
    given regular expression.

    A dataframe, along with the `index` and `column_names` are
    returned.
    """

    # TODO: allow `janitor.patterns` to accept a list/tuple
    # of regular expresssions.
    if isinstance(column_names, Pattern):
        column_names = [col for col in df if column_names.search(col)]

    if isinstance(index, Pattern):
        index = [col for col in df if index.search(col)]

    return df, index, column_names


def _computations_pivot_longer(
    df: pd.DataFrame,
    index: Optional[Union[List, Tuple]] = None,
    column_names: Optional[Union[List, Tuple]] = None,
    names_to: Optional[Union[List, Tuple, str]] = None,
    values_to: Optional[str] = "value",
    names_sep: Optional[Union[str, Pattern]] = None,
    names_pattern: Optional[Union[str, Pattern]] = None,
    dtypes: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    This is the main workhorse of the `pivot_longer` function.
    The data is stacked, using ``pd.DataFrame.unstack``, to ensure
    that the order of appearance from the source data is maintained.

    If the length of `names_to` is > 1, or '.value' is in `names_to`,
    then the last level in the unpivoted dataframe's index is split,
    using `str.split(expand=True)`, if `names_sep` is provided.
    If `names_pattern` is provided, then either `pd.Series.str.extractall`
    or `pd.Series.str.contains` is used instead.

    If `names_to` contains `.value`, then categorical index is created,
    and a new dataframe created with this new index. The index level with
    the name `.value` is unstacked and becomes column names of the new
    dataframe.

    A dataframe is returned, with the index reset.
    """

    if index is not None:
        check_column(df, index, present=True)

    if column_names is not None:
        check_column(df, column_names, present=True)

    if index is None and (column_names is not None):
        index = [col for col in df if col not in column_names]

    if index:
        df = df.set_index(index)

    if column_names:
        df = df.filter(column_names)

    if all((names_pattern is None, names_sep is None)):
        # this ensures explicit missing variables are shown
        # in the final dataframe; comes in handy if there
        # are nulls in the original dataframe
        df = df.stack(dropna=False)  # noqa: PD013
        df = df.reset_index()
        if index:
            df.columns = index + names_to + [values_to]
        else:
            # this is necessary to exclude the reset index from
            # the final dataframe.
            df = df.iloc[:, 1:]
            df.columns = names_to + [values_to]
        if dtypes:
            df = df.astype(dtypes)

        return df

    mapping = None
    index_sorter = None
    columns_sorter = None
    extra_index = None
    drop_cols = None
    # splitting the columns before flipping is more efficient
    # if flipped before splitting, you have to deal with more rows
    # and string manipulations in Pandas are run within Python
    # so the larger the number of items the slower it will be.
    # however, if the columns are split before flipping,
    # we can take advantage of `stack`, which is vectorized
    if any((names_pattern, names_sep)):
        df.columns.names = ["._variable"]
        if names_sep:
            mapping = pd.Series(df.columns).str.split(names_sep, expand=True)

            if len(mapping.columns) != len(names_to):
                raise ValueError(
                    """
                    Length of ``names_to`` does not match
                    number of columns extracted.
                    """
                )

        else:
            if isinstance(names_pattern, str):
                mapping = df.columns.str.extract(names_pattern)
                if mapping.dropna().empty:
                    raise ValueError(
                        """
                        The regular expression in ``names_pattern``
                        did not return any matches.
                        """
                    )
                if len(names_to) != len(mapping.columns):
                    raise ValueError(
                        """
                        Length of ``names_to`` does not match
                        number of columns extracted.
                        """
                    )
            else:  # list/tuple of regular expressions
                mapping = [
                    df.columns.str.contains(regex) for regex in names_pattern
                ]
                if not np.any(mapping):
                    raise ValueError(
                        """
                        No match was returned for the regular expressions
                        in `names_pattern`.
                        """
                    )
                mapping = pd.DataFrame(np.select(mapping, names_to, None))
                mapping.columns = [".value"]

        if not isinstance(names_pattern, (list, tuple)):
            mapping.columns = names_to

        # attach to mapping, as it will be used as a join key
        # to preserve order of appearance
        mapping.index = df.columns

        if ".value" not in mapping.columns:
            # join keeps data in order of appearance
            df = mapping.join(
                df.stack(dropna=False).rename(values_to),  # noqa: PD013
                how="right",
                sort=False,
            )
            df = df.droplevel("._variable").reset_index()
            if dtypes:
                df = df.astype(dtypes)
            return df

        # '.value' kicks off here
        if df.index.duplicated().any():
            extra_index = pd.Index(np.arange(len(df)), name="._extra_index")
            df = df.set_index(extra_index, append=True)

        # avoids conflict in joins due to overlapping column names
        if values_to in mapping.columns:
            values_to = values_to + "_x"

        if mapping.duplicated().any():
            # creates unique indices so that unstack can occur
            mapping["._cumcount"] = mapping.groupby(".value").cumcount()

        # join keeps data in order of appearance
        df = mapping.join(
            df.stack(dropna=False).rename(values_to),  # noqa: PD013
            how="right",
            sort=False,
        )
        df = df.set_index(list(mapping.columns), append=True).droplevel(
            "._variable"
        )

        columns_sorter = mapping.loc[:, ".value"].unique()
        index_sorter = df.index.droplevel(".value")
        if index_sorter.duplicated().any():
            index_sorter = index_sorter.drop_duplicates()

        # unstack has an impact on performance as the data grows
        # possible touch point for improvement in the code
        df = (
            df.unstack(".value")  # noqa: PD010
            .droplevel(level=0, axis=1)
            .loc[index_sorter, columns_sorter]
            .rename_axis(columns=None)
            .reset_index()
        )

        drop_cols = [
            col for col in df if col in ("._extra_index", "._cumcount")
        ]
        if drop_cols:
            df = df.drop(drop_cols, axis=1)

        if not index:
            df = df.iloc[:, 1:]

        if dtypes:
            df = df.astype(dtypes)

        return df
