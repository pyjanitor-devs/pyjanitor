"""Miscellaneous internal PyJanitor helper functions."""

import functools
import os
import sys
import warnings
from itertools import chain, product
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

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
    :returns: ValueError if data is not the expected type.
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
    df, index, column_names, names_sep, names_pattern, names_to, values_to
):

    """
    This function raises errors or warnings if the arguments have the wrong
    python type, or if an unneeded argument is provided.
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
        warnings.warn(
            """pivot_longer is designed for single index dataframes and
               may produce unexpected results for multiIndex dataframes;
               for such cases, kindly use pandas.melt."""
        )

    if index is not None:
        if isinstance(index, str):
            index = [index]
        check("index", index, [list, tuple, Pattern])

    if column_names is not None:
        if isinstance(column_names, str):
            column_names = [column_names]
        check("column_names", column_names, [list, tuple, Pattern])

    if names_to is not None:
        check("names_to", names_to, [list, tuple, str])

        if isinstance(names_to, (list, tuple)):
            if not all(isinstance(word, str) for word in names_to):
                raise TypeError(
                    "All entries in `names_to` argument must be strings."
                )

            if len(names_to) > 1:
                if all((names_pattern is not None, names_sep is not None)):
                    raise ValueError(
                        """Only one of names_pattern or names_sep
                       should be provided."""
                    )
        if isinstance(names_to, str) or (len(names_to) == 1):
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
        check("names_pattern", names_pattern, [str, Pattern])

    if names_sep is not None:
        check("names_sep", names_sep, [str, Pattern])

    check("values_to", values_to, [str])

    return (
        df,
        index,
        column_names,
        names_sep,
        names_pattern,
        names_to,
        values_to,
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
    """

    if isinstance(column_names, Pattern):
        column_names = [col for col in df if column_names.search(col)]

    if isinstance(index, Pattern):
        index = [col for col in df if index.search(col)]

    return df, index, column_names


def _reindex_func(frame: pd.DataFrame, indexer=None) -> pd.DataFrame:
    """
    Function to reshape dataframe in pivot_longer, to try and make it look
    similar to the source data in terms of direction of the columns. It is a
    temporary measure until the minimum pandas version is 1.1, where we can
    take advantage of the `ignore_index` argument in `pd.melt`.

    Example: if columns are `id, ht1, ht2, ht3`, then depending on the
    arguments passed, the column in the reshaped dataframe, based on this
    function, will look like `1,2,3,1,2,3,1,2,3...`. This way, we ensure that
    for every index, there is a complete set of the data.

    A reindexed dataframe is returned.
    """

    if indexer is None:
        uniq_index_length = len(frame.drop_duplicates())
    else:
        uniq_index_length = len(frame.loc[:, indexer].drop_duplicates())
        if "index" in indexer:
            frame = frame.drop("index", axis=1)
    sorter = np.reshape(frame.index, (-1, uniq_index_length))
    # reshaped in Fortan order achieves the alternation
    sorter = np.ravel(sorter, order="F")
    return frame.reindex(sorter)


def _computations_pivot_longer(
    df: pd.DataFrame,
    index: Optional[Union[List, Tuple]] = None,
    column_names: Optional[Union[List, Tuple]] = None,
    names_sep: Optional[Union[str, Pattern]] = None,
    names_pattern: Optional[Union[str, Pattern]] = None,
    names_to: Optional[Union[List, Tuple, str]] = None,
    values_to: Optional[str] = "value",
) -> pd.DataFrame:
    """
    This is the main workhorse of the `pivot_longer` function.

    There are a couple of scenarios that this function takes care of when
    unpivoting :

    1. Regular data unpivoting is covered with pandas melt.
    2. if the length of `names_to` is > 1, the function unpivots the data,
       using `pd.melt`, and then separates into individual columns, using
       `str.split(expand=True)` if `names_sep` is provided or
       `str.extractall()` if `names_pattern is provided. The labels in
       `names_to` become the new column names.
    3. If `names_to` contains `.value`, then the function replicates
       `pd.wide_to_long`, using `pd.melt`. Unlike `pd.wide_to_long`, the
       stubnames do not have to be prefixes, they just need to match the
       position of `.value` in `names_to`. Just like in 2 above, the columns
       are separated into individual columns. The labels in the column
       corresponding to `.value` become the new column names, and override
       `values_to` in the process. The other extracted column stays
       (if len(`names_to`) is > 1), with the other name in `names_to` as
       its column name.

    The function also tries to emulate the way the source data is structured.
    Say data looks like this :
        id, a1, a2, a3, A1, A2, A3
         1, a, b, c, A, B, C

    when pivoted into long form, it will look like this :
              id instance    a     A
        0     1     1        a     A
        1     1     2        b     B
        2     1     3        c     C

    where the columns `a` comes before `A`, as it was in the source data,
    and in column `a`, `a > b > c`, also as it was in the source data.
    This also visually creates a complete set of the data per index.
    """

    if index is not None:
        check_column(df, index, present=True)
        # this should take care of non unique index
        # we'll get rid of the extra in _reindex_func
        # TODO: what happens if `index` is already a name
        # in the columns?
        if df.loc[:, index].duplicated().any():
            df = df.reset_index()
            index = ["index"] + index

    if column_names is not None:
        check_column(df, column_names, present=True)

    if index is None and (column_names is not None):
        index = df.columns.difference(column_names)

    # scenario 1
    if all((names_pattern is None, names_sep is None)):
        df = pd.melt(
            df,
            id_vars=index,
            value_vars=column_names,
            var_name=names_to,
            value_name=values_to,
        )

        # reshape in the order that the data appears
        # this should be easier to do with ignore_index in pandas version 1.1
        if index is not None:
            df = _reindex_func(df, index).reset_index(drop=True)
            return df.transform(pd.to_numeric, errors="ignore")
        return df

    # scenario 2
    if any((names_pattern is not None, names_sep is not None)):
        # should avoid conflict if index/columns has a string named `variable`
        uniq_name = "*^#variable!@?$%"
        df = pd.melt(
            df, id_vars=index, value_vars=column_names, var_name=uniq_name
        )

        # pd.melt returns uniq_name and value as the last columns. We can use
        # that knowledge to get the data before( the index column(s)),
        # the data between (our uniq_name column),
        #  and the data after (our values column)
        position = df.columns.get_loc(uniq_name)
        if position == 0:
            before_df = pd.DataFrame([], index=df.index)
        else:
            # just before uniq_name column
            before_df = df.iloc[:, :-2]
        after_df = df.iloc[:, -1].rename(values_to)
        between_df = df.pop(uniq_name)
        if names_sep is not None:
            between_df = between_df.str.split(names_sep, expand=True)
        else:
            between_df = between_df.str.extractall(names_pattern).droplevel(-1)
        # set_axis function labels argument takes only list-like objects
        if isinstance(names_to, str):
            names_to = [names_to]

        if len(names_to) != between_df.shape[-1]:
            raise ValueError(
                """
                Length of ``names_to`` does not match
                number of columns extracted.
                """
            )
        before_df = _reindex_func(before_df, index)
        between_df = between_df.set_axis(names_to, axis="columns")

        # we take a detour here to deal with paired columns, where the user
        # might want one of the names in the paired column as part of the
        # new column names. The `.value` indicates that that particular
        # value becomes a header.

        # It is also another way of achieving pandas wide_to_long.

        # Let's see an example of a paired column
        # say we have this data :
        # code is copied from pandas wide_to_long documentation
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html
        #     famid  birth  ht1  ht2
        # 0      1      1  2.8  3.4
        # 1      1      2  2.9  3.8
        # 2      1      3  2.2  2.9
        # 3      2      1  2.0  3.2
        # 4      2      2  1.8  2.8
        # 5      2      3  1.9  2.4
        # 6      3      1  2.2  3.3
        # 7      3      2  2.3  3.4
        # 8      3      3  2.1  2.9

        # and we want to reshape into data that looks like this :
        #      famid  birth age   ht
        # 0       1      1   1  2.8
        # 1       1      1   2  3.4
        # 2       1      2   1  2.9
        # 3       1      2   2  3.8
        # 4       1      3   1  2.2
        # 5       1      3   2  2.9
        # 6       2      1   1  2.0
        # 7       2      1   2  3.2
        # 8       2      2   1  1.8
        # 9       2      2   2  2.8
        # 10      2      3   1  1.9
        # 11      2      3   2  2.4
        # 12      3      1   1  2.2
        # 13      3      1   2  3.3
        # 14      3      2   1  2.3
        # 15      3      2   2  3.4
        # 16      3      3   1  2.1
        # 17      3      3   2  2.9

        # we have height(`ht`) and age(`1,2`) paired in the column name.
        # Note how `1, 2` is repeated for the extracted age column for each
        # combination of `famid` and `birth`. The repeat of `1,2` also
        # simulates how it looks in the source data : `ht1 > ht2`.
        # As such, for every index, there is a complete set of the data;
        # the user can visually see the unpivoted data for each index
        # and be assured of complete/accurate sync.
        # The code below achieves that.

        # scenario 3
        if ".value" in names_to:
            if names_to.count(".value") > 1:
                raise ValueError(
                    "Column name `.value` must not be duplicated."
                )
            # extract new column names and assign category dtype
            after_df_cols = pd.unique(between_df.loc[:, ".value"])
            dot_value_dtype = CategoricalDtype(after_df_cols, ordered=True)
            between_df = between_df.astype({".value": dot_value_dtype})
            if len(names_to) > 1:
                other_header = between_df.columns.difference([".value"])[0]
                other_header_values = pd.unique(
                    between_df.loc[:, other_header]
                )
                other_header_dtype = CategoricalDtype(
                    other_header_values, ordered=True
                )
                between_df = between_df.astype(
                    {other_header: other_header_dtype}
                )
                between_df = between_df.sort_values([".value", other_header])
            else:
                other_header = None
                other_header_values = None
                # index order not assured if just .value and quicksort
                between_df = between_df.sort_values(
                    [".value"], kind="mergesort"
                )

            # reshape index_sorter and use the first column as the index
            # of the reshaped after_df. after_df will be reshaped into
            # specific number of columns, based on the length of
            # `after_df_cols`
            index_sorter = between_df.index
            after_df = after_df.reindex(index_sorter).to_numpy()
            after_index = np.reshape(
                index_sorter, (-1, len(after_df_cols)), order="F"
            )
            after_index = after_index[:, 0]
            after_df = np.reshape(
                after_df, (-1, len(after_df_cols)), order="F"
            )
            after_df = pd.DataFrame(
                after_df, columns=after_df_cols, index=after_index
            )
            # if `names_to` has a length more than 1,
            # then we need to sort the other header, so that there is
            # an alternation, ensuring a complete representation of
            # each value per index.
            # if, however, `names_to` is of length 1, then between_df
            # will be an empty dataframe, and its index will be the
            # same as the index of `after_df`
            # once the indexes are assigned to before, after, and between
            # we can recombine with a join to get the proper alternation
            # and complete data per index/section
            if other_header:
                other_header_index = np.reshape(
                    after_index, (-1, len(other_header_values)), order="F"
                )
                other_header_index = np.ravel(other_header_index)
                between_df = between_df.loc[other_header_index, [other_header]]
            else:
                other_header_index = None
                between_df = pd.DataFrame([], index=after_index)
            if position == 0:  # no index or column_names supplied
                df = pd.DataFrame.join(between_df, after_df, how="inner")
            else:
                df = pd.DataFrame.join(
                    before_df, [between_df, after_df], how="inner"
                )
            return df.reset_index(drop=True).transform(
                pd.to_numeric, errors="ignore"
            )

        # this kicks in if there is no `.value` in `names_to`
        # here we reindex the before_df, to simulate the order of the columns
        # in the source data.
        df = pd.DataFrame.join(
            before_df, [between_df, after_df], how="inner"
        ).reset_index(drop=True)
        return df.transform(pd.to_numeric, errors="ignore")
