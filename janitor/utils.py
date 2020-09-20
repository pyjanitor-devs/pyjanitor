"""Miscellaneous internal PyJanitor helper functions."""

import functools
import os
import sys
import warnings
from itertools import chain, product
from typing import Callable, Dict, List, Pattern, Union
from collections import defaultdict


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

    # a better docstring is needed here

    """
    This function checks that the arguments meet the requirements
    before proceeding to the computation phase.
    """

    # put good description/comments for the checks
    # as well as the purpose/reason behind the checks
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
        if not isinstance(index, (list, tuple, str, Pattern)):
            raise TypeError(
                """index argument must be a list/tuple of columns,
                   a string, or a `patterns` function."""
            )
    if column_names is not None:
        if not isinstance(column_names, (list, tuple, str, Pattern)):
            raise TypeError(
                """column_names argument must be a list/tuple of columns,
                   a string, or a `patterns` function."""
            )
    if names_to is not None:
        if not isinstance(names_to, (list, tuple, str)):
            raise TypeError(
                """names_to argument must be a single string or
                   a list/tuple of strings."""
            )
        if isinstance(names_to, (list, tuple)) and (len(names_to) > 1):
            if all((names_pattern is not None, names_sep is not None)):
                raise ValueError(
                    """Only one of names_pattern or names_sep
                       should be provided."""
                )
        if isinstance(names_to, str) or (len(names_to) == 1):
            # names_sep creates more than one column
            # whereas names_pattern can be limited to one column
            if names_sep is not None:
                raise ValueError(
                    """
                    For a single names_to value,
                    names_sep is not required.
                    """
                )
    if names_pattern is not None:
        if not isinstance(names_pattern, str):
            raise TypeError(
                """names_pattern argument should be a
                   regular expression."""
            )
    if names_sep is not None:
        if not isinstance(names_sep, str):
            raise TypeError(
                """
                names_sep argument should be a string or
                regular expression.
                """
            )
    if not isinstance(values_to, str):
        raise TypeError("""values_to argument should be a string.""")

    return df


def _pivot_longer_pattern_match(df, index, column_names):
    """
    This checks if a pattern (regular expression) is supplied
    to index or columns and extracts the columns that match.
    """

    if isinstance(column_names, str):
        column_names = [column_names]
    # here we extract columns based on the regex passed
    if isinstance(column_names, Pattern):
        column_names = [col for col in df if column_names.search(col)]
    elif isinstance(column_names, Callable):
        column_names = [
            col
            for pattern, col in product(column_names, df)
            if pattern.search(col)
        ]

    if index is None and (column_names is not None):
        index = df.columns.difference(column_names)
    # if index is a regular expression
    elif isinstance(index, Pattern):
        index = [col for col in df if index.search(col)]
    elif isinstance(index, Callable):
        index = [
            col for pattern, col in product(index, df) if pattern.search(col)
        ]

    return df, index, column_names


def _computations_pivot_longer(
    df, index, column_names, names_sep, names_pattern, names_to, values_to
):

    # no frills, just shoot to pandas melt
    if all((names_pattern is None, names_sep is None)):
        return pd.melt(
            df,
            id_vars=index,
            value_vars=column_names,
            var_name=names_to,
            value_name=values_to,
            # introduce ignore_index argument when minimum version is 1.1
            # this will allow for easy sorting via the index
        )

    if any((names_pattern is not None, names_sep is not None)):

        # this ensures the wrong output is not provided for non-unique
        # index column(s)
        if index is not None:
            if df.loc[:, index].duplicated().any():
                raise ValueError(
                    """
                    The index variables need to uniquely identify each row.
                    """
                )
        # should avoid conflict if index/columns has a string named `variable`
        uniq_name = "*^#variable!@?$%"
        df = pd.melt(
            df, id_vars=index, value_vars=column_names, var_name=uniq_name
        )

        # melt returns uniq_name and value as the last columns. We can use
        # that knowledge to get the data before( the index column(s)),
        # the data between (our uniq_name column)
        #  and the data after (our values column)
        position = df.columns.get_loc(uniq_name)
        if position == 0:
            before = pd.DataFrame([], index=range(len(df)))
        else:
            # just before uniq_name column
            before = df.iloc[:, :-2]
        after = df.iloc[:, -1]
        between = df.pop(uniq_name)
        if names_sep is not None:
            between = between.str.split(names_sep, expand=True)
        else:
            between = between.str.extractall(names_pattern).reset_index(
                drop=True
            )
        # set_axis function labels argument takes only list-like objects
        if isinstance(names_to, str):
            names_to = [names_to]
        if len(names_to) != between.shape[-1]:
            raise ValueError(
                """
                Length of ``names_to`` does not match
                number of columns extracted.
                """
            )
        between = between.set_axis(names_to, axis="columns")

        # we take a detour here to deal with paired columns, where the user
        # might want one of the names in the paired column as part of the
        # new column names. The `.value` indicates that that particular
        # value becomes a header.It is also another way of achieving
        # pandas wide_to_long.

        # Let's see an example of a paired column
        # say we have this data :
        #     id  a1  a2  a3  A1  A2  A3
        # 0    1   a   b   c   A   B   C

        # and we want data that looks like this : 
        #     id instance   a   A
        # 0    1        1   a   A
        # 1    1        2   b   B
        # 2    1        3   c   C

        # In the reshaping process we need chunks where `a, b, c` is repeated
        # That way we get complete `chunks` of each extraction that can be
        # paired with the rest of the data, and be assured of complete/accurate
        # sync. The code below achieves that.

        if all((len(names_to) > 1, ".value" in names_to)):
            if names_to.count(".value") > 1:
                raise ValueError(
                    """Column name `.value` must not be duplicated."""
                )
            # Get the name in names_to that is not `.value`
            first_header = set(names_to).difference([".value"])
            # should be a single value, plus this method is significantly
            # faster than np.unique.
            # Not that it matters really for one item.
            first_header = next(iter(first_header))
            # aim here is to get the new column names in their present order
            # pd.unique does not sort, which serves our purpose
            value_headers = pd.unique(between.loc[:, ".value"])
            # reduced memory usage
            # but more importantly, it allows us to use the current state
            # to sort the data. This is necessary for reshaping later on.
            first_header_dtype = pd.api.types.CategoricalDtype(
                pd.unique(between.loc[:, first_header]), ordered=True
            )
            between.loc[:, first_header] = between.loc[:, first_header].astype(
                first_header_dtype
            )
            between = between.sort_values([".value", first_header])
            between_index = between.index

            len_value_headers = len(value_headers)
            len_unique_first_header = len(set(between.loc[:, first_header]))
            # this helps in getting the sorter, which will be used to ensure
            # the data in `before`, `after` and `between` align.
            number_of_columns = len_value_headers * len_unique_first_header

            # this gets us alternates of the non `.value`,
            # like `start,end,start,end` or `off, on, off, on`
            # and ensures proper sync of data
            sorter = np.reshape(
                between_index, (-1, number_of_columns), order="F"
            )
            sorter = np.ravel(sorter)

            # much easier and faster(I presume) to reorder using the index
            between = between.reindex(sorter)
            # this will serve as the headers for the after dataframe
            after_columns = between.loc[:, ".value"]

            between = between.loc[:, first_header]
            after = after.to_numpy()[sorter]

            # here we ensure correct pairing between the non `.value`
            # and the `after` data
            container = defaultdict(list)
            for k, v in zip(after_columns.to_numpy(), after):
                container[k].append(v)
            after = pd.DataFrame(container).reindex(columns=value_headers)

            if position == 0:  
                # pd.unique is used to ensure data is not sorted
                # which we need since we are aiming for alternation
                # `start, end`, `on,off`, `loc,lat,long` ...
                # which gets us a complete `chunk`
                between = np.resize(pd.unique(between), len(after))
                after.insert(0, first_header, between)
                return after

            before = before.reindex(sorter)
            before.loc[:, first_header] = between
            before = before.drop_duplicates(ignore_index=True)

            return pd.concat((before, after), axis=1)

        return pd.concat((before, between, after), axis=1)

    return df
