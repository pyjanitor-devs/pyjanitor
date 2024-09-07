from __future__ import annotations

import fnmatch
import inspect
import re
from collections.abc import Callable as dispatch_callable
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import pandas_flavor as pf
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_dtype,
    is_list_like,
    is_scalar,
)
from pandas.core.common import is_bool_indexer
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

from janitor.functions.utils import _is_str_or_cat
from janitor.utils import check, deprecated_alias, refactored_function


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `jn.select` instead."
    )
)
def select_columns(
    df: pd.DataFrame,
    *args: Any,
    invert: bool = False,
) -> pd.DataFrame:
    """Method-chainable selection of columns.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    Selection on a MultiIndex on a level, or multiple levels,
    is possible with a dictionary.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of columns available as well.

    !!!note

        The preferred option when selecting columns or rows in a Pandas DataFrame
        is with `.loc` or `.iloc` methods.
        `select_columns` is primarily for convenience.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `jn.select` instead.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> from numpy import nan
        >>> pd.set_option("display.max_columns", None)
        >>> pd.set_option("display.expand_frame_repr", False)
        >>> pd.set_option("max_colwidth", None)
        >>> data = {'name': ['Cheetah','Owl monkey','Mountain beaver',
        ...                  'Greater short-tailed shrew','Cow'],
        ...         'genus': ['Acinonyx', 'Aotus', 'Aplodontia', 'Blarina', 'Bos'],
        ...         'vore': ['carni', 'omni', 'herbi', 'omni', 'herbi'],
        ...         'order': ['Carnivora','Primates','Rodentia','Soricomorpha','Artiodactyla'],
        ...         'conservation': ['lc', nan, 'nt', 'lc', 'domesticated'],
        ...         'sleep_total': [12.1, 17.0, 14.4, 14.9, 4.0],
        ...         'sleep_rem': [nan, 1.8, 2.4, 2.3, 0.7],
        ...         'sleep_cycle': [nan, nan, nan, 0.133333333, 0.666666667],
        ...         'awake': [11.9, 7.0, 9.6, 9.1, 20.0],
        ...         'brainwt': [nan, 0.0155, nan, 0.00029, 0.423],
        ...         'bodywt': [50.0, 0.48, 1.35, 0.019, 600.0]}
        >>> df = pd.DataFrame(data)
        >>> df
                                 name       genus   vore         order  conservation  sleep_total  sleep_rem  sleep_cycle  awake  brainwt   bodywt
        0                     Cheetah    Acinonyx  carni     Carnivora            lc         12.1        NaN          NaN   11.9      NaN   50.000
        1                  Owl monkey       Aotus   omni      Primates           NaN         17.0        1.8          NaN    7.0  0.01550    0.480
        2             Mountain beaver  Aplodontia  herbi      Rodentia            nt         14.4        2.4          NaN    9.6      NaN    1.350
        3  Greater short-tailed shrew     Blarina   omni  Soricomorpha            lc         14.9        2.3     0.133333    9.1  0.00029    0.019
        4                         Cow         Bos  herbi  Artiodactyla  domesticated          4.0        0.7     0.666667   20.0  0.42300  600.000

        Explicit label selection:
        >>> df.select_columns('name', 'order')
                                 name         order
        0                     Cheetah     Carnivora
        1                  Owl monkey      Primates
        2             Mountain beaver      Rodentia
        3  Greater short-tailed shrew  Soricomorpha
        4                         Cow  Artiodactyla

        Selection via globbing:
        >>> df.select_columns("sleep*", "*wt")
           sleep_total  sleep_rem  sleep_cycle  brainwt   bodywt
        0         12.1        NaN          NaN      NaN   50.000
        1         17.0        1.8          NaN  0.01550    0.480
        2         14.4        2.4          NaN      NaN    1.350
        3         14.9        2.3     0.133333  0.00029    0.019
        4          4.0        0.7     0.666667  0.42300  600.000

        Selection via regex:
        >>> import re
        >>> df.select_columns(re.compile(r"o.+er"))
                  order  conservation
        0     Carnivora            lc
        1      Primates           NaN
        2      Rodentia            nt
        3  Soricomorpha            lc
        4  Artiodactyla  domesticated

        Selection via slicing:
        >>> df.select_columns(slice('name','order'), slice('sleep_total','sleep_cycle'))
                                 name       genus   vore         order  sleep_total  sleep_rem  sleep_cycle
        0                     Cheetah    Acinonyx  carni     Carnivora         12.1        NaN          NaN
        1                  Owl monkey       Aotus   omni      Primates         17.0        1.8          NaN
        2             Mountain beaver  Aplodontia  herbi      Rodentia         14.4        2.4          NaN
        3  Greater short-tailed shrew     Blarina   omni  Soricomorpha         14.9        2.3     0.133333
        4                         Cow         Bos  herbi  Artiodactyla          4.0        0.7     0.666667

        Selection via callable:
        >>> from pandas.api.types import is_numeric_dtype
        >>> df.select_columns(is_numeric_dtype)
           sleep_total  sleep_rem  sleep_cycle  awake  brainwt   bodywt
        0         12.1        NaN          NaN   11.9      NaN   50.000
        1         17.0        1.8          NaN    7.0  0.01550    0.480
        2         14.4        2.4          NaN    9.6      NaN    1.350
        3         14.9        2.3     0.133333    9.1  0.00029    0.019
        4          4.0        0.7     0.666667   20.0  0.42300  600.000
        >>> df.select_columns(lambda f: f.isna().any())
           conservation  sleep_rem  sleep_cycle  brainwt
        0            lc        NaN          NaN      NaN
        1           NaN        1.8          NaN  0.01550
        2            nt        2.4          NaN      NaN
        3            lc        2.3     0.133333  0.00029
        4  domesticated        0.7     0.666667  0.42300

        Exclude columns with the `invert` parameter:
        >>> df.select_columns(is_numeric_dtype, invert=True)
                                 name       genus   vore         order  conservation
        0                     Cheetah    Acinonyx  carni     Carnivora            lc
        1                  Owl monkey       Aotus   omni      Primates           NaN
        2             Mountain beaver  Aplodontia  herbi      Rodentia            nt
        3  Greater short-tailed shrew     Blarina   omni  Soricomorpha            lc
        4                         Cow         Bos  herbi  Artiodactyla  domesticated

        Exclude columns with the `DropLabel` class:
        >>> from janitor import DropLabel
        >>> df.select_columns(DropLabel(slice("name", "awake")), "conservation")
           brainwt   bodywt  conservation
        0      NaN   50.000            lc
        1  0.01550    0.480           NaN
        2      NaN    1.350            nt
        3  0.00029    0.019            lc
        4  0.42300  600.000  domesticated

        Selection on MultiIndex columns:
        >>> d = {'num_legs': [4, 4, 2, 2],
        ...      'num_wings': [0, 0, 2, 2],
        ...      'class': ['mammal', 'mammal', 'mammal', 'bird'],
        ...      'animal': ['cat', 'dog', 'bat', 'penguin'],
        ...      'locomotion': ['walks', 'walks', 'flies', 'walks']}
        >>> df = pd.DataFrame(data=d)
        >>> df = df.set_index(['class', 'animal', 'locomotion']).T
        >>> df
        class      mammal                bird
        animal        cat   dog   bat penguin
        locomotion  walks walks flies   walks
        num_legs        4     4     2       2
        num_wings       0     0     2       2

        Selection with a scalar:
        >>> df.select_columns('mammal')
        class      mammal
        animal        cat   dog   bat
        locomotion  walks walks flies
        num_legs        4     4     2
        num_wings       0     0     2

        Selection with a tuple:
        >>> df.select_columns(('mammal','bat'))
        class      mammal
        animal        bat
        locomotion  flies
        num_legs        2
        num_wings       2

        Selection within a level is possible with a dictionary,
        where the key is either a level name or number:
        >>> df.select_columns({'animal':'cat'})
        class      mammal
        animal        cat
        locomotion  walks
        num_legs        4
        num_wings       0
        >>> df.select_columns({1:["bat", "cat"]})
        class      mammal
        animal        bat   cat
        locomotion  flies walks
        num_legs        2     4
        num_wings       2     0

        Selection on multiple levels:
        >>> df.select_columns({"class":"mammal", "locomotion":"flies"})
        class      mammal
        animal        bat
        locomotion  flies
        num_legs        2
        num_wings       2

        Selection with a regex on a level:
        >>> df.select_columns({"animal":re.compile(".+t$")})
        class      mammal
        animal        cat   bat
        locomotion  walks flies
        num_legs        4     2
        num_wings       0     2

        Selection with a callable on a level:
        >>> df.select_columns({"animal":lambda f: f.str.endswith('t')})
        class      mammal
        animal        cat   bat
        locomotion  walks flies
        num_legs        4     2
        num_wings       0     2

    Args:
        df: A pandas DataFrame.
        *args: Valid inputs include: an exact column name to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection
            on a MultiIndex on different levels.
        invert: Whether or not to invert the selection.
            This will result in the selection
            of the complement of the columns provided.

    Returns:
        A pandas DataFrame with the specified columns selected.
    """  # noqa: E501

    return _select(df, columns=list(args), invert=invert)


@pf.register_dataframe_method
@refactored_function(
    message=(
        "This function will be deprecated in a 1.x release. "
        "Please use `jn.select` instead."
    )
)
def select_rows(
    df: pd.DataFrame,
    *args: Any,
    invert: bool = False,
) -> pd.DataFrame:
    """Method-chainable selection of rows.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    Selection on a MultiIndex on a level, or multiple levels,
    is possible with a dictionary.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of rows available as well.


    !!! info "New in version 0.24.0"

    !!!note

        The preferred option when selecting columns or rows in a Pandas DataFrame
        is with `.loc` or `.iloc` methods, as they are generally performant.
        `select_rows` is primarily for convenience.

    !!!note

        This function will be deprecated in a 1.x release.
        Please use `jn.select` instead.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = {"col1": [1, 2], "foo": [3, 4], "col2": [5, 6]}
        >>> df = pd.DataFrame.from_dict(df, orient='index')
        >>> df
              0  1
        col1  1  2
        foo   3  4
        col2  5  6
        >>> df.select_rows("col*")
              0  1
        col1  1  2
        col2  5  6

    More examples can be found in the
    [`select_columns`][janitor.functions.select.select_columns] section.

    Args:
        df: A pandas DataFrame.
        *args: Valid inputs include: an exact index name to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection
            on a MultiIndex on different levels.
        invert: Whether or not to invert the selection.
            This will result in the selection
            of the complement of the rows provided.

    Returns:
        A pandas DataFrame with the specified rows selected.
    """  # noqa: E501
    return _select(df, rows=list(args), invert=invert)


@pf.register_dataframe_method
@deprecated_alias(rows="index")
def select(
    df: pd.DataFrame,
    *args: tuple,
    index: Any = None,
    columns: Any = None,
    axis: str = "columns",
    invert: bool = False,
) -> pd.DataFrame:
    """Method-chainable selection of rows and columns.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    Selection on a MultiIndex on a level, or multiple levels,
    is possible with a dictionary.

    This method does not mutate the original DataFrame.

    Selection can be inverted with the `DropLabel` class.

    Optional ability to invert selection of index/columns available as well.


    !!! info "New in version 0.24.0"


    !!!note

        The preferred option when selecting columns or rows in a Pandas DataFrame
        is with `.loc` or `.iloc` methods, as they are generally performant.
        `select` is primarily for convenience.

    !!! abstract "Version Changed"

        - 0.26.0
            - Added variable `args`, `invert` and `axis` parameters.
            - `rows` keyword deprecated in favour of `index`.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=['cobra', 'viper', 'sidewinder'],
        ...      columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8
        >>> df.select(index='cobra', columns='shield')
               shield
        cobra       2

        Labels can be dropped with the `DropLabel` class:

        >>> df.select(index=DropLabel('cobra'))
                    max_speed  shield
        viper               4       5
        sidewinder          7       8

    More examples can be found in the
    [`select_columns`][janitor.functions.select.select_columns] section.

    Args:
        df: A pandas DataFrame.
        *args: Valid inputs include: an exact index name to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection
            on a MultiIndex on different levels.
        index: Valid inputs include: an exact label to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection
            on a MultiIndex on different levels.
        columns: Valid inputs include: an exact label to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection
            on a MultiIndex on different levels.
        invert: Whether or not to invert the selection.
            This will result in the selection
            of the complement of the rows/columns provided.
        axis: Whether the selection should be on the index('index'),
            or columns('columns').
            Applicable only for the variable args parameter.

    Raises:
        ValueError: If args and index/columns are provided.

    Returns:
        A pandas DataFrame with the specified rows and/or columns selected.
    """  # noqa: E501

    if args:
        check("invert", invert, [bool])
        if (index is not None) or (columns is not None):
            raise ValueError(
                "Either provide variable args with the axis parameter, "
                "or provide arguments to the index and/or columns parameters."
            )
        if axis == "index":
            return _select(df, rows=list(args), columns=columns, invert=invert)
        if axis == "columns":
            return _select(df, columns=list(args), rows=index, invert=invert)
        raise ValueError("axis should be either 'index' or 'columns'.")
    return _select(df, rows=index, columns=columns, invert=invert)


def get_index_labels(
    arg: Any, df: pd.DataFrame, axis: Literal["index", "columns"]
) -> pd.Index:
    """Convenience function to get actual labels from column/index

    !!! info "New in version 0.25.0"

    Args:
        arg: Valid inputs include: an exact column name to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection
            on a MultiIndex on different levels.
        df: The pandas DataFrame object.
        axis: Should be either `index` or `columns`.

    Returns:
        A pandas Index.
    """
    assert axis in {"index", "columns"}
    index = getattr(df, axis)
    return index[_select_index(arg, df, axis)]


def get_columns(
    group: DataFrameGroupBy | SeriesGroupBy, label: Any
) -> DataFrameGroupBy | SeriesGroupBy:
    """
    Helper function for selecting columns on a grouped object,
    using the
    [`select`][janitor.functions.select.select] syntax.

    !!! info "New in version 0.25.0"

    Args:
        group: A Pandas GroupBy object.
        label: column(s) to select.

    Returns:
        A pandas groupby object.
    """
    check("groupby object", group, [DataFrameGroupBy, SeriesGroupBy])
    label = get_index_labels(label, group.obj, axis="columns")
    label = label if is_scalar(label) else list(label)
    return group[label]


def _select_regex(index, arg, source="regex"):
    "Process regex on a Pandas Index"
    assert source in ("fnmatch", "regex"), source
    try:
        if source == "fnmatch":
            arg, regex = arg
            bools = index.str.match(regex, na=False)
        else:
            bools = index.str.contains(arg, na=False, regex=True)
        if not bools.any():
            raise KeyError(f"No match was returned for '{arg}'")
        return bools
    except Exception as exc:
        raise KeyError(f"No match was returned for '{arg}'") from exc


def _select_callable(arg, func: Callable, axis=None):
    """
    Process a callable on a Pandas DataFrame/Index.
    """
    bools = func(arg)
    bools = np.asanyarray(bools)
    if not is_bool_dtype(bools):
        raise ValueError(
            "The output of the applied callable "
            "should be a 1-D boolean array."
        )
    if axis:
        arg = getattr(arg, axis)
    if len(bools) != len(arg):
        raise IndexError(
            f"The boolean array output from the callable {arg} "
            f"has wrong length: "
            f"{len(bools)} instead of {len(arg)}"
        )
    return bools


@dataclass
class DropLabel:
    """Helper class for removing labels within the `select` syntax.

    `label` can be any of the types supported in the `select`,
    `select_rows` and `select_columns` functions.
    An array of integers not matching the labels is returned.

    !!! info "New in version 0.24.0"

    Args:
        label: Label(s) to be dropped from the index.
    """

    label: Any


@singledispatch
def _select_index(arg, df, axis):
    """Base function for selection on a Pandas Index object.

    Returns either an integer, a slice,
    a sequence of booleans, or an array of integers,
    that match the exact location of the target.
    """
    try:
        return getattr(df, axis).get_loc(arg)
    except Exception as exc:
        raise KeyError(f"No match was returned for {arg}") from exc


@_select_index.register(str)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """Base function for selection on a Pandas Index object.

    Applies only to strings.
    It is also applicable to shell-like glob strings,
    which are supported by `fnmatch`.

    Returns either a sequence of booleans, an integer,
    or a slice.
    """
    index = getattr(df, axis)
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values(0)
    if _is_str_or_cat(index) or is_datetime64_dtype(index):
        try:
            return index.get_loc(arg)
        except KeyError as exc:
            if _is_str_or_cat(index):
                if arg == "*":
                    return slice(None)
                # label selection should be case sensitive
                # fix for Github Issue 1160
                # translating to regex solves the case sensitivity
                # and also avoids the list comprehension
                # not that list comprehension is bad - i'd say it is efficient
                # however, the Pandas str.match method used in _select_regex
                # could offer more performance, especially if the
                # underlying array of the index is a PyArrow string array
                return _select_regex(
                    index, (arg, fnmatch.translate(arg)), source="fnmatch"
                )
            raise KeyError(f"No match was returned for '{arg}'") from exc
    raise KeyError(f"No match was returned for '{arg}'")


@_select_index.register(re.Pattern)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """Base function for selection on a Pandas Index object.

    Applies only to regular expressions.
    `re.compile` is required for the regular expression.

    Returns an array of booleans.
    """
    index = getattr(df, axis)
    if isinstance(index, pd.MultiIndex):
        index = index.get_level_values(0)
    return _select_regex(index, arg)


@_select_index.register(range)  # noqa: F811
@_select_index.register(slice)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to slices.

    Returns a slice object.
    """
    index = getattr(df, axis)
    if not index.is_monotonic_increasing:
        if not index.is_unique:
            raise ValueError(
                "Non-unique Index labels should be monotonic increasing."
                "Kindly sort the index."
            )
        if is_datetime64_dtype(index):
            raise ValueError(
                "The DatetimeIndex should be monotonic increasing."
                "Kindly sort the index"
            )

    return index._convert_slice_indexer(arg, kind="loc")


@_select_index.register(dispatch_callable)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to callables.

    The callable is applied to the entire DataFrame.

    Returns an array of booleans.
    """
    # special case for selecting dtypes columnwise
    dtypes = (
        arg.__name__
        for _, arg in inspect.getmembers(pd.api.types, inspect.isfunction)
        if arg.__name__.startswith("is") and arg.__name__.endswith("type")
    )
    if (arg.__name__ in dtypes) and (axis == "columns"):
        bools = df.dtypes.map(arg)
        return np.asanyarray(bools)

    return _select_callable(df, arg, axis)


@_select_index.register(dict)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to a dictionary.

    Returns an array of integers.
    """
    level_label = {}
    index = getattr(df, axis)
    if not isinstance(index, pd.MultiIndex):
        return _select_index(list(arg), df, axis)
    all_str = (isinstance(entry, str) for entry in arg)
    all_str = all(all_str)
    all_int = (isinstance(entry, int) for entry in arg)
    all_int = all(all_int)
    if not all_str | all_int:
        raise TypeError(
            "The keys in the dictionary represent the levels "
            "in the MultiIndex, and should either be all "
            "strings or integers."
        )
    for key, value in arg.items():
        if isinstance(value, dispatch_callable):
            indexer = index.get_level_values(key)
            value = _select_callable(indexer, value)
        elif isinstance(value, re.Pattern):
            indexer = index.get_level_values(key)
            value = _select_regex(indexer, value)
        level_label[key] = value

    level_label = {
        index._get_level_number(level): label
        for level, label in level_label.items()
    }
    level_label = [
        level_label.get(num, slice(None)) for num in range(index.nlevels)
    ]
    return index.get_locs(level_label)


@_select_index.register(np.ndarray)  # noqa: F811
@_select_index.register(pd.api.extensions.ExtensionArray)  # noqa: F811
@_select_index.register(pd.Index)  # noqa: F811
@_select_index.register(pd.MultiIndex)  # noqa: F811
@_select_index.register(pd.Series)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies to pd.Series/pd.Index/pd.array/np.ndarray.

    Returns an array of integers.
    """
    index = getattr(df, axis)

    if is_bool_dtype(arg):
        if len(arg) != len(index):
            raise IndexError(
                f"{arg} is a boolean dtype and has wrong length: "
                f"{len(arg)} instead of {len(index)}"
            )
        return np.asanyarray(arg)
    try:
        if isinstance(arg, pd.Series):
            arr = arg.array
        else:
            arr = arg
        if isinstance(index, pd.MultiIndex) and not isinstance(
            arg, pd.MultiIndex
        ):
            return index.get_locs([arg])
        arr = index.get_indexer_for(arr)
        not_found = arr == -1
        if not_found.all():
            raise KeyError(
                f"No match was returned for any of the labels in {arg}"
            )
        elif not_found.any():
            not_found = set(arg).difference(index)
            raise KeyError(
                f"No match was returned for these labels in {arg} - "
                f"{*not_found,}"
            )
        return arr
    except Exception as exc:
        raise KeyError(f"No match was returned for {arg}") from exc


@_select_index.register(DropLabel)  # noqa: F811
def _column_sel_dispatch(cols, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Returns the inverse of the passed label(s).

    Returns an array of integers.
    """
    arr = _select_index(cols.label, df, axis)
    index = np.arange(getattr(df, axis).size)
    arr = _index_converter(arr, index)
    return np.delete(index, arr)


@_select_index.register(set)
@_select_index.register(list)  # noqa: F811
def _index_dispatch(arg, df, axis):  # noqa: F811
    """
    Base function for selection on a Pandas Index object.
    Applies only to list type.
    It can take any of slice, str, callable, re.Pattern types, ...,
    or a combination of these types.

    Returns an array of integers.
    """
    index = getattr(df, axis)
    if is_bool_indexer(arg):
        if len(arg) != len(index):
            raise ValueError(
                "The length of the list of booleans "
                f"({len(arg)}) does not match "
                f"the length of the DataFrame's {axis}({index.size})."
            )

        return arg

    # shortcut for single unique dtype of scalars
    checks = (is_scalar(entry) for entry in arg)
    if all(checks):
        dtypes = {type(entry) for entry in arg}
        if len(dtypes) == 1:
            indices = index.get_indexer_for(list(arg))
            if (indices != -1).all():
                return indices
    # treat multiple DropLabel instances as a single unit
    checks = (isinstance(entry, DropLabel) for entry in arg)
    if sum(checks) > 1:
        drop_labels = (entry for entry in arg if isinstance(entry, DropLabel))
        drop_labels = [entry.label for entry in drop_labels]
        drop_labels = DropLabel(drop_labels)
        arg = [entry for entry in arg if not isinstance(entry, DropLabel)]
        arg.append(drop_labels)
    indices = [_select_index(entry, df, axis) for entry in arg]
    # single entry does not need to be combined
    # or materialized if possible;
    # this offers more performance
    if len(indices) == 1:
        if is_scalar(indices[0]):
            return indices
        indices = indices[0]
        if is_list_like(indices):
            indices = np.asanyarray(indices)
        return indices
    indices = [_index_converter(arr, index) for arr in indices]
    return np.concatenate(indices)


def _index_converter(arr, index):
    """Converts output from _select_index to an array_like"""
    if is_list_like(arr):
        arr = np.asanyarray(arr)
    if is_bool_dtype(arr):
        arr = arr.nonzero()[0]
    elif isinstance(arr, slice):
        arr = np.arange(index.size)[arr]
    elif isinstance(arr, int):
        arr = np.array([arr])
    return arr


def _select(
    df: pd.DataFrame,
    invert: bool = False,
    rows=None,
    columns=None,
) -> pd.DataFrame:
    """
    Index DataFrame on the index or columns.

    Returns a DataFrame.
    """
    if rows is None:
        row_indexer = slice(None)
    else:
        outcome = _select_index([rows], df, axis="index")
        if invert:
            row_indexer = np.ones(df.index.size, dtype=np.bool_)
            row_indexer[outcome] = False
        else:
            row_indexer = outcome
    if columns is None:
        column_indexer = slice(None)
    else:
        outcome = _select_index([columns], df, axis="columns")
        if invert:
            column_indexer = np.ones(df.columns.size, dtype=np.bool_)
            column_indexer[outcome] = False
        else:
            column_indexer = outcome
    return df.iloc[row_indexer, column_indexer]
