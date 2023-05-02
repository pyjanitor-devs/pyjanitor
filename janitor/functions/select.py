from typing import Any
import pandas_flavor as pf
import pandas as pd
from janitor.utils import deprecated_alias
from janitor.functions.utils import _select, DropLabel  # noqa: F401


@pf.register_dataframe_method
@deprecated_alias(search_cols="search_column_names")
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
            A dictionary can be used for selection on a MultiIndex on different levels.
        invert: Whether or not to invert the selection.
            This will result in the selection of the complement of the columns
            provided.

    Returns:
        A pandas DataFrame with the specified columns selected.
    """  # noqa: E501

    return _select(df, args=args, invert=invert, axis="columns")


@pf.register_dataframe_method
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
            A dictionary can be used for selection on a MultiIndex on different levels.
        invert: Whether or not to invert the selection.
            This will result in the selection of the complement of the rows
            provided.

    Returns:
        A pandas DataFrame with the specified rows selected.
    """  # noqa: E501
    return _select(df, args=args, invert=invert, axis="index")


@pf.register_dataframe_method
def select(
    df: pd.DataFrame, *, rows: Any = None, columns: Any = None
) -> pd.DataFrame:
    """Method-chainable selection of rows and columns.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    Selection on a MultiIndex on a level, or multiple levels,
    is possible with a dictionary.

    This method does not mutate the original DataFrame.

    Selection can be inverted with the `DropLabel` class.


    !!! info "New in version 0.24.0"


    !!!note

        The preferred option when selecting columns or rows in a Pandas DataFrame
        is with `.loc` or `.iloc` methods, as they are generally performant.
        `select` is primarily for convenience.

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
        >>> df.select(rows='cobra', columns='shield')
               shield
        cobra       2

        Labels can be dropped with the `DropLabel` class:

        >>> df.select(rows=DropLabel('cobra'))
                    max_speed  shield
        viper               4       5
        sidewinder          7       8

    More examples can be found in the
    [`select_columns`][janitor.functions.select.select_columns] section.

    Args:
        df: A pandas DataFrame.
        rows: Valid inputs include: an exact label to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection on a MultiIndex on different levels.
        columns: Valid inputs include: an exact label to look for,
            a shell-style glob string (e.g. `*_thing_*`),
            a regular expression,
            a callable,
            or variable arguments of all the aforementioned.
            A sequence of booleans is also acceptable.
            A dictionary can be used for selection on a MultiIndex on different levels.

    Returns:
        A pandas DataFrame with the specified rows and/or columns selected.
    """  # noqa: E501

    return _select(df, args=None, rows=rows, columns=columns, axis="both")
