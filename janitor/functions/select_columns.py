import pandas_flavor as pf
import pandas as pd

from janitor.utils import deprecated_alias

from janitor.functions.utils import _select_column_names
from pandas.api.types import is_list_like


@pf.register_dataframe_method
@deprecated_alias(search_cols="search_column_names")
def select_columns(
    df: pd.DataFrame,
    *args,
    invert: bool = False,
) -> pd.DataFrame:
    """
    Method-chainable selection of columns.

    Not applicable to MultiIndex columns.

    It accepts a string, shell-like glob strings `(*string*)`,
    regex, slice, array-like object, or a list of the previous options.

    This method does not mutate the original DataFrame.

    Optional ability to invert selection of columns available as well.

    ```python
        import pandas as pd
        import janitor
        import numpy as np
        import datetime
        import re
        from janitor import patterns
        from pandas.api.types import is_datetime64_dtype

        df = pd.DataFrame(
                {
                    "id": [0, 1],
                    "Name": ["ABC", "XYZ"],
                    "code": [1, 2],
                    "code1": [4, np.nan],
                    "code2": ["8", 5],
                    "type": ["S", "R"],
                    "type1": ["E", np.nan],
                    "type2": ["T", "U"],
                    "code3": pd.Series(["a", "b"], dtype="category"),
                    "type3": pd.to_datetime([np.datetime64("2018-01-01"),
                                            datetime.datetime(2018, 1, 1)]),
                }
            )

        df

           id Name  code  code1 code2 type type1 type2 code3    type3
        0   0  ABC     1    4.0     8    S     E     T     a 2018-01-01
        1   1  XYZ     2    NaN     5    R   NaN     U     b 2018-01-01
    ```

    - Select by string:

    ```
        df.select_columns("id")
           id
       0   0
       1   1
    ```

    - Select via shell-like glob strings (`*`) is possible:

    ```python
        df.select_columns("type*")

           type type1 type2      type3
        0    S     E     T 2018-01-01
        1    R   NaN     U 2018-01-01
    ```

    - Select by slice:

    ```python
        df.select_columns(slice("code1", "type1"))

           code1 code2 type type1
        0    4.0     8    S     E
        1    NaN     5    R   NaN
    ```

    - Select by `Callable` (the callable is applied to every column
      and should return a single `True` or `False` per column):

    ```python
        df.select_columns(is_datetime64_dtype)

               type3
        0 2018-01-01
        1 2018-01-01

        df.select_columns(lambda x: x.name.startswith("code") or
                                    x.name.endswith("1"))

           code  code1 code2 type1 code3
        0     1    4.0     8     E     a
        1     2    NaN     5   NaN     b

        df.select_columns(lambda x: x.isna().any())

             code1 type1
        0    4.0     E
        1    NaN   NaN
    ```

    - Select by regular expression:

    ```python
        df.select_columns(re.compile("\\d+"))

           code1 code2 type1 type2 code3      type3
        0    4.0     8     E     T     a 2018-01-01
        1    NaN     5   NaN     U     b 2018-01-01

        # same as above, with janitor.patterns
        # simply a wrapper around re.compile

        df.select_columns(patterns("\\d+"))

           code1 code2 type1 type2 code3      type3
        0    4.0     8     E     T     a 2018-01-01
        1    NaN     5   NaN     U     b 2018-01-01
    ```

    - Select a combination of the above
      (you can combine any of the previous options):

    ```python
        df.select_columns("id", "code*", slice("code", "code2"))

           id  code  code1 code2 code3
        0   0     1    4.0     8     a
        1   1     2    NaN     5     b
    ```

    - You can also pass a sequence of booleans:

    ```python
        df.select_columns([True, False, True, True, True,
                           False, False, False, True, False])

           id  code  code1 code2 code3
        0   0     1    4.0     8     a
        1   1     2    NaN     5     b
    ```

    - Setting `invert` to `True`
      returns the complement of the columns provided:

    ```python
        df.select_columns("id", "code*", slice("code", "code2"),
                          invert = True)

           Name type type1 type2      type3
        0  ABC    S     E     T 2018-01-01
        1  XYZ    R   NaN     U 2018-01-01
    ```

    Functional usage example:

    ```python
       import pandas as pd
       import janitor as jn

       df = pd.DataFrame(...)

       df = jn.select_columns('a', 'b', 'col_*',
                              invert=True)
    ```

    Method-chaining example:

    ```python
        df = (pd.DataFrame(...)
              .select_columns('a', 'b', 'col_*',
              invert=True))
    ```

    :param df: A pandas DataFrame.
    :param args: Valid inputs include:
        - an exact column name to look for
        - a shell-style glob string (e.g., `*_thing_*`)
        - a regular expression
        - a callable which is applicable to each Series in the dataframe
        - variable arguments of all the aforementioned.
        - a sequence of booleans.
    :param invert: Whether or not to invert the selection.
        This will result in the selection of the complement of the columns
        provided.
    :returns: A pandas DataFrame with the specified columns selected.
    """

    # applicable for any
    # list-like object (ndarray, Series, pd.Index, ...)
    # excluding tuples, which are returned as is
    search_column_names = []
    for arg in args:
        if is_list_like(arg) and (not isinstance(arg, tuple)):
            search_column_names.extend([*arg])
        else:
            search_column_names.append(arg)
    if len(search_column_names) == 1:
        search_column_names = search_column_names[0]

    full_column_list = _select_column_names(search_column_names, df)

    if invert:
        return df.drop(columns=full_column_list)
    return df.loc[:, full_column_list]
