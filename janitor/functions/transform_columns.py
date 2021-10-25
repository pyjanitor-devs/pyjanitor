from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union
import pandas_flavor as pf
import pandas as pd

from janitor.utils import check, deprecated_alias


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", dest_col_name="dest_column_name")
def transform_column(
    df: pd.DataFrame,
    column_name: Hashable,
    function: Callable,
    dest_column_name: Optional[str] = None,
    elementwise: bool = True,
) -> pd.DataFrame:
    """Transform the given column in-place using the provided function.

    Functions can be applied one of two ways:

    - Element-wise (default; `elementwise=True``)
    - Column-wise  (alternative; `elementwise=False``)

    If the function is applied "elementwise",
    then the first argument of the function signature
    should be the individual element of each function.
    This is the default behaviour of `transform_column``,
    because it is easy to understand.
    For example:



        def elemwise_func(x):
            modified_x = ... # do stuff here
            return modified_x

        df.transform_column(column_name="my_column", function=elementwise_func)

    On the other hand, columnwise application of a function
    behaves as if the function takes in a pandas Series
    and emits back a sequence that is of identical length to the original.
    One place where this is desirable
    is to gain access to `pandas` native string methods,
    which are super fast!



        def columnwise_func(s: pd.Series) -> pd.Series:
            return s.str[0:5]

        df.transform_column(
            column_name="my_column",
            lambda s: s.str[0:5],
            elementwise=False
        )

    This method does not mutate the original DataFrame.

    Let's say we wanted to apply a log10 transform a column of data.

    Originally one would write code like this:



        # YOU NO LONGER NEED TO WRITE THIS!
        df[column_name] = df[column_name].apply(np.log10)

    With the method chaining syntax, we can do the following instead:



        df = (
            pd.DataFrame(...)
            .transform_column(column_name, np.log10)
        )

    With the functional syntax:



        df = pd.DataFrame(...)
        df = transform_column(df, column_name, np.log10)

    :param df: A pandas DataFrame.
    :param column_name: The column to transform.
    :param function: A function to apply on the column.
    :param dest_column_name: The column name to store the transformation result
        in. Defaults to None, which will result in the original column
        name being overwritten. If a name is provided here, then a new column
        with the transformed values will be created.
    :param elementwise: Whether to apply the function elementwise or not.
        If elementwise is True, then the function's first argument
        should be the data type of each datum in the column of data,
        and should return a transformed datum.
        If elementwise is False, then the function's should expect
        a pandas Series passed into it, and return a pandas Series.

    :returns: A pandas DataFrame with a transformed column.
    """
    if dest_column_name is None:
        dest_column_name = column_name

    if elementwise:
        result = df[column_name].apply(function)
    else:
        result = function(df[column_name])

    df = df.assign(**{dest_column_name: result})
    return df


@pf.register_dataframe_method
@deprecated_alias(columns="column_names", new_names="new_column_names")
def transform_columns(
    df: pd.DataFrame,
    column_names: Union[List[str], Tuple[str]],
    function: Callable,
    suffix: Optional[str] = None,
    elementwise: bool = True,
    new_column_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Transform multiple columns through the same transformation.

    This method mutates the original DataFrame.

    Super syntactic sugar!

    Basically wraps `transform_column` and calls it repeatedly over all column
    names provided.

    User can optionally supply either a suffix to create a new set of columns
    with the specified suffix, or provide a dictionary mapping each original
    column name to its corresponding new column name. Note that all column
    names must be strings.

    A few examples below. Firstly, to just log10 transform a list of columns
    without creating new columns to hold the transformed values:



        df = (
            pd.DataFrame(...)
            .transform_columns(['col1', 'col2', 'col3'], np.log10)
        )

    Secondly, to add a '_log' suffix when creating a new column, which we think
    is going to be the most common use case:



        df = (
            pd.DataFrame(...)
            .transform_columns(
                ['col1', 'col2', 'col3'],
                np.log10,
                suffix="_log"
            )
        )

    Finally, to provide new names explicitly:



        df = (
            pd.DataFrame(...)
            .transform_column(
                ['col1', 'col2', 'col3'],
                np.log10,
                new_column_names={
                    'col1': 'transform1',
                    'col2': 'transform2',
                    'col3': 'transform3',
                    }
                )
        )

    :param df: A pandas DataFrame.
    :param column_names: An iterable of columns to transform.
    :param function: A function to apply on each column.
    :param suffix: (optional) Suffix to use when creating new columns to hold
        the transformed values.
    :param elementwise: Passed on to `transform_column`; whether or not
        to apply the transformation function elementwise (True)
        or columnwise (False).
    :param new_column_names: (optional) An explicit mapping of old column names
        to new column names.
    :returns: A pandas DataFrame with transformed columns.
    :raises ValueError: if both `suffix` and `new_column_names` are
        specified
    """
    dest_column_names = dict(zip(column_names, column_names))

    check("column_names", column_names, [list, tuple])

    if suffix is not None and new_column_names is not None:
        raise ValueError(
            "only one of suffix or new_column_names should be specified"
        )

    if suffix:  # If suffix is specified...
        check("suffix", suffix, [str])
        for col in column_names:
            dest_column_names[col] = col + suffix

    if new_column_names:  # If new_column_names is specified...
        check("new_column_names", new_column_names, [dict])
        dest_column_names = new_column_names

    # Now, transform columns.
    for old_col, new_col in dest_column_names.items():
        df = transform_column(
            df, old_col, function, new_col, elementwise=elementwise
        )

    return df
