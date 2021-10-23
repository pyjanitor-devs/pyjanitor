from typing import Hashable, Iterable, Union
import pandas_flavor as pf
import pandas as pd
from pandas.api.types import is_list_like
import warnings
from janitor.utils import check, check_column, deprecated_alias
from enum import Enum


@pf.register_dataframe_method
@deprecated_alias(columns="column_names")
def encode_categorical(
    df: pd.DataFrame,
    column_names: Union[str, Iterable[str], Hashable] = None,
    **kwargs,
) -> pd.DataFrame:
    """Encode the specified columns with Pandas' [category dtype][cat].

    [cat]: http://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html

    It is syntactic sugar around `pd.Categorical`.

    This method does not mutate the original DataFrame.

    Note: In versions < 0.20.11, this method mutates the original DataFrame.

    TODO: The big chunk of examples below
    should be moved into a Jupyter notebook.
    This will keep the docstring consistent and to-the-point.

    Examples:

    ```python
        col1	col2	col3
    0	2.0	a	2020-01-01
    1	1.0	b	2020-01-02
    2	3.0	c	2020-01-03
    3	1.0	d	2020-01-04
    4	NaN	a	2020-01-05

    df.dtypes

    col1           float64
    col2            object
    col3    datetime64[ns]
    dtype: object
    ```

    Specific columns can be converted to category type:

    ```python
    df = (
        pd.DataFrame(...)
        .encode_categorical(
            column_names=['col1', 'col2', 'col3']
        )
    )

    df.dtypes

    col1    category
    col2    category
    col3    category
    dtype: object
    ```

    Note that for the code above, the categories were inferred from
    the columns, and is unordered:

        df['col3']
        0   2020-01-01
        1   2020-01-02
        2   2020-01-03
        3   2020-01-04
        4   2020-01-05
        Name: col3, dtype: category
        Categories (5, datetime64[ns]):
        [2020-01-01, 2020-01-02, 2020-01-03, 2020-01-04, 2020-01-05]


    Explicit categories can be provided, and ordered via the `kwargs``
    parameter:

        df = (pd.DataFrame(...)
                .encode_categorical(
                    col1 = ([3, 2, 1, 4], "appearance"),
                    col2 = (['a','d','c','b'], "sort")
                    )
            )

        df['col1']
        0      2
        1      1
        2      3
        3      1
        4    NaN
        Name: col1, dtype: category
        Categories (4, int64): [3 < 2 < 1 < 4]

        df['col2']
        0    a
        1    b
        2    c
        3    d
        4    a
        Name: col2, dtype: category
        Categories (4, object): [a < b < c < d]

    When the `order` parameter is "appearance",
    the categories argument is used as-is;
    if the `order` is "sort",
    the categories argument is sorted in ascending order;
    if `order` is `None``,
    then the categories argument is applied unordered.


    A User Warning will be generated if some or all of the unique values
    in the column are not present in the provided `categories` argument.

    ```python
        df = (pd.DataFrame(...)
                .encode_categorical(
                    col1 = (
                            categories = [4, 5, 6],
                            order = "appearance"
                            )
            )

        UserWarning: None of the values in col1 are in [4, 5, 6];
                     this might create nulls for all your values
                     in the new categorical column.

        df['col1']
        0    NaN
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        Name: col1, dtype: category
        Categories (3, int64): [4 < 5 < 6]
    ```

    .. note:: if `categories` is None in the `kwargs` tuple, then the
        values for `categories` are inferred from the column; if `order`
        is None, then the values for categories are applied unordered.

    .. note:: `column_names` and `kwargs` parameters cannot be used at
        the same time.

    Functional usage syntax:

    ```python

        import pandas as pd
        import janitor as jn

    - With `column_names``::

        categorical_cols = ['col1', 'col2', 'col4']
        df = jn.encode_categorical(
                    df,
                    columns = categorical_cols)  # one way

    - With `kwargs``::

        df = jn.encode_categorical(
                    df,
                    col1 = (categories, order),
                    col2 = (categories = [values],
                    order="sort"  # or "appearance" or None

                )

    Method chaining syntax:

    - With `column_names``::

        categorical_cols = ['col1', 'col2', 'col4']
        df = (pd.DataFrame(...)
                .encode_categorical(columns=categorical_cols)
            )

    - With `kwargs``::

        df = (
            pd.DataFrame(...)
            .encode_categorical(
                col1 = (categories, order),
                col2 = (categories = [values]/None,
                        order="sort"  # or "appearance" or None
                        )
        )

    :param df: The pandas DataFrame object.
    :param column_names: A column name or an iterable (list or
        tuple) of column names.
    :param kwargs: A pairing of column name to a tuple of (`categories`, `order`).
        This is useful in creating categorical columns that are ordered, or
        if the user needs to explicitly specify the categories.
    :returns: A pandas DataFrame.
    :raises ValueError: if both ``column_names`` and ``kwargs`` are provided.
    """  # noqa: E501

    if all((column_names, kwargs)):
        raise ValueError(
            """
            Only one of `column_names` or `kwargs`
            can be provided.
            """
        )
    # column_names deal with only category dtype (unordered)
    # kwargs takes care of scenarios where user wants an ordered category
    # or user supplies specific categories to create the categorical
    if column_names is not None:
        check("column_names", column_names, [list, tuple, Hashable])
        if isinstance(column_names, (list, tuple)):
            check_column(df, column_names)
            dtypes = {col: "category" for col in column_names}
            return df.astype(dtypes)
        if isinstance(column_names, Hashable):
            check_column(df, [column_names])
            return df.astype({column_names: "category"})

    return _computations_as_categorical(df, **kwargs)


def _computations_as_categorical(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    This function handles cases where
    categorical columns are created with an order,
    or specific values supplied for the categories.
    It uses a kwarg, where the key is the column name,
    and the value is a tuple of categories, order.
    The defaults for the tuple are (None, None)
    and will return a categorical dtype
    with no order and categories inferred from the column.
    A DataFrame, with catetorical columns, is returned.
    """

    categories_dict = as_categorical_checks(df, **kwargs)

    categories_dtypes = {}

    for column_name, (
        cat,
        order,
    ) in categories_dict.items():
        error_msg = f"""
                     Kindly ensure there is at least
                     one non-null value in {column_name}.
                     """
        if (cat is None) and (order is None):
            cat_dtype = pd.CategoricalDtype()

        elif (cat is None) and (order is _CategoryOrder.SORT.value):
            cat = df[column_name].factorize(sort=True)[-1]
            if cat.empty:
                raise ValueError(error_msg)
            cat_dtype = pd.CategoricalDtype(categories=cat, ordered=True)

        elif (cat is None) and (order is _CategoryOrder.APPEARANCE.value):
            cat = df[column_name].factorize(sort=False)[-1]
            if cat.empty:
                raise ValueError(error_msg)
            cat_dtype = pd.CategoricalDtype(categories=cat, ordered=True)

        elif cat is not None:  # order is irrelevant if cat is provided
            cat_dtype = pd.CategoricalDtype(categories=cat, ordered=True)

        categories_dtypes[column_name] = cat_dtype

    return df.astype(categories_dtypes)


def as_categorical_checks(df: pd.DataFrame, **kwargs) -> dict:
    """
    This function raises errors if columns in `kwargs` are
    absent in the the dataframe's columns.
    It also raises errors if the tuple in `kwargs`
    has a length greater than 2, or the `order` value,
    if not None, is not one of `appearance` or `sort`.
    Error is raised if the `categories` in the tuple in `kwargs`
    is not a 1-D array-like object.

    This function is executed before proceeding to the computation phase.

    If all checks pass, a dictionary of column names and tuple
    of (categories, order) is returned.

    :param df: The pandas DataFrame object.
    :param kwargs: A pairing of column name
        to a tuple of (`categories`, `order`).
    :returns: A dictionary.
    :raises TypeError: if the value in ``kwargs`` is not a tuple.
    :raises ValueError: if ``categories`` is not a 1-D array.
    :raises ValueError: if ``order`` is not one of
        `sort`, `appearance`, or `None`.
    """

    # column checks
    check_column(df, kwargs)

    categories_dict = {}

    for column_name, value in kwargs.items():
        # type check
        check("Pair of `categories` and `order`", value, [tuple])

        len_value = len(value)

        if len_value != 2:
            raise ValueError(
                f"""
                The tuple of (categories, order) for {column_name}
                should be length 2; the tuple provided is
                length {len_value}.
                """
            )

        cat, order = value
        if cat is not None:
            if not is_list_like(cat):
                raise TypeError(f"{cat} should be list-like.")

            if not hasattr(cat, "shape"):
                checker = pd.Index([*cat])
            else:
                checker = cat

            arr_ndim = checker.ndim
            if (arr_ndim != 1) or isinstance(checker, pd.MultiIndex):
                raise ValueError(
                    f"""
                    {cat} is not a 1-D array.
                    Kindly provide a 1-D array-like object to `categories`.
                    """
                )

            if not isinstance(checker, (pd.Series, pd.Index)):
                checker = pd.Index(checker)

            if checker.hasnans:
                raise ValueError(
                    "Kindly ensure there are no nulls in `categories`."
                )

            if not checker.is_unique:
                raise ValueError(
                    """
                    Kindly provide unique,
                    non-null values for `categories`.
                    """
                )

            if checker.empty:
                raise ValueError(
                    """
                    Kindly ensure there is at least
                    one non-null value in `categories`.
                    """
                )

            # uniques, without nulls
            uniques = df[column_name].factorize(sort=False)[-1]
            if uniques.empty:
                raise ValueError(
                    f"""
                     Kindly ensure there is at least
                     one non-null value in {column_name}.
                     """
                )

            missing = uniques.difference(checker, sort=False)
            if not missing.empty and (uniques.size > missing.size):
                warnings.warn(
                    f"""
                     Values {tuple(missing)} are missing from
                     the provided categories {cat}
                     for {column_name}; this may create nulls
                     in the new categorical column.
                     """,
                    UserWarning,
                    stacklevel=2,
                )

            elif uniques.equals(missing):
                warnings.warn(
                    f"""
                     None of the values in {column_name} are in
                     {cat};
                     this might create nulls for all values
                     in the new categorical column.
                     """,
                    UserWarning,
                    stacklevel=2,
                )

        if order is not None:
            check("order", order, [str])

            category_order_types = [ent.value for ent in _CategoryOrder]
            if order.lower() not in category_order_types:
                raise ValueError(
                    """
                    `order` argument should be one of
                    "appearance", "sort" or `None`.
                    """
                )

        categories_dict[column_name] = value

    return categories_dict


class _CategoryOrder(Enum):
    """
    order types for encode_categorical.
    """

    SORT = "sort"
    APPEARANCE = "appearance"
