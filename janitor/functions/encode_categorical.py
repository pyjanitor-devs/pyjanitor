import warnings
from enum import Enum
from typing import Hashable, Iterable, Union

import pandas_flavor as pf
import pandas as pd
from pandas.api.types import is_list_like

from janitor.utils import check, check_column, deprecated_alias


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

    Simply pass a string, or a sequence of column names to `column_names`;
    alternatively, you can pass kwargs, where the keys are the column names
    and the values can either be None, `sort`, `appearance`
    or a 1-D array-like object.

    - None: column is cast to an unordered categorical.
    - `sort`: column is cast to an ordered categorical,
              with the order defined by the sort-order of the categories.
    - `appearance`: column is cast to an ordered categorical,
                    with the order defined by the order of appearance
                    in the original column.
    - 1d-array-like object: column is cast to an ordered categorical,
                            with the categories and order as specified
                            in the input array.

    `column_names` and `kwargs` parameters cannot be used at the same time.

    Example: Using `column_names`

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "foo": ["b", "b", "a", "c", "b"],
        ...     "bar": range(4, 9),
        ... })
        >>> df
          foo  bar
        0   b    4
        1   b    5
        2   a    6
        3   c    7
        4   b    8
        >>> df.dtypes
        foo    object
        bar     int64
        dtype: object
        >>> enc_df = df.encode_categorical(column_names="foo")
        >>> enc_df.dtypes
        foo    category
        bar       int64
        dtype: object
        >>> enc_df["foo"].cat.categories
        Index(['a', 'b', 'c'], dtype='object')
        >>> enc_df["foo"].cat.ordered
        False

    Example: Using `kwargs` to specify an ordered categorical.

        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "foo": ["b", "b", "a", "c", "b"],
        ...     "bar": range(4, 9),
        ... })
        >>> df.dtypes
        foo    object
        bar     int64
        dtype: object
        >>> enc_df = df.encode_categorical(foo="appearance")
        >>> enc_df.dtypes
        foo    category
        bar       int64
        dtype: object
        >>> enc_df["foo"].cat.categories
        Index(['b', 'a', 'c'], dtype='object')
        >>> enc_df["foo"].cat.ordered
        True

    :param df: A pandas DataFrame object.
    :param column_names: A column name or an iterable (list or tuple)
        of column names.
    :param **kwargs: A mapping from column name to either `None`,
        `'sort'` or `'appearance'`, or a 1-D array. This is useful
        in creating categorical columns that are ordered, or
        if the user needs to explicitly specify the categories.
    :returns: A pandas DataFrame.
    :raises ValueError: If both `column_names` and `kwargs` are provided.
    """  # noqa: E501

    if all((column_names, kwargs)):
        raise ValueError(
            "Only one of `column_names` or `kwargs` can be provided."
        )
    # column_names deal with only category dtype (unordered)
    # kwargs takes care of scenarios where user wants an ordered category
    # or user supplies specific categories to create the categorical
    if column_names is not None:
        check("column_names", column_names, [list, tuple, Hashable])
        if isinstance(column_names, Hashable):
            column_names = [column_names]
        check_column(df, column_names)
        dtypes = {col: "category" for col in column_names}
        return df.astype(dtypes)

    return _computations_as_categorical(df, **kwargs)


def _computations_as_categorical(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    This function handles cases where
    categorical columns are created with an order,
    or specific values supplied for the categories.
    It uses a kwarg, where the key is the column name,
    and the value is either a string, or a 1D array.
    The default for value is None and will return a categorical dtype
    with no order and categories inferred from the column.
    A DataFrame, with categorical columns, is returned.
    """

    categories_dict = _as_categorical_checks(df, **kwargs)

    categories_dtypes = {}

    for column_name, value in categories_dict.items():
        if value is None:
            cat_dtype = pd.CategoricalDtype()
        elif isinstance(value, str):
            if value == _CategoryOrder.SORT.value:
                _, cat_dtype = df[column_name].factorize(sort=True)
            else:
                _, cat_dtype = df[column_name].factorize(sort=False)
            if cat_dtype.empty:
                raise ValueError(
                    "Kindly ensure there is at least "
                    f"one non-null value in {column_name}."
                )
            cat_dtype = pd.CategoricalDtype(categories=cat_dtype, ordered=True)

        else:  # 1-D array
            cat_dtype = pd.CategoricalDtype(categories=value, ordered=True)

        categories_dtypes[column_name] = cat_dtype

    return df.astype(categories_dtypes)


def _as_categorical_checks(df: pd.DataFrame, **kwargs) -> dict:
    """
    This function raises errors if columns in `kwargs` are
    absent from the dataframe's columns.
    It also raises errors if the value in `kwargs`
    is not a string (`'appearance'` or `'sort'`), or a 1D array.

    This function is executed before proceeding to the computation phase.

    If all checks pass, a dictionary of column names and value is returned.

    :param df: The pandas DataFrame object.
    :param **kwargs: A pairing of column name and value.
    :returns: A dictionary.
    :raises TypeError: If `value` is not a 1-D array, or a string.
    :raises ValueError: If `value` is a 1-D array, and contains nulls,
        or is non-unique.
    """

    check_column(df, kwargs)

    categories_dict = {}

    for column_name, value in kwargs.items():
        # type check
        if (value is not None) and not (
            is_list_like(value) or isinstance(value, str)
        ):
            raise TypeError(f"{value} should be list-like or a string.")
        if is_list_like(value):
            if not hasattr(value, "shape"):
                value = pd.Index([*value])

            arr_ndim = value.ndim
            if (arr_ndim != 1) or isinstance(value, pd.MultiIndex):
                raise ValueError(
                    f"{value} is not a 1-D array. "
                    "Kindly provide a 1-D array-like object."
                )

            if not isinstance(value, (pd.Series, pd.Index)):
                value = pd.Index(value)

            if value.hasnans:
                raise ValueError(
                    "Kindly ensure there are no nulls in the array provided."
                )

            if not value.is_unique:
                raise ValueError(
                    "Kindly provide unique, "
                    "non-null values for the array provided."
                )

            if value.empty:
                raise ValueError(
                    "Kindly ensure there is at least "
                    "one non-null value in the array provided."
                )

            # uniques, without nulls
            uniques = df[column_name].factorize(sort=False)[-1]
            if uniques.empty:
                raise ValueError(
                    "Kindly ensure there is at least "
                    f"one non-null value in {column_name}."
                )

            missing = uniques.difference(value, sort=False)
            if not missing.empty and (uniques.size > missing.size):
                warnings.warn(
                    f"Values {tuple(missing)} are missing from "
                    f"the provided categories {value} "
                    f"for {column_name}; this may create nulls "
                    "in the new categorical column.",
                    UserWarning,
                    stacklevel=2,
                )

            elif uniques.equals(missing):
                warnings.warn(
                    f"None of the values in {column_name} are in "
                    f"{value}; this might create nulls for all values "
                    f"in the new categorical column.",
                    UserWarning,
                    stacklevel=2,
                )

        elif isinstance(value, str):
            category_order_types = {ent.value for ent in _CategoryOrder}
            if value.lower() not in category_order_types:
                raise ValueError(
                    "Argument should be one of 'appearance' or 'sort'."
                )

        categories_dict[column_name] = value

    return categories_dict


class _CategoryOrder(Enum):
    """
    order types for encode_categorical.
    """

    SORT = "sort"
    APPEARANCE = "appearance"
