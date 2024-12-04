"""Utility functions for all of the functions submodule."""

from __future__ import annotations

import re
import unicodedata
import warnings
from enum import Enum
from typing import (
    Any,
    Hashable,
    Iterable,
    List,
    Optional,
    Pattern,
    Union,
)

import numpy as np
import pandas as pd
from multipledispatch import dispatch
from pandas.api.types import (
    is_list_like,
    is_scalar,
    is_string_dtype,
    union_categoricals,
)

from janitor.errors import JanitorError
from janitor.utils import (
    _expand_grid,
    check,
    check_column,
    find_stack_level,
)

warnings.simplefilter("always", DeprecationWarning)


def unionize_dataframe_categories(
    *dataframes: Any,
    column_names: Optional[Iterable[pd.CategoricalDtype]] = None,
) -> List[pd.DataFrame]:
    """
    Given a group of dataframes which contain some categorical columns, for
    each categorical column present, find all the possible categories across
    all the dataframes which have that column.
    Update each dataframes' corresponding column with a new categorical object
    that contains the original data
    but has labels for all the possible categories from all dataframes.
    This is useful when concatenating a list of dataframes which all have the
    same categorical columns into one dataframe.

    If, for a given categorical column, all input dataframes do not have at
    least one instance of all the possible categories,
    Pandas will change the output dtype of that column from `category` to
    `object`, losing out on dramatic speed gains you get from the former
    format.

    Examples:
        Usage example for concatenation of categorical column-containing
        dataframes:

        Instead of:

        ```python
        concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)
        ```

        which in your case has resulted in `category` -> `object` conversion,
        use:

        ```python
        unionized_dataframes = unionize_dataframe_categories(df1, df2, df2)
        concatenated_df = pd.concat(unionized_dataframes, ignore_index=True)
        ```

    Args:
        *dataframes: The dataframes you wish to unionize the categorical
            objects for.
        column_names: If supplied, only unionize this subset of columns.

    Raises:
        TypeError: If any of the inputs are not pandas DataFrames.

    Returns:
        A list of the category-unioned dataframes in the same order they
            were provided.
    """

    if any(not isinstance(df, pd.DataFrame) for df in dataframes):
        raise TypeError("Inputs must all be dataframes.")

    if column_names is None:
        # Find all columns across all dataframes that are categorical

        column_names = set()

        for dataframe in dataframes:
            column_names = column_names.union(
                [
                    column_name
                    for column_name in dataframe.columns
                    if isinstance(
                        dataframe[column_name].dtype, pd.CategoricalDtype
                    )
                ]
            )

    else:
        column_names = [column_names]
    # For each categorical column, find all possible values across the DFs

    category_unions = {
        column_name: union_categoricals(
            [df[column_name] for df in dataframes if column_name in df.columns]
        )
        for column_name in column_names
    }

    # Make a shallow copy of all DFs and modify the categorical columns
    # such that they can encode the union of all possible categories for each.

    refactored_dfs = []

    for df in dataframes:
        df = df.copy(deep=False)

        for column_name, categorical in category_unions.items():
            if column_name in df.columns:
                df[column_name] = pd.Categorical(
                    df[column_name], categories=categorical.categories
                )

        refactored_dfs.append(df)

    return refactored_dfs


def patterns(regex_pattern: Union[str, Pattern]) -> Pattern:
    """This function converts a string into a compiled regular expression.

    It can be used to select columns in the index or columns_names
    arguments of `pivot_longer` function.

    !!!warning

        This function is deprecated. Kindly use `re.compile` instead.

    Args:
        regex_pattern: String to be converted to compiled regular
            expression.

    Returns:
        A compile regular expression from provided `regex_pattern`.
    """
    warnings.warn(
        "This function is deprecated. Kindly use `re.compile` instead.",
        DeprecationWarning,
        stacklevel=find_stack_level(),
    )
    check("regular expression", regex_pattern, [str, Pattern])

    return re.compile(regex_pattern)


def _computations_expand_grid(others: dict) -> dict:
    """
    Creates a cartesian product of all the inputs in `others`.
    Uses numpy's `mgrid` to generate indices, which is used to
    `explode` all the inputs in `others`.

    There is a performance penalty for small entries
    in using this method, instead of `itertools.product`;
    however, there are significant performance benefits
    as the size of the data increases.

    Another benefit of this approach, in addition to the significant
    performance gains, is the preservation of data types.
    This is particularly relevant for pandas' extension arrays `dtypes`
    (categoricals, nullable integers, ...).

    A dictionary of all possible combinations is returned.
    """

    for key in others:
        check("key", key, [Hashable])

    grid = {}

    for key, value in others.items():
        if is_scalar(value):
            value = np.asarray([value])
        elif is_list_like(value) and (not hasattr(value, "shape")):
            value = np.asarray([*value])
        if not value.size:
            raise ValueError(f"Kindly provide a non-empty array for {key}.")

        grid[key] = value

    others = None

    # slice obtained here is used in `np.mgrid`
    # to generate cartesian indices
    # which is then paired with grid.items()
    # to blow up each individual value
    # before creating the final DataFrame.
    grid = grid.items()
    grid_index = [slice(len(value)) for _, value in grid]
    grid_index = map(np.ravel, np.mgrid[grid_index])
    grid = zip(grid, grid_index)
    grid = ((*left, right) for left, right in grid)
    contents = {}
    for key, value, grid_index in grid:
        contents.update(_expand_grid(value, grid_index, key))
    # check length of keys and pad if necessary
    lengths = set(map(len, contents))
    if len(lengths) > 1:
        lengths = max(lengths)
        others = {}
        for key, value in contents.items():
            len_key = len(key)
            if len_key < lengths:
                padding = [""] * (lengths - len_key)
                key = (*key, *padding)
            others[key] = value
        return others
    return contents


@dispatch(pd.DataFrame, (list, tuple), str)
def _factorize(df, column_names, suffix, **kwargs):
    check_column(df, column_names=column_names, present=True)
    for col in column_names:
        df[f"{col}{suffix}"] = pd.factorize(df[col], **kwargs)[0]
    return df


@dispatch(pd.DataFrame, str, str)
def _factorize(df, column_name, suffix, **kwargs):  # noqa: F811
    check_column(df, column_names=column_name, present=True)
    df[f"{column_name}{suffix}"] = pd.factorize(df[column_name], **kwargs)[0]
    return df


class _JoinOperator(Enum):
    """
    List of operators used in conditional_join.
    """

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STRICTLY_EQUAL = "=="
    NOT_EQUAL = "!="


less_than_join_types = {
    _JoinOperator.LESS_THAN.value,
    _JoinOperator.LESS_THAN_OR_EQUAL.value,
}
greater_than_join_types = {
    _JoinOperator.GREATER_THAN.value,
    _JoinOperator.GREATER_THAN_OR_EQUAL.value,
}


def _null_checks_cond_join(
    left: pd.Series, right: pd.Series
) -> Union[tuple, None]:
    """
    Checks for nulls in the arrays before conducting binary search.

    Relevant to _less_than_indices and _greater_than_indices
    """
    any_nulls = left.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        left = left[~any_nulls]
    any_nulls = right.isna()
    if any_nulls.all():
        return None
    if any_nulls.any():
        right = right[~any_nulls]
    any_nulls = any_nulls.any()
    right_is_sorted = right.is_monotonic_increasing
    if not right_is_sorted:
        right = right.sort_values(kind="stable")

    left_index = left.index._values
    left = left._values
    right_index = right.index._values
    right = right._values

    return left, right, left_index, right_index, right_is_sorted, any_nulls


def _equal_indices(
    left: pd.Series, right: pd.Series, return_ragged_arrays: bool
) -> tuple:
    """
    Use binary search to get indices where left
    is equal to right.

    A tuple of integer indexes
    for left and right is returned.
    """
    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome
    # steal some perf here within the binary search
    # search for uniques
    # and later index them with left_positions
    # it is assumed that users will only reach for this
    # if the data is reasonably duplicated; if not
    # pd.merge is superb especially if it's a one-to-one
    # or one-to-many
    positions, left = pd.factorize(left, sort=False)
    if return_ragged_arrays:
        starts = right.searchsorted(left, side="left")
        starts = starts[positions]
        ends = right.searchsorted(left, side="right")
        ends = ends[positions]
        booleans = starts < ends
        if not booleans.any():
            return None
        if not booleans.all():
            left_index = left_index[booleans]
            starts = starts[booleans]
            ends = ends[booleans]
        right = [slice(start, end) for start, end in zip(starts, ends)]
        if right_is_sorted & (not any_nulls):
            return left_index, right
        right = [right_index[slicer] for slicer in right]
        return left_index, right
    # necessary step to remove non matches in right
    # vital to ensuring correct output in numba_equi_join
    # when building the regions
    booleans = pd.Index(left).get_indexer(right) != -1
    if not booleans.any():
        return None
    if not booleans.all():
        right_index = right_index[booleans]
        right = right[booleans]
    starts = right.searchsorted(left, side="left")
    starts = starts[positions]
    ends = right.searchsorted(left, side="right")
    ends = ends[positions]
    booleans = starts < ends
    if not booleans.any():
        return None
    if not booleans.all():
        left_index = left_index[booleans]
        starts = starts[booleans]
    return left_index, right_index, starts


def _less_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    multiple_conditions: bool,
    keep: str,
    return_ragged_arrays: bool,
) -> tuple:
    """
    Use binary search to get indices where left
    is less than or equal to right.

    If strict is True, then only indices
    where `left` is less than
    (but not equal to) `right` are returned.

    A tuple of integer indexes
    for left and right is returned.
    """

    # no point going through all the hassle
    if left.min() > right.max():
        return None

    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome

    search_indices = right.searchsorted(left, side="left")
    # if any of the positions in `search_indices`
    # is equal to the length of `right_keys`
    # that means the respective position in `left`
    # has no values from `right` that are less than
    # or equal, and should therefore be discarded
    len_right = right.size
    booleans = search_indices < len_right

    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]

    # the idea here is that if there are any equal values
    # shift to the right to the immediate next position
    # that is not equal
    if strict:
        booleans = left == right[search_indices]
        # replace positions where rows are equal
        # with positions from searchsorted('right')
        # positions from searchsorted('right') will never
        # be equal and will be the furthermost in terms of position
        # example : right -> [2, 2, 2, 3], and we need
        # positions where values are not equal for 2;
        # the furthermost will be 3, and searchsorted('right')
        # will return position 3.
        if booleans.any():
            replacements = right.searchsorted(left, side="right")
            # now we can safely replace values
            # with strictly less than positions
            search_indices = np.where(booleans, replacements, search_indices)
        # check again if any of the values
        # have become equal to length of right
        # and get rid of them
        booleans = search_indices < len_right

        if not booleans.all():
            left_index = left_index[booleans]
            search_indices = search_indices[booleans]

        if not search_indices.size:
            return None
    if multiple_conditions:
        return left_index, right_index, search_indices
    if right_is_sorted & (keep == "last"):
        indexer = np.empty_like(search_indices)
        indexer[:] = len_right - 1
        return left_index, right_index[indexer]
    if right_is_sorted & (keep == "first") & any_nulls:
        return left_index, right_index[search_indices]
    if right_is_sorted & (keep == "first"):
        return left_index, search_indices
    if return_ragged_arrays & right_is_sorted & (not any_nulls):
        right = [slice(ind, len_right) for ind in search_indices]
        return left_index, right
    right = [right_index[ind:len_right] for ind in search_indices]
    if return_ragged_arrays:
        return left_index, right
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(len_right - search_indices)
    return left, right


def _greater_than_indices(
    left: pd.Series,
    right: pd.Series,
    strict: bool,
    multiple_conditions: bool,
    keep: str,
    return_ragged_arrays: bool,
) -> tuple:
    """
    Use binary search to get indices where left
    is greater than or equal to right.

    If strict is True, then only indices
    where `left` is greater than
    (but not equal to) `right` are returned.

    if multiple_conditions is False, a tuple of integer indexes
    for left and right is returned;
    else a tuple of the index for left, right, as well
    as the positions of left in right is returned.
    """

    # quick break, avoiding the hassle
    if left.max() < right.min():
        return None

    outcome = _null_checks_cond_join(left=left, right=right)
    if not outcome:
        return None
    left, right, left_index, right_index, right_is_sorted, any_nulls = outcome
    search_indices = right.searchsorted(left, side="right")
    # if any of the positions in `search_indices`
    # is equal to 0 (less than 1), it implies that
    # left[position] is not greater than any value
    # in right
    booleans = search_indices > 0
    if not booleans.all():
        left = left[booleans]
        left_index = left_index[booleans]
        search_indices = search_indices[booleans]

    # the idea here is that if there are any equal values
    # shift downwards to the immediate next position
    # that is not equal
    if strict:
        booleans = left == right[search_indices - 1]
        # replace positions where rows are equal with
        # searchsorted('left');
        # this works fine since we will be using the value
        # as the right side of a slice, which is not included
        # in the final computed value
        if booleans.any():
            replacements = right.searchsorted(left, side="left")
            # now we can safely replace values
            # with strictly greater than positions
            search_indices = np.where(booleans, replacements, search_indices)
        # any value less than 1 should be discarded
        # since the lowest value for binary search
        # with side='right' should be 1
        booleans = search_indices > 0
        if not booleans.all():
            left_index = left_index[booleans]
            search_indices = search_indices[booleans]

        if not search_indices.size:
            return None
    if multiple_conditions:
        return left_index, right_index, search_indices
    if right_is_sorted & (keep == "first"):
        indexer = np.zeros_like(search_indices)
        return left_index, right_index[indexer]
    if right_is_sorted & (keep == "last") & any_nulls:
        return left_index, right_index[search_indices - 1]
    if right_is_sorted & (keep == "last"):
        return left_index, search_indices - 1
    if return_ragged_arrays & right_is_sorted & (not any_nulls):
        right = [slice(0, ind) for ind in search_indices]
        return left_index, right
    right = [right_index[:ind] for ind in search_indices]
    if return_ragged_arrays:
        return left_index, right
    if keep == "first":
        right = [arr.min() for arr in right]
        return left_index, right
    if keep == "last":
        right = [arr.max() for arr in right]
        return left_index, right
    right = np.concatenate(right)
    left = left_index.repeat(search_indices)
    return left, right


def _not_equal_indices(left: pd.Series, right: pd.Series, keep: str) -> tuple:
    """
    Use binary search to get indices where
    `left` is exactly  not equal to `right`.

    It is a combination of strictly less than
    and strictly greater than indices.

    A tuple of integer indexes for left and right
    is returned.
    """

    dummy = np.array([], dtype=int)

    # deal with nulls
    l1_nulls = dummy
    r1_nulls = dummy
    l2_nulls = dummy
    r2_nulls = dummy
    any_left_nulls = left.isna()
    any_right_nulls = right.isna()
    if any_left_nulls.any():
        l1_nulls = left.index[any_left_nulls.array]
        l1_nulls = l1_nulls.to_numpy(copy=False)
        r1_nulls = right.index
        # avoid NAN duplicates
        if any_right_nulls.any():
            r1_nulls = r1_nulls[~any_right_nulls.array]
        r1_nulls = r1_nulls.to_numpy(copy=False)
        nulls_count = l1_nulls.size
        # blow up nulls to match length of right
        l1_nulls = np.tile(l1_nulls, r1_nulls.size)
        # ensure length of right matches left
        if nulls_count > 1:
            r1_nulls = np.repeat(r1_nulls, nulls_count)
    if any_right_nulls.any():
        r2_nulls = right.index[any_right_nulls.array]
        r2_nulls = r2_nulls.to_numpy(copy=False)
        l2_nulls = left.index
        nulls_count = r2_nulls.size
        # blow up nulls to match length of left
        r2_nulls = np.tile(r2_nulls, l2_nulls.size)
        # ensure length of left matches right
        if nulls_count > 1:
            l2_nulls = np.repeat(l2_nulls, nulls_count)

    l1_nulls = np.concatenate([l1_nulls, l2_nulls])
    r1_nulls = np.concatenate([r1_nulls, r2_nulls])

    outcome = _less_than_indices(
        left,
        right,
        strict=True,
        multiple_conditions=False,
        keep=keep,
        return_ragged_arrays=False,
    )

    if outcome is None:
        lt_left = dummy
        lt_right = dummy
    else:
        lt_left, lt_right = outcome

    outcome = _greater_than_indices(
        left,
        right,
        strict=True,
        multiple_conditions=False,
        keep=keep,
        return_ragged_arrays=False,
    )

    if outcome is None:
        gt_left = dummy
        gt_right = dummy
    else:
        gt_left, gt_right = outcome

    left = np.concatenate([lt_left, gt_left, l1_nulls])
    right = np.concatenate([lt_right, gt_right, r1_nulls])

    if (not left.size) & (not right.size):
        return None
    return _keep_output(keep, left, right)


def _generic_func_cond_join(
    left: pd.Series,
    right: pd.Series,
    op: str,
    multiple_conditions: bool,
    keep: str,
    return_ragged_arrays: bool = False,
) -> tuple:
    """
    Generic function to call any of the individual functions
    (_less_than_indices, _greater_than_indices,
    or _not_equal_indices).
    """
    strict = False

    if op in {
        _JoinOperator.GREATER_THAN.value,
        _JoinOperator.LESS_THAN.value,
        _JoinOperator.NOT_EQUAL.value,
    }:
        strict = True

    if op in less_than_join_types:
        return _less_than_indices(
            left=left,
            right=right,
            strict=strict,
            multiple_conditions=multiple_conditions,
            keep=keep,
            return_ragged_arrays=return_ragged_arrays,
        )
    if op in greater_than_join_types:
        return _greater_than_indices(
            left=left,
            right=right,
            strict=strict,
            multiple_conditions=multiple_conditions,
            keep=keep,
            return_ragged_arrays=return_ragged_arrays,
        )
    if op == _JoinOperator.NOT_EQUAL.value:
        return _not_equal_indices(left=left, right=right, keep=keep)
    return _equal_indices(
        left=left, right=right, return_ragged_arrays=return_ragged_arrays
    )


def _keep_output(keep: str, left: np.ndarray, right: np.ndarray):
    """return indices for left and right index based on the value of `keep`."""
    if keep == "all":
        return left, right
    grouped = pd.Series(right).groupby(left)
    if keep == "first":
        grouped = grouped.min()
        return grouped.index, grouped._values
    grouped = grouped.max()
    return grouped.index, grouped._values


def _change_case(
    obj: str,
    case_type: str,
) -> str:
    """Change case of obj."""
    case_types = {"preserve", "upper", "lower", "snake"}
    case_type = case_type.lower()
    if case_type not in case_types:
        raise JanitorError(f"type must be one of: {case_types}")

    if case_type == "preserve":
        return obj
    if case_type == "upper":
        return obj.upper()
    if case_type == "lower":
        return obj.lower()
    # Implementation adapted from: https://gist.github.com/jaytaylor/3660565
    # by @jtaylor
    obj = re.sub(pattern=r"(.)([A-Z][a-z]+)", repl=r"\1_\2", string=obj)
    obj = re.sub(pattern=r"([a-z0-9])([A-Z])", repl=r"\1_\2", string=obj)
    return obj.lower()


def _normalize_1(obj: str) -> str:
    """Perform normalization of obj."""
    FIXES = [(r"[ /:,?()\.-]", "_"), (r"['’]", ""), (r"[\xa0]", "_")]
    for search, replace in FIXES:
        obj = re.sub(pattern=search, repl=replace, string=obj)

    return obj


def _remove_special(
    obj: str,
) -> str:
    """Remove special characters from obj."""
    obj = [item for item in obj if item.isalnum() or (item == "_")]
    return "".join(obj)


def _strip_accents(
    obj: str,
) -> str:
    """Remove accents from obj.

    Inspired from [StackOverflow][so].

    [so]: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-strin
    """  # noqa: E501

    obj = [
        letter
        for letter in unicodedata.normalize("NFD", obj)
        if not unicodedata.combining(letter)
    ]
    return "".join(obj)


def _strip_underscores_func(
    obj: str,
    strip_underscores: Union[str, bool] = None,
) -> str:
    """Strip underscores from obj."""
    underscore_options = {None, "left", "right", "both", "l", "r", True}
    if strip_underscores not in underscore_options:
        raise JanitorError(
            f"strip_underscores must be one of: {underscore_options}"
        )

    if strip_underscores in {"left", "l"}:
        return obj.lstrip("_")
    if strip_underscores in {"right", "r"}:
        return obj.rstrip("_")
    if strip_underscores in {True, "both"}:
        return obj.strip("_")
    return obj


def _is_str_or_cat(index):
    """
    Check if the column/index is a string,
    or categorical with strings.
    """
    if isinstance(index.dtype, pd.CategoricalDtype):
        return is_string_dtype(index.categories)
    return is_string_dtype(index)
