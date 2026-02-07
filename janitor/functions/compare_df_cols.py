"""Compare DataFrame column types before binding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import json
import warnings
from types import FrameType
from typing import Any, Iterable

import pandas as pd


_VALID_RETURN_VALUES = {"all", "match", "mismatch"}
_VALID_BIND_METHODS = {"bind_rows", "rbind"}


def describe_class(obj: Any, strict_description: bool = True) -> str:
    """Describe the class of an object or pandas series.

    This function is used by :func:`compare_df_cols` to summarize column types.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> janitor.describe_class(pd.Series([1, 2, 3]))
        'int64'
        >>> janitor.describe_class(pd.Categorical(["a", "b", "a"]))
        'category(levels=["a", "b"])'
        >>> janitor.describe_class(
        ...     pd.Categorical(["a", "b", "a"]), strict_description=False
        ... )
        'category'

    Args:
        obj: Object to describe.
        strict_description: Whether to include categorical levels in the
            description. Defaults to True.

    Returns:
        A string describing the object's class.
    """
    if isinstance(obj, pd.Categorical):
        return _describe_categorical(obj, strict_description)

    if isinstance(obj, pd.Series):
        if isinstance(obj.dtype, pd.CategoricalDtype):
            return _describe_categorical(obj, strict_description)
        return str(obj.dtype)

    dtype = getattr(obj, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype):
        return _describe_categorical(dtype, strict_description)
    if dtype is not None:
        return str(dtype)

    return type(obj).__name__


def compare_df_cols(
    *dfs: pd.DataFrame | Sequence[pd.DataFrame] | Mapping[str, pd.DataFrame],
    return_: str = "all",
    bind_method: str = "bind_rows",
    strict_description: bool = False,
    **named_dfs: pd.DataFrame | Sequence[pd.DataFrame] | Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare column types across DataFrames.

    Provide multiple DataFrames (or lists/dicts of DataFrames) to compare the
    column types prior to binding. Named arguments become output column names.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df1 = pd.DataFrame(
        ...     {
        ...         "A": [1, 2],
        ...         "B": pd.Series(["x", "y"], dtype="object"),
        ...     }
        ... )
        >>> df2 = pd.DataFrame(
        ...     {
        ...         "A": [3.0, 4.0],
        ...         "B": pd.Series(["z", "w"], dtype="object"),
        ...         "C": [True, False],
        ...     }
        ... )
        >>> df3 = pd.DataFrame(
        ...     {
        ...         "A": pd.Series(["a", "b"], dtype="object"),
        ...         "B": pd.Series(["c", "d"], dtype="object"),
        ...     }
        ... )
        >>> janitor.compare_df_cols(df1, df2, df3)
          column_name    df1     df2     df3
        0           A   int64  float64  object
        1           B  object  object  object
        2           C     NaN     bool     NaN
        >>> janitor.compare_df_cols(train=df1, test=df2)
          column_name  train    test
        0           A  int64  float64
        1           B  object  object
        2           C    NaN     bool
        >>> janitor.compare_df_cols(train=df1, test=df2, return_="mismatch")
          column_name  train    test
        0           A  int64  float64

    Args:
        *dfs: DataFrames or lists/dicts of DataFrames to compare.
        return_: Whether to return "all", only "match"ing columns, or only
            "mismatch"ing columns. Defaults to "all".
        bind_method: "bind_rows" treats missing columns as matching; "rbind"
            treats missing columns as mismatches. Defaults to "bind_rows".
        strict_description: Whether to include categorical levels in type
            descriptions. Defaults to False.
        **named_dfs: Named DataFrames or lists/dicts of DataFrames to compare.

    Raises:
        TypeError: If inputs are not DataFrames or lists/dicts of DataFrames.
        ValueError: If an invalid option is provided or no DataFrames are given.

    Returns:
        A DataFrame with a "column_name" column and one column per input
        DataFrame containing the dtype description.
    """
    if "return" in named_dfs:
        return_ = named_dfs.pop("return")

    if return_ not in _VALID_RETURN_VALUES:
        raise ValueError(
            "return_ must be one of 'all', 'match', or 'mismatch'. "
            f"Received '{return_}'."
        )
    if bind_method not in _VALID_BIND_METHODS:
        raise ValueError(
            "bind_method must be 'bind_rows' or 'rbind'. "
            f"Received '{bind_method}'."
        )

    if not dfs and not named_dfs:
        raise ValueError("At least one DataFrame is required.")

    inferred_names = _infer_arg_names(dfs)
    entries: list[tuple[str, pd.DataFrame]] = []
    for idx, (arg, inferred_name) in enumerate(
        zip(dfs, inferred_names, strict=True)
    ):
        default_name = inferred_name or f"df{idx + 1}"
        entries.extend(_expand_input(arg, default_name))
    for name, arg in named_dfs.items():
        entries.extend(_expand_input(arg, name))

    if not entries:
        raise ValueError("At least one DataFrame is required.")

    for name, _ in entries:
        if name == "column_name":
            raise ValueError(
                "None of the input names may be 'column_name'."
            )

    frames: list[pd.DataFrame] = []
    dataframes: list[pd.DataFrame] = []
    for name, df in entries:
        frames.append(_describe_dataframe(df, name, strict_description))
        dataframes.append(df)

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="column_name", how="outer", sort=False)

    if result["column_name"].is_unique:
        column_order = _column_order(dataframes)
        if column_order:
            result = (
                result.set_index("column_name")
                .reindex(column_order)
                .reset_index()
            )

    if return_ == "all" or result.shape[1] <= 2:
        if return_ != "all":
            warnings.warn(
                "Only one DataFrame provided, so all column descriptions are returned.",
                stacklevel=2,
            )
        return result

    matches = _match_rows(result, bind_method)
    if return_ == "match":
        return result[matches].reset_index(drop=True)
    return result[~matches].reset_index(drop=True)


def compare_df_cols_same(
    *dfs: pd.DataFrame | Sequence[pd.DataFrame] | Mapping[str, pd.DataFrame],
    bind_method: str = "bind_rows",
    verbose: bool = True,
    **named_dfs: pd.DataFrame | Sequence[pd.DataFrame] | Mapping[str, pd.DataFrame],
) -> bool:
    """Check if DataFrames can be safely row-bound.

    This function returns True if there are no mismatching column types.

    Examples:
        >>> import pandas as pd
        >>> import janitor
        >>> df1 = pd.DataFrame({"A": [1, 2]})
        >>> df2 = pd.DataFrame({"A": [3.0, 4.0]})
        >>> janitor.compare_df_cols_same(df1, df2, verbose=False)
        False

    Args:
        *dfs: DataFrames or lists/dicts of DataFrames to compare.
        bind_method: "bind_rows" treats missing columns as matching; "rbind"
            treats missing columns as mismatches. Defaults to "bind_rows".
        verbose: Whether to print mismatching columns. Defaults to True.
        **named_dfs: Named DataFrames or lists/dicts of DataFrames to compare.

    Returns:
        True if binding is safe, otherwise False.
    """
    mismatches = compare_df_cols(
        *dfs,
        bind_method=bind_method,
        return_="mismatch",
        **named_dfs,
    )
    if not mismatches.empty and verbose:
        print(mismatches)
    return mismatches.empty


def _describe_categorical(
    obj: pd.Categorical | pd.Series | pd.CategoricalDtype,
    strict_description: bool,
) -> str:
    if not strict_description:
        return "category"

    if isinstance(obj, pd.Series):
        categories = obj.cat.categories
        ordered = obj.cat.ordered
    elif isinstance(obj, pd.Categorical):
        categories = obj.categories
        ordered = obj.ordered
    else:
        categories = obj.categories
        ordered = obj.ordered

    levels_text = ", ".join(_format_level(level) for level in categories)
    category_text = f"category(levels=[{levels_text}])"
    if ordered:
        return f"ordered, {category_text}"
    return category_text


def _format_level(level: Any) -> str:
    try:
        return json.dumps(level)
    except TypeError:
        return repr(level)


def _infer_arg_names(args: Iterable[Any]) -> list[str | None]:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return [None for _ in args]
    try:
        caller = frame.f_back
        names: list[str | None] = []
        for arg in args:
            names.append(_find_name_in_frame(arg, caller))
        return names
    finally:
        del frame


def _find_name_in_frame(value: Any, frame: FrameType) -> str | None:
    for scope in (frame.f_locals, frame.f_globals):
        for name, candidate in scope.items():
            if candidate is value:
                return name
    return None


def _expand_input(
    arg: pd.DataFrame | Sequence[pd.DataFrame] | Mapping[str, pd.DataFrame],
    base_name: str | None,
) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(arg, pd.DataFrame):
        name = _validate_name(base_name)
        return [(name, arg)]

    if isinstance(arg, Mapping):
        entries = []
        for idx, (name, df) in enumerate(arg.items(), start=1):
            if name is None or name == "":
                name = f"{_validate_name(base_name)}_{idx}"
            name = _validate_name(name)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    "All mapping values must be pandas DataFrames. "
                    f"Received {type(df).__name__}."
                )
            entries.append((name, df))
        return entries

    if isinstance(arg, Sequence):
        entries = []
        for idx, df in enumerate(arg, start=1):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    "All sequence items must be pandas DataFrames. "
                    f"Received {type(df).__name__}."
                )
            name = _validate_name(base_name)
            entries.append((f"{name}_{idx}", df))
        return entries

    raise TypeError(
        "Inputs must be pandas DataFrames or sequences/mappings of DataFrames. "
        f"Received {type(arg).__name__}."
    )


def _validate_name(name: str | None) -> str:
    if name is None or name == "":
        return "df"
    if not isinstance(name, str):
        raise TypeError("DataFrame names must be strings.")
    return name


def _describe_dataframe(
    df: pd.DataFrame, name: str, strict_description: bool
) -> pd.DataFrame:
    if df.shape[1] == 0:
        warnings.warn(
            f"{name} has zero columns and will not appear in output.",
            stacklevel=2,
        )
        return pd.DataFrame({"column_name": []})

    column_names = []
    descriptions = []
    for column_name, series in df.items():
        column_names.append(column_name)
        descriptions.append(
            describe_class(series, strict_description=strict_description)
        )

    return pd.DataFrame(
        {
            "column_name": column_names,
            name: descriptions,
        }
    )


def _column_order(dataframes: Iterable[pd.DataFrame]) -> list[Any]:
    order: list[Any] = []
    for df in dataframes:
        for column in df.columns:
            if column not in order:
                order.append(column)
    return order


def _match_rows(result: pd.DataFrame, bind_method: str) -> pd.Series:
    data = result.iloc[:, 1:]
    if bind_method == "rbind":
        return data.apply(_row_matches_rbind, axis=1)
    return data.apply(_row_matches_bind_rows, axis=1)


def _row_matches_rbind(row: pd.Series) -> bool:
    values = row.to_list()
    first = values[0]
    if pd.isna(first):
        return False
    return all(value == first for value in values[1:])


def _row_matches_bind_rows(row: pd.Series) -> bool:
    values = [value for value in row.to_list() if not pd.isna(value)]
    if len(values) <= 1:
        return True
    first = values[0]
    return all(value == first for value in values[1:])
