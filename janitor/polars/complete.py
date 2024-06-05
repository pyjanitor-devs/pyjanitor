"""complete implementation for polars."""

from __future__ import annotations

from typing import Any

from janitor.utils import check, import_message

try:
    import polars as pl
    import polars.selectors as cs
    from polars.type_aliases import ColumnNameOrSelector
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def _complete(
    df: pl.DataFrame | pl.LazyFrame,
    columns: tuple[ColumnNameOrSelector],
    fill_value: dict | Any | pl.Expr,
    explicit: bool,
    sort: bool,
    by: ColumnNameOrSelector,
) -> pl.DataFrame | pl.LazyFrame:
    """
    This function computes the final output for the `complete` function.

    A DataFrame, with rows of missing values, if any, is returned.
    """
    if not columns:
        return df

    check("sort", sort, [bool])
    check("explicit", explicit, [bool])
    _columns = []
    for column in columns:
        if isinstance(column, str):
            col = pl.col(column).unique()
            if sort:
                col = col.sort()
            _columns.append(col)
        elif cs.is_selector(column):
            col = column.as_expr().unique()
            if sort:
                col = col.sort()
            _columns.append(col)
        elif isinstance(column, pl.Expr):
            _columns.append(column)
        else:
            raise TypeError(
                f"The argument passed to the columns parameter "
                "should either be a string, a column selector, "
                "or a polars expression, instead got - "
                f"{type(column)}."
            )
    by_does_not_exist = by is None
    if by_does_not_exist:
        _columns = [column.implode() for column in _columns]
        uniques = df.select(_columns)
        _columns = uniques.columns
    else:
        uniques = df.group_by(by, maintain_order=sort).agg(_columns)
        _by = uniques.select(by).columns
        _columns = uniques.select(pl.exclude(_by)).columns
    for column in _columns:
        uniques = uniques.explode(column)

    _columns = [
        column
        for column, dtype in zip(_columns, uniques.select(_columns).dtypes)
        # this way we ensure there is no tampering with existing struct columns
        if (dtype == pl.Struct) and (column not in df.columns)
    ]

    if _columns:
        for column in _columns:
            uniques = uniques.unnest(columns=column)

    if fill_value is None:
        return uniques.join(df, on=uniques.columns, how="full", coalesce=True)
    idx = None
    columns_to_select = df.columns
    if not explicit:
        idx = "".join(df.columns)
        df = df.with_row_index(name=idx)
    df = uniques.join(df, on=uniques.columns, how="full", coalesce=True)
    # exclude columns that were not used
    # to generate the combinations
    exclude_columns = uniques.columns
    if idx:
        exclude_columns.append(idx)
    expression = pl.exclude(exclude_columns).is_null().any()
    booleans = df.select(expression)
    if isinstance(booleans, pl.LazyFrame):
        booleans = booleans.collect()
    _columns = [
        column
        for column in booleans.columns
        if booleans.get_column(column).item()
    ]
    if _columns and isinstance(fill_value, dict):
        fill_value = [
            pl.col(column_name).fill_null(value=value)
            for column_name, value in fill_value.items()
            if column_name in _columns
        ]
    elif _columns:
        fill_value = [
            pl.col(column).fill_null(value=fill_value) for column in _columns
        ]
    if _columns and not explicit:
        condition = pl.col(idx).is_null()
        fill_value = [
            pl.when(condition).then(_fill_value).otherwise(pl.col(column_name))
            for column_name, _fill_value in zip(_columns, fill_value)
        ]
    if _columns:
        df = df.with_columns(fill_value)

    return df.select(columns_to_select)
