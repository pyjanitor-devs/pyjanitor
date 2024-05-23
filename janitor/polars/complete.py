"""complete implementation for polars."""

from janitor.utils import import_message

try:
    import polars as pl
    import polars.selectors as cs
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def _complete(
    df: pl.DataFrame | pl.LazyFrame,
    columns: tuple,
    fill_value: dict | int | float | str,
    explicit: bool,
    sort: bool,
):
    """
    This function computes the final output for the `complete` function.

    A DataFrame, with rows of missing values, if any, is returned.
    """
    if not columns:
        return df

    all_strings = (isinstance(column, str) for column in columns)
    all_strings = all(all_strings)
    if all_strings:
        _columns = pl.col(columns)
    else:
        _columns = []
        for column in columns:
            if isinstance(column, str):
                _columns.append(pl.col(column))
            elif cs.is_selector(column):
                _columns.append(column.as_expr())
            elif isinstance(column, pl.Expr):
                _columns.append(column)
            else:
                raise TypeError(
                    f"The argument passed to the columns parameter "
                    "should either be a string, a column selector, "
                    "or a polars expression, instead got - "
                    f"{type(column)}."
                )

    if sort and all_strings:
        _columns = _columns.unique().sort().implode()
    elif all_strings:
        _columns = _columns.unique().implode()
    elif sort:
        _columns = [column.unique().sort().implode() for column in _columns]
    else:
        _columns = [column.unique().implode() for column in _columns]
    uniques = df.select(_columns)

    for column in uniques.columns:
        uniques = uniques.explode(column)

    structs_exist = any(entry == pl.Struct for entry in uniques.dtypes)

    if structs_exist:
        for label, dtype in zip(uniques.columns, uniques.dtypes):
            if dtype == pl.Struct:
                uniques = uniques.unnest(label)

    df = uniques.join(df, on=uniques.columns, how="left")
    return df
