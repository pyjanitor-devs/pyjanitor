"""pivot_wider_spec implementation for polars."""

from __future__ import annotations

from janitor.utils import check, import_message

try:
    import polars as pl
    from polars._typing import ColumnNameOrSelector
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def pivot_wider_spec(
    df: pl.DataFrame,
    spec: pl.DataFrame,
    index: ColumnNameOrSelector = None,
) -> pl.DataFrame:
    """A declarative interface to pivot a DataFrame from long to wide form,
    where you describe how the data will be pivoted,
    using a DataFrame.

    This gives you, the user,
    more control over pivoting, where you create a “spec”
    data frame that describes exactly how data stored
    in the column names becomes variables.

    It can come in handy for situations where
    `pl.DataFrame.pivot`
    seems inadequate for the transformation.

    !!! info "New in version 0.31.0"

    Examples:
        >>> import polars as pl
        >>> from janitor.polars import pivot_wider_spec
        >>> df = pl.DataFrame(
        ... [
        ...    {"famid": 1, "birth": 1, "age": 1, "ht": 2.8},
        ...    {"famid": 1, "birth": 1, "age": 2, "ht": 3.4},
        ...    {"famid": 1, "birth": 2, "age": 1, "ht": 2.9},
        ...    {"famid": 1, "birth": 2, "age": 2, "ht": 3.8},
        ...    {"famid": 1, "birth": 3, "age": 1, "ht": 2.2},
        ...    {"famid": 1, "birth": 3, "age": 2, "ht": 2.9},
        ...    {"famid": 2, "birth": 1, "age": 1, "ht": 2.0},
        ...    {"famid": 2, "birth": 1, "age": 2, "ht": 3.2},
        ...    {"famid": 2, "birth": 2, "age": 1, "ht": 1.8},
        ...    {"famid": 2, "birth": 2, "age": 2, "ht": 2.8},
        ...    {"famid": 2, "birth": 3, "age": 1, "ht": 1.9},
        ...    {"famid": 2, "birth": 3, "age": 2, "ht": 2.4},
        ...    {"famid": 3, "birth": 1, "age": 1, "ht": 2.2},
        ...    {"famid": 3, "birth": 1, "age": 2, "ht": 3.3},
        ...    {"famid": 3, "birth": 2, "age": 1, "ht": 2.3},
        ...    {"famid": 3, "birth": 2, "age": 2, "ht": 3.4},
        ...    {"famid": 3, "birth": 3, "age": 1, "ht": 2.1},
        ...    {"famid": 3, "birth": 3, "age": 2, "ht": 2.9},
        ... ]
        ... )
        >>> df
        shape: (18, 4)
        ┌───────┬───────┬─────┬─────┐
        │ famid ┆ birth ┆ age ┆ ht  │
        │ ---   ┆ ---   ┆ --- ┆ --- │
        │ i64   ┆ i64   ┆ i64 ┆ f64 │
        ╞═══════╪═══════╪═════╪═════╡
        │ 1     ┆ 1     ┆ 1   ┆ 2.8 │
        │ 1     ┆ 1     ┆ 2   ┆ 3.4 │
        │ 1     ┆ 2     ┆ 1   ┆ 2.9 │
        │ 1     ┆ 2     ┆ 2   ┆ 3.8 │
        │ 1     ┆ 3     ┆ 1   ┆ 2.2 │
        │ 1     ┆ 3     ┆ 2   ┆ 2.9 │
        │ 2     ┆ 1     ┆ 1   ┆ 2.0 │
        │ 2     ┆ 1     ┆ 2   ┆ 3.2 │
        │ 2     ┆ 2     ┆ 1   ┆ 1.8 │
        │ 2     ┆ 2     ┆ 2   ┆ 2.8 │
        │ 2     ┆ 3     ┆ 1   ┆ 1.9 │
        │ 2     ┆ 3     ┆ 2   ┆ 2.4 │
        │ 3     ┆ 1     ┆ 1   ┆ 2.2 │
        │ 3     ┆ 1     ┆ 2   ┆ 3.3 │
        │ 3     ┆ 2     ┆ 1   ┆ 2.3 │
        │ 3     ┆ 2     ┆ 2   ┆ 3.4 │
        │ 3     ┆ 3     ┆ 1   ┆ 2.1 │
        │ 3     ┆ 3     ┆ 2   ┆ 2.9 │
        └───────┴───────┴─────┴─────┘
        >>> spec = {".name": ["ht1", "ht2"],
        ...         ".value": ["ht", "ht"],
        ...         "age": [1, 2]}
        >>> spec = pl.DataFrame(spec)
        >>> spec
        shape: (2, 3)
        ┌───────┬────────┬─────┐
        │ .name ┆ .value ┆ age │
        │ ---   ┆ ---    ┆ --- │
        │ str   ┆ str    ┆ i64 │
        ╞═══════╪════════╪═════╡
        │ ht1   ┆ ht     ┆ 1   │
        │ ht2   ┆ ht     ┆ 2   │
        └───────┴────────┴─────┘
        >>> pivot_wider_spec(df=df,spec=spec, index=['famid','birth'])
        shape: (9, 4)
        ┌───────┬───────┬─────┬─────┐
        │ famid ┆ birth ┆ ht1 ┆ ht2 │
        │ ---   ┆ ---   ┆ --- ┆ --- │
        │ i64   ┆ i64   ┆ f64 ┆ f64 │
        ╞═══════╪═══════╪═════╪═════╡
        │ 1     ┆ 1     ┆ 2.8 ┆ 3.4 │
        │ 1     ┆ 2     ┆ 2.9 ┆ 3.8 │
        │ 1     ┆ 3     ┆ 2.2 ┆ 2.9 │
        │ 2     ┆ 1     ┆ 2.0 ┆ 3.2 │
        │ 2     ┆ 2     ┆ 1.8 ┆ 2.8 │
        │ 2     ┆ 3     ┆ 1.9 ┆ 2.4 │
        │ 3     ┆ 1     ┆ 2.2 ┆ 3.3 │
        │ 3     ┆ 2     ┆ 2.3 ┆ 3.4 │
        │ 3     ┆ 3     ┆ 2.1 ┆ 2.9 │
        └───────┴───────┴─────┴─────┘

    Args:
        df: A polars DataFrame.
        spec: A specification DataFrame.
            At a minimum, the spec DataFrame
            must have a '.name' and a '.value' columns.
            The '.name' column  should contain the
            the names of the columns in the output DataFrame.
            The '.value' column should contain the name of the column(s)
            in the source DataFrame that will be serve as the values.
            Additional columns in spec will serves as the columns
            to be flipped to wide form.
            Note that these additional columns should already exist
            in the source DataFrame.
        index: Column(s) or selector(s) to use as identifier variables

    Returns:
        A polars DataFrame that has been pivoted from long to wide form.
    """  # noqa: E501
    check("spec", spec, [pl.DataFrame])
    spec_columns = spec.collect_schema().names()
    if ".name" not in spec_columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.name` column."
        )
    if ".value" not in spec_columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.value` column."
        )
    if spec.get_column(".name").is_duplicated().any():
        raise ValueError("The labels in the `.name` column should be unique.")

    if spec_columns[:2] != [".name", ".value"]:
        raise ValueError(
            "The first two columns of the spec DataFrame "
            "should be '.name' and '.value', "
            "with '.name' coming before '.value'."
        )
    if len(spec_columns) == 2:
        raise ValueError(
            "Kindly provide the column(s) "
            "to use to make new frame’s columns"
        )
    # df_columns = df.collect_schema().names()
    # cols = spec[2:]
    if index is not None:
        index = df.select(index).collect_schema().names()
