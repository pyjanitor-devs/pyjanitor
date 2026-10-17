"""pivot_longer implementation for polars."""

from __future__ import annotations

from janitor.utils import check, import_message

from .polars_flavor import register_dataframe_method, register_lazyframe_method

try:
    import polars as pl
    import polars.selectors as cs
    from polars._typing import ColumnNameOrSelector
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def pivot_longer_spec(
    df: pl.DataFrame | pl.LazyFrame,
    spec: pl.DataFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """
    A declarative interface to pivot a Polars Frame
    from wide to long form,
    where you describe how the data will be unpivoted,
    using a DataFrame.

    It is modeled after tidyr's `pivot_longer_spec`.

    This gives you, the user,
    more control over the transformation to long form,
    using a *spec* DataFrame that describes exactly
    how data stored in the column names
    becomes variables.

    It can come in handy for situations where
    [`pivot_longer`][janitor.polars.pivot_longer.pivot_longer]
    seems inadequate for the transformation.

    !!! info "New in version 0.28.0"

    Examples:
        >>> import pandas as pd
        >>> from janitor.polars import pivot_longer_spec
        >>> df = pl.DataFrame(
        ...     {
        ...         "Sepal.Length": [5.1, 5.9],
        ...         "Sepal.Width": [3.5, 3.0],
        ...         "Petal.Length": [1.4, 5.1],
        ...         "Petal.Width": [0.2, 1.8],
        ...         "Species": ["setosa", "virginica"],
        ...     }
        ... )
        >>> df
        shape: (2, 5)
        ┌──────────────┬─────────────┬──────────────┬─────────────┬───────────┐
        │ Sepal.Length ┆ Sepal.Width ┆ Petal.Length ┆ Petal.Width ┆ Species   │
        │ ---          ┆ ---         ┆ ---          ┆ ---         ┆ ---       │
        │ f64          ┆ f64         ┆ f64          ┆ f64         ┆ str       │
        ╞══════════════╪═════════════╪══════════════╪═════════════╪═══════════╡
        │ 5.1          ┆ 3.5         ┆ 1.4          ┆ 0.2         ┆ setosa    │
        │ 5.9          ┆ 3.0         ┆ 5.1          ┆ 1.8         ┆ virginica │
        └──────────────┴─────────────┴──────────────┴─────────────┴───────────┘
        >>> spec = {
        ...     ".name": ["Sepal.Length", "Petal.Length", "Sepal.Width", "Petal.Width"],
        ...     ".value": ["Length", "Length", "Width", "Width"],
        ...     "part": ["Sepal", "Petal", "Sepal", "Petal"],
        ... }
        >>> spec = pl.DataFrame(spec)
        >>> spec
        shape: (4, 3)
        ┌──────────────┬────────┬───────┐
        │ .name        ┆ .value ┆ part  │
        │ ---          ┆ ---    ┆ ---   │
        │ str          ┆ str    ┆ str   │
        ╞══════════════╪════════╪═══════╡
        │ Sepal.Length ┆ Length ┆ Sepal │
        │ Petal.Length ┆ Length ┆ Petal │
        │ Sepal.Width  ┆ Width  ┆ Sepal │
        │ Petal.Width  ┆ Width  ┆ Petal │
        └──────────────┴────────┴───────┘
        >>> df.pipe(pivot_longer_spec, spec=spec).sort(by=pl.all())
        shape: (4, 4)
        ┌───────────┬───────┬────────┬───────┐
        │ Species   ┆ part  ┆ Length ┆ Width │
        │ ---       ┆ ---   ┆ ---    ┆ ---   │
        │ str       ┆ str   ┆ f64    ┆ f64   │
        ╞═══════════╪═══════╪════════╪═══════╡
        │ setosa    ┆ Petal ┆ 1.4    ┆ 0.2   │
        │ setosa    ┆ Sepal ┆ 5.1    ┆ 3.5   │
        │ virginica ┆ Petal ┆ 5.1    ┆ 1.8   │
        │ virginica ┆ Sepal ┆ 5.9    ┆ 3.0   │
        └───────────┴───────┴────────┴───────┘

    Args:
        df: The source DataFrame to unpivot.
            It can also be a LazyFrame.
        spec: A specification DataFrame.
            At a minimum, the spec DataFrame
            must have a `.name` column
            and a `.value` column.
            The `.name` column  should contain the
            columns in the source DataFrame that will be
            transformed to long form.
            The `.value` column gives the name of the column
            that the values in the source DataFrame will go into.
            Additional columns in the spec DataFrame
            should be named to match columns
            in the long format of the dataset and contain values
            corresponding to columns pivoted from the wide format.
            Note that these additional columns should not already exist
            in the source DataFrame.
            If there are additional columns, the combination of these columns
            and the `.value` column must be unique.

    Raises:
        KeyError: If `.name` or `.value` is missing from the spec's columns.
        ValueError: If the labels in spec's `.name` column is not unique.

    Returns:
        A polars DataFrame/LazyFrame.
    """
    check("spec", spec, [pl.DataFrame])
    spec_columns = spec.columns
    if ".name" not in spec_columns:
        raise KeyError("Kindly ensure the spec DataFrame has a `.name` column.")
    if ".value" not in spec_columns:
        raise KeyError("Kindly ensure the spec DataFrame has a `.value` column.")
    if spec.get_column(".name").is_duplicated().any():
        raise ValueError("The labels in the `.name` column should be unique.")
    df_columns = df.collect_schema().names()
    exclude = set(df_columns).intersection(spec_columns)
    if exclude:
        raise ValueError(
            f"Labels {(*exclude,)} in the spec dataframe already exist "
            "as column labels in the source dataframe. "
            "Kindly ensure the spec DataFrame's columns "
            "are not present in the source DataFrame."
        )

    index = [label for label in df_columns if label not in spec.get_column(".name")]
    others = [label for label in spec_columns if label not in {".name", ".value"}]
    return _pivot_longer_dot_value(
        df=df,
        index=index,
        spec=spec.select(".name", ".value", *others),
        names_cols=others,
        names_transform=None,
    )


@register_lazyframe_method
@register_dataframe_method
def pivot_longer(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector = None,
    column_names: ColumnNameOrSelector = None,
    names_to: list | tuple | str = "variable",
    values_to: str = "value",
    names_sep: str = None,
    names_pattern: str = None,
    names_transform: pl.Expr = None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Unpivots a DataFrame from *wide* to *long* format.

    It is modeled after the `pivot_longer` function in R's tidyr package,
    and also takes inspiration from the `melt` function in R's data.table package.

    This function is useful to massage a DataFrame into a format where
    one or more columns are considered measured variables, and all other
    columns are considered as identifier variables.

    All measured variables are *unpivoted* (and typically duplicated) along the
    row axis.

    If `names_pattern`, use a valid regular expression pattern containing at least
    one capture group, compatible with the [regex crate](https://docs.rs/regex/latest/regex/).

    For more granular control on the unpivoting, have a look at
    [`pivot_longer_spec`][janitor.polars.pivot_longer.pivot_longer_spec].

    `pivot_longer` can also be applied to a LazyFrame.

    Examples:
        >>> import polars as pl
        >>> import polars.selectors as cs
        >>> import janitor.polars
        >>> df = pl.DataFrame(
        ...     {
        ...         "Sepal.Length": [5.1, 5.9],
        ...         "Sepal.Width": [3.5, 3.0],
        ...         "Petal.Length": [1.4, 5.1],
        ...         "Petal.Width": [0.2, 1.8],
        ...         "Species": ["setosa", "virginica"],
        ...     }
        ... )
        >>> df
        shape: (2, 5)
        ┌──────────────┬─────────────┬──────────────┬─────────────┬───────────┐
        │ Sepal.Length ┆ Sepal.Width ┆ Petal.Length ┆ Petal.Width ┆ Species   │
        │ ---          ┆ ---         ┆ ---          ┆ ---         ┆ ---       │
        │ f64          ┆ f64         ┆ f64          ┆ f64         ┆ str       │
        ╞══════════════╪═════════════╪══════════════╪═════════════╪═══════════╡
        │ 5.1          ┆ 3.5         ┆ 1.4          ┆ 0.2         ┆ setosa    │
        │ 5.9          ┆ 3.0         ┆ 5.1          ┆ 1.8         ┆ virginica │
        └──────────────┴─────────────┴──────────────┴─────────────┴───────────┘

        Replicate polars' [melt](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.unpivot.html#polars-dataframe-melt):
        >>> df.pivot_longer(index="Species").sort(by=pl.all())
        shape: (8, 3)
        ┌───────────┬──────────────┬───────┐
        │ Species   ┆ variable     ┆ value │
        │ ---       ┆ ---          ┆ ---   │
        │ str       ┆ str          ┆ f64   │
        ╞═══════════╪══════════════╪═══════╡
        │ setosa    ┆ Petal.Length ┆ 1.4   │
        │ setosa    ┆ Petal.Width  ┆ 0.2   │
        │ setosa    ┆ Sepal.Length ┆ 5.1   │
        │ setosa    ┆ Sepal.Width  ┆ 3.5   │
        │ virginica ┆ Petal.Length ┆ 5.1   │
        │ virginica ┆ Petal.Width  ┆ 1.8   │
        │ virginica ┆ Sepal.Length ┆ 5.9   │
        │ virginica ┆ Sepal.Width  ┆ 3.0   │
        └───────────┴──────────────┴───────┘

        Split the column labels into individual columns:
        >>> df.pivot_longer(
        ...     index="Species",
        ...     names_to=("part", "dimension"),
        ...     names_sep=".",
        ... ).select("Species", "part", "dimension", "value").sort(by=pl.all())
        shape: (8, 4)
        ┌───────────┬───────┬───────────┬───────┐
        │ Species   ┆ part  ┆ dimension ┆ value │
        │ ---       ┆ ---   ┆ ---       ┆ ---   │
        │ str       ┆ str   ┆ str       ┆ f64   │
        ╞═══════════╪═══════╪═══════════╪═══════╡
        │ setosa    ┆ Petal ┆ Length    ┆ 1.4   │
        │ setosa    ┆ Petal ┆ Width     ┆ 0.2   │
        │ setosa    ┆ Sepal ┆ Length    ┆ 5.1   │
        │ setosa    ┆ Sepal ┆ Width     ┆ 3.5   │
        │ virginica ┆ Petal ┆ Length    ┆ 5.1   │
        │ virginica ┆ Petal ┆ Width     ┆ 1.8   │
        │ virginica ┆ Sepal ┆ Length    ┆ 5.9   │
        │ virginica ┆ Sepal ┆ Width     ┆ 3.0   │
        └───────────┴───────┴───────────┴───────┘

        Retain parts of the column names as headers:
        >>> df.pivot_longer(
        ...     index="Species",
        ...     names_to=("part", ".value"),
        ...     names_sep=".",
        ... ).select("Species", "part", "Length", "Width").sort(by=pl.all())
        shape: (4, 4)
        ┌───────────┬───────┬────────┬───────┐
        │ Species   ┆ part  ┆ Length ┆ Width │
        │ ---       ┆ ---   ┆ ---    ┆ ---   │
        │ str       ┆ str   ┆ f64    ┆ f64   │
        ╞═══════════╪═══════╪════════╪═══════╡
        │ setosa    ┆ Petal ┆ 1.4    ┆ 0.2   │
        │ setosa    ┆ Sepal ┆ 5.1    ┆ 3.5   │
        │ virginica ┆ Petal ┆ 5.1    ┆ 1.8   │
        │ virginica ┆ Sepal ┆ 5.9    ┆ 3.0   │
        └───────────┴───────┴────────┴───────┘

        Split the column labels based on regex:
        >>> df = pl.DataFrame({"id": [1], "new_sp_m5564": [2], "newrel_f65": [3]})
        >>> df
        shape: (1, 3)
        ┌─────┬──────────────┬────────────┐
        │ id  ┆ new_sp_m5564 ┆ newrel_f65 │
        │ --- ┆ ---          ┆ ---        │
        │ i64 ┆ i64          ┆ i64        │
        ╞═════╪══════════════╪════════════╡
        │ 1   ┆ 2            ┆ 3          │
        └─────┴──────────────┴────────────┘
        >>> df.pivot_longer(
        ...     index="id",
        ...     names_to=("diagnosis", "gender", "age"),
        ...     names_pattern=r"new_?(.+)_(.)([0-9]+)",
        ... ).select("id", "diagnosis", "gender", "age", "value").sort(by=pl.all())
        shape: (2, 5)
        ┌─────┬───────────┬────────┬──────┬───────┐
        │ id  ┆ diagnosis ┆ gender ┆ age  ┆ value │
        │ --- ┆ ---       ┆ ---    ┆ ---  ┆ ---   │
        │ i64 ┆ str       ┆ str    ┆ str  ┆ i64   │
        ╞═════╪═══════════╪════════╪══════╪═══════╡
        │ 1   ┆ rel       ┆ f      ┆ 65   ┆ 3     │
        │ 1   ┆ sp        ┆ m      ┆ 5564 ┆ 2     │
        └─────┴───────────┴────────┴──────┴───────┘

        Convert the dtypes of specific columns with `names_transform`:
        >>> df.pivot_longer(
        ...     index="id",
        ...     names_pattern=r"new_?(.+)_(.)([0-9]+)",
        ...     names_to=("diagnosis", "gender", "age"),
        ...     names_transform=pl.col("age").cast(pl.Int32),
        ... ).select("id", "diagnosis", "gender", "age", "value").sort(by=pl.all())
        shape: (2, 5)
        ┌─────┬───────────┬────────┬──────┬───────┐
        │ id  ┆ diagnosis ┆ gender ┆ age  ┆ value │
        │ --- ┆ ---       ┆ ---    ┆ ---  ┆ ---   │
        │ i64 ┆ str       ┆ str    ┆ i32  ┆ i64   │
        ╞═════╪═══════════╪════════╪══════╪═══════╡
        │ 1   ┆ rel       ┆ f      ┆ 65   ┆ 3     │
        │ 1   ┆ sp        ┆ m      ┆ 5564 ┆ 2     │
        └─────┴───────────┴────────┴──────┴───────┘

        Use multiple `.value` to reshape the dataframe:
        >>> df = pl.DataFrame(
        ...     [
        ...         {
        ...             "x_1_mean": 10,
        ...             "x_2_mean": 20,
        ...             "y_1_mean": 30,
        ...             "y_2_mean": 40,
        ...             "unit": 50,
        ...         }
        ...     ]
        ... )
        >>> df
        shape: (1, 5)
        ┌──────────┬──────────┬──────────┬──────────┬──────┐
        │ x_1_mean ┆ x_2_mean ┆ y_1_mean ┆ y_2_mean ┆ unit │
        │ ---      ┆ ---      ┆ ---      ┆ ---      ┆ ---  │
        │ i64      ┆ i64      ┆ i64      ┆ i64      ┆ i64  │
        ╞══════════╪══════════╪══════════╪══════════╪══════╡
        │ 10       ┆ 20       ┆ 30       ┆ 40       ┆ 50   │
        └──────────┴──────────┴──────────┴──────────┴──────┘
        >>> df.pivot_longer(
        ...     index="unit",
        ...     names_to=(".value", "time", ".value"),
        ...     names_pattern=r"(x|y)_([0-9])(_mean)",
        ... ).select("unit", "time", "x_mean", "y_mean").sort(by=pl.all())
        shape: (2, 4)
        ┌──────┬──────┬────────┬────────┐
        │ unit ┆ time ┆ x_mean ┆ y_mean │
        │ ---  ┆ ---  ┆ ---    ┆ ---    │
        │ i64  ┆ str  ┆ i64    ┆ i64    │
        ╞══════╪══════╪════════╪════════╡
        │ 50   ┆ 1    ┆ 10     ┆ 30     │
        │ 50   ┆ 2    ┆ 20     ┆ 40     │
        └──────┴──────┴────────┴────────┘

    !!! info "New in version 0.28.0"

    Args:
        index: Column(s) or selector(s) to use as identifier variables.
        column_names: Column(s) or selector(s) to unpivot.
        names_to: Name of new column as a string that will contain
            what were previously the column names in `column_names`.
            The default is `variable` if no value is provided. It can
            also be a list/tuple of strings that will serve as new column
            names, if `name_sep` or `names_pattern` is provided.
            If `.value` is in `names_to`, new column names will be extracted
            from part of the existing column names and overrides `values_to`.
        values_to: Name of new column as a string that will contain what
            were previously the values of the columns in `column_names`.
        names_sep: Determines how the column name is broken up, if
            `names_to` contains multiple values. It takes the same
            specification as polars' `str.split` method.
        names_pattern: Determines how the column name is broken up.
            It can be a regular expression containing matching groups.
            It takes the same specification as
            polars' `str.extract_groups` method.
        names_transform: Use this option to change the types of columns that
            have been transformed to rows.
            This does not applies to the values' columns.
            Accepts a polars expression or a list of polars expressions.
            Applicable only if one of names_sep
            or names_pattern is provided.

    Returns:
        A polars DataFrame/LazyFrame that has been unpivoted
        from wide to long format.
    """  # noqa: E501
    return _pivot_longer(
        df=df,
        index=index,
        column_names=column_names,
        names_pattern=names_pattern,
        names_sep=names_sep,
        names_to=names_to,
        values_to=values_to,
        names_transform=names_transform,
    )


def _pivot_longer(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple | str | None,
    values_to: str,
    names_sep: str,
    names_pattern: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Unpivots a DataFrame/LazyFrame from wide to long form.
    """

    if all((names_pattern is None, names_sep is None)):
        return df.unpivot(
            index=index,
            on=column_names,
            variable_name=names_to,
            value_name=values_to,
        )

    schema = df.collect_schema()
    (
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
    ) = _data_checks_pivot_longer(
        df=df,
        schema=schema,
        index=index,
        column_names=column_names,
        names_to=names_to,
        values_to=values_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
    )

    variable_name = "".join(schema.names())
    variable_name = f"{variable_name}_"
    spec = _pivot_longer_create_spec(
        column_names=column_names,
        names_to=names_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
        variable_name=variable_name,
    )

    if ".value" not in names_to:
        return _pivot_longer_no_dot_value(
            df=df,
            index=index,
            spec=spec,
            column_names=column_names,
            names_to=names_to,
            values_to=values_to,
            variable_name=variable_name,
            names_transform=names_transform,
        )
    return _pivot_longer_dot_value(
        df=df,
        index=index,
        spec=spec,
        names_cols=[
            label for label in spec.columns if label not in {".name", ".value"}
        ],
        names_transform=names_transform,
    )


def _pivot_longer_create_spec(
    column_names: list,
    names_to: list,
    names_sep: str | None,
    names_pattern: str | None,
    variable_name: str,
) -> pl.DataFrame:
    """
    This is where the spec DataFrame is created,
    before the transformation to long form.
    """
    spec = pl.DataFrame({".name": column_names})
    if names_sep is not None:
        expression = (
            pl.col(".name")
            .str.split(by=names_sep)
            .list.to_struct(upper_bound=len(names_to))
            .alias(variable_name)
        )
    else:
        expression = (
            pl.col(".name")
            .str.extract_groups(pattern=names_pattern)
            .alias(variable_name)
        )
    spec = spec.with_columns(expression)
    len_fields = len(spec.get_column(variable_name).struct.fields)
    len_names_to = len(names_to)

    if len_names_to != len_fields:
        raise ValueError(
            f"The length of names_to does not match "
            "the number of fields extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of fields extracted is "
            f"{len_fields}."
        )
    if names_pattern is not None:
        expression = pl.exclude(".name").is_null().any()
        expression = pl.any_horizontal(expression)
        null_check = (
            spec.unnest(columns=variable_name).filter(expression).get_column(".name")
        )
        if null_check.len():
            column_name = null_check.gather(0).item()
            raise ValueError(
                f"Column label '{column_name}' "
                "could not be matched with any of the groups "
                "in the provided regex. Kindly provide a regular expression "
                "(with the correct groups) that matches all labels in the columns."
            )

    if ".value" not in names_to:
        spec = spec.get_column(variable_name)
        spec = spec.struct.rename_fields(names=names_to)
        return spec
    fields = spec.get_column(variable_name).struct.fields
    parts = spec.get_column(variable_name).struct.unnest()
    dot_value = [field for field, label in zip(fields, names_to) if label == ".value"]
    not_dot_value = [
        pl.col(field).alias(label)
        for field, label in zip(fields, names_to)
        if label != ".value"
    ]
    return parts.select(
        spec.get_column(".name"),
        pl.concat_str(dot_value).alias(".value"),
        *not_dot_value,
    )


def _pivot_longer_no_dot_value(
    df: pl.DataFrame | pl.LazyFrame,
    spec: pl.DataFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple,
    values_to: str,
    variable_name: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    flip polars Frame to long form,
    if no .value in names_to.
    """
    # the implode/explode approach is used here
    # for efficiency
    # do the operation on a smaller size
    # and then blow it up after
    # it is usually much faster
    # than unpivoting and running the string operations after
    outcome = (
        df.select(pl.all().implode())
        .unpivot(
            index=index,
            on=column_names,
            variable_name=variable_name,
            value_name=values_to,
        )
        .with_columns(spec)
    )

    outcome = outcome.unnest(variable_name)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)
    return outcome.explode(cs.exclude(*names_to))


def _pivot_longer_dot_value(
    df: pl.DataFrame | pl.LazyFrame,
    spec: pl.DataFrame,
    index: ColumnNameOrSelector,
    names_cols: list[str],
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Build one select per spec group and concatenate the resulting frames.

    The spec is small and eager, while the user's frame may be lazy. Keeping
    the grouping and header construction eager avoids the struct-pack,
    unpivot, and unnest work used by the previous implementation.
    """
    key = names_cols
    if not key:
        key = ["__pivot_longer_group__"]
        spec = spec.with_columns(pl.int_range(pl.len()).over(".value").alias(key[0]))
    else:
        keys = spec.select([*names_cols, ".value"]).rows()
        if len(keys) != len(set(keys)):
            raise ValueError(
                "The combination of the `.value` column and additional "
                "specification columns should be unique."
            )

    if spec.get_column(".value").is_null().any():
        raise ValueError("The `.value` column cannot contain null values.")

    groups = spec.group_by(key).agg(".name", ".value")
    if names_transform is not None:
        groups = groups.with_columns(names_transform)

    fields = spec.get_column(".value").unique(maintain_order=True).to_list()
    lazy = isinstance(df, pl.LazyFrame)
    frame = df if lazy else df.lazy()
    index_exprs = [pl.col(label) for label in index]
    parts = []
    for position, (names, values) in enumerate(
        zip(
            groups.get_column(".name").to_list(),
            groups.get_column(".value").to_list(),
        )
    ):
        mapping = dict(zip(values, names))
        headers = [
            pl.lit(groups.get_column(label).slice(position, 1)).first().alias(label)
            for label in names_cols
        ]
        columns = [
            pl.col(mapping[field]).alias(field)
            if field in mapping
            else pl.lit(None).alias(field)
            for field in fields
        ]
        parts.append(frame.select(*index_exprs, *headers, *columns))

    outcome = pl.concat(parts, how="vertical_relaxed")
    return outcome if lazy else outcome.collect()


def _data_checks_pivot_longer(
    df,
    schema,
    index,
    column_names,
    names_to,
    values_to,
    names_sep,
    names_pattern,
) -> tuple:
    """
    This function majorly does type checks on the passed arguments.

    This function is executed before proceeding to the computation phase.

    Type annotations are not provided because this function is where type
    checking happens.
    """
    if isinstance(names_to, str):
        names_to = [names_to]
    elif isinstance(names_to, (list, tuple)):
        uniques = set()
        for word in names_to:
            if not isinstance(word, str):
                raise TypeError(
                    f"'{word}' in names_to should be a string type; "
                    f"instead got type {type(word).__name__}"
                )
            if (word in uniques) and (word != ".value"):
                raise ValueError(f"'{word}' is duplicated in names_to.")
            uniques.add(word)
    else:
        raise TypeError(
            "names_to should be a string, list, or tuple; "
            f"instead got type {type(names_to).__name__}"
        )

    if names_sep and names_pattern:
        raise ValueError("Only one of names_pattern or names_sep should be provided.")

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    else:
        check("names_pattern", names_pattern, [str])

    check("values_to", values_to, [str])

    if (index is None) and (column_names is None):
        column_names = schema.names()
        index = []
    elif (index is None) and (column_names is not None):
        column_names = _resolve_columns(df, schema, column_names)
        excluded = set(column_names)
        index = [label for label in schema.names() if label not in excluded]
    elif (index is not None) and (column_names is None):
        index = _resolve_columns(df, schema, index)
        excluded = set(index)
        column_names = [label for label in schema.names() if label not in excluded]
    else:
        index = _resolve_columns(df, schema, index)
        column_names = _resolve_columns(df, schema, column_names)

    return (
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
    )


def _resolve_columns(df, schema, selection):
    """Resolve a column selection against an already-collected schema."""
    if isinstance(selection, str):
        selection = [selection]
    if isinstance(selection, (list, tuple)) and all(
        isinstance(label, str) for label in selection
    ):
        if not any(
            label.startswith("^") and label.endswith("$") for label in selection
        ):
            missing = [label for label in selection if label not in schema]
            if missing:
                raise pl.exceptions.ColumnNotFoundError(
                    f'unable to find column "{missing[0]}"; '
                    f"valid columns: {schema.names()}"
                )
            return list(selection)
    # Regex strings, selectors, and expressions need Polars expansion. An eager
    # frame's schema is already known; a lazy frame would re-resolve its whole
    # plan.
    frame = df if isinstance(df, pl.DataFrame) else pl.LazyFrame(schema=schema)
    return frame.select(selection).collect_schema().names()
