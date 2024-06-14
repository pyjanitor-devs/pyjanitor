"""pivot_longer implementation for polars."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from janitor.utils import check, import_message

from .polars_flavor import register_dataframe_method, register_lazyframe_method

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


def pivot_longer_spec(
    df: pl.DataFrame | pl.LazyFrame,
    spec: pl.DataFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """
    A declarative interface to pivot a DataFrame
    from wide to long form,
    where you describe how the data will be unpivoted,
    using a DataFrame. This gives you, the user,
    more control over the transformation to long form,
    using a *spec* DataFrame that describes exactly
    how data stored in the column names
    becomes variables.

    It can come in handy for situations where
    `janitor.polars.pivot_longer`
    seems inadequate for the transformation.

    !!! info "New in version 0.28.0"

    Examples:
        >>> import pandas as pd
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
        >>> spec = {'.name':['Sepal.Length','Petal.Length',
        ...                  'Sepal.Width','Petal.Width'],
        ...         '.value':['Length','Length','Width','Width'],
        ...         'part':['Sepal','Petal','Sepal','Petal']}
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
        >>> df.pipe(pivot_longer_spec,spec=spec)
        shape: (4, 4)
        ┌───────────┬────────┬───────┬───────┐
        │ Species   ┆ Length ┆ Width ┆ part  │
        │ ---       ┆ ---    ┆ ---   ┆ ---   │
        │ str       ┆ f64    ┆ f64   ┆ str   │
        ╞═══════════╪════════╪═══════╪═══════╡
        │ setosa    ┆ 5.1    ┆ 3.5   ┆ Sepal │
        │ virginica ┆ 5.9    ┆ 3.0   ┆ Sepal │
        │ setosa    ┆ 1.4    ┆ 0.2   ┆ Petal │
        │ virginica ┆ 5.1    ┆ 1.8   ┆ Petal │
        └───────────┴────────┴───────┴───────┘

    Args:
        df: The source DataFrame to unpivot.
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

    Raises:
        KeyError: If `.name` or `.value` is missing from the spec's columns.
        ValueError: If the labels in `spec['.name']` is not unique.

    Returns:
        A polars DataFrame/LazyFrame.
    """
    check("spec", spec, [pl.DataFrame])
    if ".name" not in spec.columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.name` column."
        )
    if ".value" not in spec.columns:
        raise KeyError(
            "Kindly ensure the spec DataFrame has a `.value` column."
        )
    if spec.select(pl.col(".name").is_duplicated().any()).item():
        raise ValueError("The labels in the `.name` column should be unique.")

    exclude = set(df.columns).intersection(spec.columns)
    if exclude:
        raise ValueError(
            f"Labels {*exclude, } in the spec dataframe already exist "
            "as column labels in the source dataframe. "
            "Kindly ensure the spec DataFrame's columns "
            "are not present in the source DataFrame."
        )

    if spec.columns[:2] != [".name", ".value"]:
        raise ValueError(
            "The first two columns of the spec DataFrame "
            "should be '.name' and '.value', "
            "with '.name' coming before '.value'."
        )

    return _pivot_longer_dot_value(
        df=df,
        spec=spec,
    )


@register_lazyframe_method
@register_dataframe_method
def pivot_longer(
    df,
    index: ColumnNameOrSelector = None,
    column_names: ColumnNameOrSelector = None,
    names_to: list | tuple | str = "variable",
    values_to: str = "value",
    names_sep: str = None,
    names_pattern: str = None,
    names_transform: pl.Expr = None,
) -> pl.DataFrame:
    """
    Unpivots a DataFrame from *wide* to *long* format.

    It is modeled after the `pivot_longer` function in R's tidyr package,
    and also takes inspiration from the `melt` function in R's data.table package.

    This function is useful to massage a DataFrame into a format where
    one or more columns are considered measured variables, and all other
    columns are considered as identifier variables.

    All measured variables are *unpivoted* (and typically duplicated) along the
    row axis.

    For more granular control on the unpivoting, have a look at
    `pivot_longer_spec`.

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

        Replicate polars' [melt](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.melt.html#polars-dataframe-melt):
        >>> df.pivot_longer(index = 'Species')
        shape: (8, 3)
        ┌───────────┬──────────────┬───────┐
        │ Species   ┆ variable     ┆ value │
        │ ---       ┆ ---          ┆ ---   │
        │ str       ┆ str          ┆ f64   │
        ╞═══════════╪══════════════╪═══════╡
        │ setosa    ┆ Sepal.Length ┆ 5.1   │
        │ virginica ┆ Sepal.Length ┆ 5.9   │
        │ setosa    ┆ Sepal.Width  ┆ 3.5   │
        │ virginica ┆ Sepal.Width  ┆ 3.0   │
        │ setosa    ┆ Petal.Length ┆ 1.4   │
        │ virginica ┆ Petal.Length ┆ 5.1   │
        │ setosa    ┆ Petal.Width  ┆ 0.2   │
        │ virginica ┆ Petal.Width  ┆ 1.8   │
        └───────────┴──────────────┴───────┘

        Split the column labels into individual columns:
        >>> df.pivot_longer(
        ...     index = 'Species',
        ...     names_to = ('part', 'dimension'),
        ...     names_sep = '.',
        ... ).select('Species','part','dimension','value')
        shape: (8, 4)
        ┌───────────┬───────┬───────────┬───────┐
        │ Species   ┆ part  ┆ dimension ┆ value │
        │ ---       ┆ ---   ┆ ---       ┆ ---   │
        │ str       ┆ str   ┆ str       ┆ f64   │
        ╞═══════════╪═══════╪═══════════╪═══════╡
        │ setosa    ┆ Sepal ┆ Length    ┆ 5.1   │
        │ virginica ┆ Sepal ┆ Length    ┆ 5.9   │
        │ setosa    ┆ Sepal ┆ Width     ┆ 3.5   │
        │ virginica ┆ Sepal ┆ Width     ┆ 3.0   │
        │ setosa    ┆ Petal ┆ Length    ┆ 1.4   │
        │ virginica ┆ Petal ┆ Length    ┆ 5.1   │
        │ setosa    ┆ Petal ┆ Width     ┆ 0.2   │
        │ virginica ┆ Petal ┆ Width     ┆ 1.8   │
        └───────────┴───────┴───────────┴───────┘

        Retain parts of the column names as headers:
        >>> df.pivot_longer(
        ...     index = 'Species',
        ...     names_to = ('part', '.value'),
        ...     names_sep = '.',
        ... ).select('Species','part','Length','Width')
        shape: (4, 4)
        ┌───────────┬───────┬────────┬───────┐
        │ Species   ┆ part  ┆ Length ┆ Width │
        │ ---       ┆ ---   ┆ ---    ┆ ---   │
        │ str       ┆ str   ┆ f64    ┆ f64   │
        ╞═══════════╪═══════╪════════╪═══════╡
        │ setosa    ┆ Sepal ┆ 5.1    ┆ 3.5   │
        │ virginica ┆ Sepal ┆ 5.9    ┆ 3.0   │
        │ setosa    ┆ Petal ┆ 1.4    ┆ 0.2   │
        │ virginica ┆ Petal ┆ 5.1    ┆ 1.8   │
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
        ...     index = 'id',
        ...     names_to = ('diagnosis', 'gender', 'age'),
        ...     names_pattern = r"new_?(.+)_(.)([0-9]+)",
        ... ).select('id','diagnosis','gender','age','value').sort(by=pl.all())
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
        ...     index = "id",
        ...     names_pattern=r"new_?(.+)_(.)([0-9]+)",
        ...     names_to=("diagnosis", "gender", "age"),
        ...     names_transform=pl.col('age').cast(pl.Int32),
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
        ... ).select('unit','time','x_mean','y_mean').sort(by=pl.all())
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
            It takes the same
            specification as polars' `str.extract_groups` method.
        names_transform: Use this option to change the types of columns that
            have been transformed to rows.
            This does not applies to the values' columns.
            Accepts a polars expression or a list of polars expressions.
            Applicable only if one of names_sep
            or names_pattern is provided.

    Returns:
        A polars DataFrame that has been unpivoted from wide to long
            format.
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
    names_to: list | tuple | str,
    values_to: str,
    names_sep: str,
    names_pattern: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Unpivots a DataFrame/LazyFrame from wide to long form.
    """

    (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        names_transform,
    ) = _data_checks_pivot_longer(
        df=df,
        index=index,
        column_names=column_names,
        names_to=names_to,
        values_to=values_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
        names_transform=names_transform,
    )

    if not column_names:
        return df

    if all((names_pattern is None, names_sep is None)):
        return df.melt(
            id_vars=index,
            value_vars=column_names,
            variable_name=names_to,
            value_name=values_to,
        )

    df = df.select(pl.col(index), pl.col(column_names))
    if isinstance(names_to, str):
        names_to = [names_to]

    spec = _pivot_longer_create_spec(
        column_names=column_names,
        names_to=names_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
        values_to=values_to,
        names_transform=names_transform,
    )

    return _pivot_longer_dot_value(df=df, spec=spec)


def _pivot_longer_create_spec(
    column_names: Iterable,
    names_to: Iterable,
    names_sep: str | None,
    names_pattern: str | None,
    values_to: str,
    names_transform: pl.Expr,
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
            .list.to_struct(n_field_strategy="max_width")
            .alias("extract")
        )
    else:
        expression = (
            pl.col(".name")
            .str.extract_groups(pattern=names_pattern)
            .alias("extract")
        )
    spec = spec.with_columns(expression)
    len_fields = len(spec.get_column("extract").struct.fields)
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
            spec.unnest(columns="extract")
            .filter(expression)
            .get_column(".name")
        )
        if null_check.len():
            column_name = null_check.gather(0).item()
            raise ValueError(
                f"Column label '{column_name}' "
                "could not be matched with any of the groups "
                "in the provided regex. Kindly provide a regular expression "
                "(with the correct groups) that matches all labels in the columns."
            )
    if names_to.count(".value") < 2:
        expression = pl.col("extract").struct.rename_fields(names=names_to)
        spec = spec.with_columns(expression).unnest(columns="extract")
    else:
        spec = _squash_multiple_dot_value(spec=spec, names_to=names_to)
    if ".value" not in names_to:
        expression = pl.lit(value=values_to).alias(".value")
        spec = spec.with_columns(expression)

    spec = spec.select(
        pl.col([".name", ".value"]), pl.exclude([".name", ".value"])
    )
    if names_transform is not None:
        spec = spec.with_columns(names_transform)
    return spec


def _pivot_longer_dot_value(
    df: pl.DataFrame | pl.LazyFrame, spec: pl.DataFrame
) -> pl.DataFrame | pl.LazyFrame:
    """
    Reshape DataFrame to long form based on metadata in `spec`.
    """
    index = [column for column in df.columns if column not in spec[".name"]]
    not_dot_value = [
        column for column in spec.columns if column not in {".name", ".value"}
    ]
    idx = "".join(spec.columns)
    if not_dot_value:
        # assign a number to each group (grouped by not_dot_value)
        expression = pl.first(idx).over(not_dot_value).rank("dense").sub(1)
        spec = spec.with_row_index(name=idx).with_columns(expression)
    else:
        # use a cumulative count to properly pair the columns
        # grouped by .value
        expression = pl.cum_count(".value").over(".value").alias(idx)
        spec = spec.with_columns(expression)
    mapping = defaultdict(list)
    for position, column_name, replacement_name in zip(
        spec.get_column(name=idx),
        spec.get_column(name=".name"),
        spec.get_column(name=".value"),
    ):
        expression = pl.col(column_name).alias(replacement_name)
        mapping[position].append(expression)

    mapping = (
        (
            [
                *index,
                *columns_to_select,
            ],
            pl.lit(position, dtype=pl.UInt32).alias(idx),
        )
        for position, columns_to_select in mapping.items()
    )
    df = [
        df.select(columns_to_select).with_columns(position)
        for columns_to_select, position in mapping
    ]
    # rechunking can be expensive;
    # however subsequent operations are faster
    # since data is contiguous in memory
    df = pl.concat(df, how="diagonal_relaxed", rechunk=True)
    expression = pl.cum_count(".value").over(".value").eq(1)
    dot_value = spec.filter(expression).select(".value")
    columns_to_select = [*index, *dot_value.to_series(0)]
    if not_dot_value:
        if isinstance(df, pl.LazyFrame):
            ranges = df.select(idx).collect().get_column(idx)
        else:
            ranges = df.get_column(idx)
        spec = spec.select(pl.struct(not_dot_value))
        _value = spec.columns[0]
        expression = pl.cum_count(_value).over(_value).eq(1)
        # using a gather approach, instead of a join
        # offers more performance - not sure why
        # maybe in the join there is another rechunking?
        spec = spec.filter(expression).select(pl.col(_value).gather(ranges))
        df = df.with_columns(spec).unnest(_value)
        columns_to_select.extend(not_dot_value)
    return df.select(columns_to_select)


def _squash_multiple_dot_value(
    spec: pl.DataFrame, names_to: Iterable
) -> pl.DataFrame:
    """
    Combine multiple .values into a single .value column
    """
    extract = spec.get_column("extract")
    fields = extract.struct.fields
    dot_value = [
        field for field, label in zip(fields, names_to) if label == ".value"
    ]
    dot_value = pl.concat_str(dot_value).alias(".value")
    not_dot_value = [
        pl.col(field).alias(label)
        for field, label in zip(fields, names_to)
        if label != ".value"
    ]
    select_expr = [".name", dot_value]
    if not_dot_value:
        select_expr.extend(not_dot_value)

    return spec.unnest("extract").select(select_expr)


def _data_checks_pivot_longer(
    df,
    index,
    column_names,
    names_to,
    values_to,
    names_sep,
    names_pattern,
    names_transform,
) -> tuple:
    """
    This function majorly does type checks on the passed arguments.

    This function is executed before proceeding to the computation phase.

    Type annotations are not provided because this function is where type
    checking happens.
    """

    def _check_type(arg_name: str, arg_value: Any):
        """
        Raise if argument is not a valid type
        """

        def _check_type_single(entry):
            if (
                not isinstance(entry, str)
                and not cs.is_selector(entry)
                and not isinstance(entry, pl.Expr)
            ):
                raise TypeError(
                    f"The argument passed to the {arg_name} parameter "
                    "should be a type that is supported in the polars' "
                    "select function."
                )

        if isinstance(arg_value, (list, tuple)):
            for entry in arg_value:
                _check_type_single(entry=entry)
        else:
            _check_type_single(entry=arg_value)

    if (index is None) and (column_names is None):
        column_names = df.columns
        index = []
    elif (index is not None) and (column_names is not None):
        _check_type(arg_name="index", arg_value=index)
        index = df.select(index).columns
        _check_type(arg_name="column_names", arg_value=column_names)
        column_names = df.select(column_names).columns

    elif (index is None) and (column_names is not None):
        _check_type(arg_name="column_names", arg_value=column_names)
        column_names = df.select(column_names).columns
        index = df.select(pl.exclude(column_names)).columns

    elif (index is not None) and (column_names is None):
        _check_type(arg_name="index", arg_value=index)
        index = df.select(index).columns
        column_names = df.select(pl.exclude(index)).columns

    check("names_to", names_to, [list, tuple, str])
    if isinstance(names_to, (list, tuple)):
        uniques = set()
        for word in names_to:
            check(f"'{word}' in names_to", word, [str])
            if (word in uniques) and (word != ".value"):
                raise ValueError(f"'{word}' is duplicated in names_to.")
            uniques.add(word)

    if names_sep and names_pattern:
        raise ValueError(
            "Only one of names_pattern or names_sep should be provided."
        )

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    if names_pattern is not None:
        check("names_pattern", names_pattern, [str])

    check("values_to", values_to, [str])

    return (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
        names_transform,
    )
