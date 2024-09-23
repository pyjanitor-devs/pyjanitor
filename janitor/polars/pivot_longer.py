"""pivot_longer implementation for polars."""

from __future__ import annotations

from janitor.utils import check, import_message

from .polars_flavor import register_dataframe_method, register_lazyframe_method

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
        >>> df.pipe(pivot_longer_spec,spec=spec).sort(by=pl.all())
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
    df_columns = df.collect_schema().names()
    exclude = set(df_columns).intersection(spec_columns)
    if exclude:
        raise ValueError(
            f"Labels {*exclude, } in the spec dataframe already exist "
            "as column labels in the source dataframe. "
            "Kindly ensure the spec DataFrame's columns "
            "are not present in the source DataFrame."
        )

    index = [
        label for label in df_columns if label not in spec.get_column(".name")
    ]
    others = [
        label for label in spec_columns if label not in {".name", ".value"}
    ]
    if others:
        if (len(others) == 1) & (
            spec.get_column(others[0]).dtype == pl.String
        ):
            # shortcut that avoids the implode/explode approach - and is faster
            # if the requirements are met
            # inspired by https://github.com/pola-rs/polars/pull/18519#issue-2500860927
            return _pivot_longer_dot_value_string(
                df=df,
                index=index,
                spec=spec,
                variable_name=others[0],
            )
        variable_name = "".join(df_columns + spec_columns)
        variable_name = f"{variable_name}_"
        dot_value_only = False
        expression = pl.struct(others).alias(variable_name)
        spec = spec.select(".name", ".value", expression)
    else:
        variable_name = "".join(df_columns + spec_columns)
        variable_name = f"{variable_name}_"
        dot_value_only = True
        expression = pl.cum_count(".value").over(".value").alias(variable_name)
        spec = spec.with_columns(expression)
    return _pivot_longer_dot_value(
        df=df,
        index=index,
        spec=spec,
        variable_name=variable_name,
        dot_value_only=dot_value_only,
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
        >>> df.pivot_longer(index = 'Species').sort(by=pl.all())
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
        ...     index = 'Species',
        ...     names_to = ('part', 'dimension'),
        ...     names_sep = '.',
        ... ).select('Species','part','dimension','value').sort(by=pl.all())
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
        ...     index = 'Species',
        ...     names_to = ('part', '.value'),
        ...     names_sep = '.',
        ... ).select('Species','part','Length','Width').sort(by=pl.all())
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

    (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
    ) = _data_checks_pivot_longer(
        df=df,
        index=index,
        column_names=column_names,
        names_to=names_to,
        values_to=values_to,
        names_sep=names_sep,
        names_pattern=names_pattern,
    )

    variable_name = "".join(df.collect_schema().names())
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
    if {".name", ".value"}.symmetric_difference(spec.collect_schema().names()):
        # shortcut that avoids the implode/explode approach - and is faster
        # if the requirements are met
        # inspired by https://github.com/pola-rs/polars/pull/18519#issue-2500860927
        data = spec.get_column(variable_name)
        others = data.struct.fields
        data = data.struct[others[0]]
        if (
            (len(others) == 1)
            & (data.dtype == pl.String)
            & (names_transform is None)
        ):
            spec = spec.unnest(variable_name)
            return _pivot_longer_dot_value_string(
                df=df,
                index=index,
                spec=spec,
                variable_name=others[0],
            )
        dot_value_only = False
    else:
        dot_value_only = True
        expression = pl.cum_count(".value").over(".value").alias(variable_name)
        spec = spec.with_columns(expression)

    return _pivot_longer_dot_value(
        df=df,
        index=index,
        spec=spec,
        variable_name=variable_name,
        dot_value_only=dot_value_only,
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
            .list.to_struct(n_field_strategy="max_width")
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
            spec.unnest(columns=variable_name)
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

    if ".value" not in names_to:
        spec = spec.get_column(variable_name)
        spec = spec.struct.rename_fields(names=names_to)
        return spec
    if names_to.count(".value") == 1:
        spec = spec.with_columns(
            pl.col(variable_name).struct.rename_fields(names=names_to)
        )
        not_dot_value = [name for name in names_to if name != ".value"]
        spec = spec.unnest(variable_name)
        if not_dot_value:
            return spec.select(
                ".name",
                ".value",
                pl.struct(not_dot_value).alias(variable_name),
            )
        return spec.select(".name", ".value")
    _spec = spec.get_column(variable_name)
    _spec = _spec.struct.unnest()
    fields = _spec.collect_schema().names()

    if len(set(names_to)) == 1:
        expression = pl.concat_str(fields).alias(".value")
        dot_value = _spec.select(expression)
        dot_value = dot_value.to_series(0)
        return spec.select(".name", dot_value)
    dot_value = [
        field for field, label in zip(fields, names_to) if label == ".value"
    ]
    dot_value = pl.concat_str(dot_value).alias(".value")
    not_dot_value = [
        pl.col(field).alias(label)
        for field, label in zip(fields, names_to)
        if label != ".value"
    ]
    not_dot_value = pl.struct(not_dot_value).alias(variable_name)
    return _spec.select(spec.get_column(".name"), not_dot_value, dot_value)


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
    columns = [
        name
        for name in outcome.collect_schema().names()
        if name not in names_to
    ]
    outcome = outcome.explode(columns=columns)
    return outcome


def _pivot_longer_dot_value_string(
    df: pl.DataFrame | pl.LazyFrame,
    spec: pl.DataFrame,
    index: ColumnNameOrSelector,
    variable_name: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    fastpath for .value - does not require implode/explode approach.
    """
    spec = spec.group_by(variable_name)
    spec = spec.agg(pl.all())
    expressions = []
    for names, fields, header in zip(
        spec.get_column(".name").to_list(),
        spec.get_column(".value").to_list(),
        spec.get_column(variable_name).to_list(),
    ):
        expression = pl.struct(names).struct.rename_fields(names=fields)
        expression = expression.alias(header)
        expressions.append(expression)
    expressions = [*index, *expressions]
    df = (
        df.select(expressions)
        .unpivot(index=index, variable_name=variable_name, value_name=".value")
        .unnest(".value")
    )
    return df


def _pivot_longer_dot_value(
    df: pl.DataFrame | pl.LazyFrame,
    spec: pl.DataFrame,
    index: ColumnNameOrSelector,
    variable_name: str,
    dot_value_only: bool,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    flip polars Frame to long form,
    if .value in names_to.
    """
    spec = spec.group_by(variable_name)
    spec = spec.agg(pl.all())
    expressions = []
    for names, fields in zip(
        spec.get_column(".name").to_list(),
        spec.get_column(".value").to_list(),
    ):
        expression = pl.struct(names).struct.rename_fields(names=fields)
        expressions.append(expression)
    expressions = [*index, *expressions]
    spec = spec.get_column(variable_name)
    if dot_value_only:
        outcome = (
            df.select(expressions)
            .unpivot(
                index=index, variable_name=variable_name, value_name=".value"
            )
            .select(pl.exclude(variable_name))
            .unnest(".value")
        )
        return outcome

    outcome = (
        df.select(expressions)
        .select(pl.all().implode())
        .unpivot(index=index, variable_name=variable_name, value_name=".value")
        .with_columns(spec)
    )

    outcome = outcome.unnest(variable_name)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)
    columns = [
        label
        for label in outcome.collect_schema().names()
        if label not in spec.struct.fields
    ]
    outcome = outcome.explode(columns)
    outcome = outcome.unnest(".value")

    return outcome


def _data_checks_pivot_longer(
    df,
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
        raise ValueError(
            "Only one of names_pattern or names_sep should be provided."
        )

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    else:
        check("names_pattern", names_pattern, [str])

    check("values_to", values_to, [str])

    if (index is None) and (column_names is None):
        column_names = df.collect_schema().names()
        index = []
    elif (index is None) and (column_names is not None):
        column_names = df.select(column_names).collect_schema().names()
        index = df.select(pl.exclude(column_names)).collect_schema().names()
    elif (index is not None) and (column_names is None):
        index = df.select(index).collect_schema().names()
        column_names = df.select(pl.exclude(index)).collect_schema().names()
    else:
        index = df.select(index).collect_schema().names()
        column_names = df.select(column_names).collect_schema().names()

    return (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
    )
