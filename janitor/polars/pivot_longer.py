"""pivot_longer implementation for polars."""

from __future__ import annotations

from janitor.utils import check, import_message

try:
    import polars as pl
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
        >>> df.pipe(pivot_longer_spec,spec=spec).sort*by=pl.all())
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
    index = [
        label for label in df.columns if label not in spec.get_column(".name")
    ]
    others = [
        label for label in spec.columns if label not in {".name", ".value"}
    ]
    variable_name = "".join(df.columns + spec.columns)
    variable_name = f"{variable_name}_"
    if others:
        dot_value_only = False
        expression = pl.struct(others).alias(variable_name)
        spec = spec.select(".name", ".value", expression)
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
        names_transform=None,
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
        return df.melt(
            id_vars=index,
            value_vars=column_names,
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

    variable_name = "".join(df.columns)
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

    if {".name", ".value"}.symmetric_difference(spec.columns):
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
        if ".value" not in names_to:
            return spec.get_column(variable_name)
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
    fields = _spec.columns

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
    # than running on the actual data
    outcome = (
        df.select(pl.all().implode())
        .melt(
            id_vars=index,
            value_vars=column_names,
            variable_name=variable_name,
            value_name=values_to,
        )
        .with_columns(spec)
    )

    outcome = outcome.unnest(variable_name)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)
    columns = [name for name in outcome.columns if name not in names_to]
    outcome = outcome.explode(columns=columns)
    return outcome


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
    if names_sep and .value in names_to.
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
    outcome = (
        df.select(expressions)
        .select(pl.all().implode())
        .melt(id_vars=index, variable_name=variable_name, value_name=".value")
        .with_columns(spec)
    )

    if dot_value_only:
        columns = [
            label for label in outcome.columns if label != variable_name
        ]
        outcome = outcome.explode(columns).unnest(".value")
        outcome = outcome.select(pl.exclude(variable_name))
        return outcome
    outcome = outcome.unnest(variable_name)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)
    columns = [
        label for label in outcome.columns if label not in spec.struct.fields
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
        column_names = df.columns
        index = []
    elif (index is None) and (column_names is not None):
        column_names = df.select(column_names).columns
        index = df.select(pl.exclude(column_names)).columns
    elif (index is not None) and (column_names is None):
        index = df.select(index).columns
        column_names = df.select(pl.exclude(index)).columns
    else:
        index = df.select(index).columns
        column_names = df.select(column_names).columns

    return (
        df,
        index,
        column_names,
        names_to,
        values_to,
        names_sep,
        names_pattern,
    )
