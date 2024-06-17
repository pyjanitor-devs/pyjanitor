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

    df_columns = pl.DataFrame({".name": df.columns})

    spec = df_columns.join(spec, on=".name", how="left")
    spec = spec.select(pl.exclude(".name"))
    if len(spec.columns) == 1:
        return _pivot_longer_dot_value_only(
            df=df,
            outcome=spec,
        )
    return


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

    if names_sep and (".value" not in names_to):
        return _pivot_longer_names_sep_no_dot_value(
            df=df,
            index=index,
            column_names=column_names,
            names_to=names_to,
            values_to=values_to,
            names_sep=names_sep,
            names_transform=names_transform,
        )
    if names_pattern and (".value" not in names_to):
        return _pivot_longer_names_pattern_no_dot_value(
            df=df,
            index=index,
            column_names=column_names,
            names_to=names_to,
            values_to=values_to,
            names_pattern=names_pattern,
            names_transform=names_transform,
        )
    if names_sep:
        return _pivot_longer_names_sep_dot_value(
            df=df,
            index=index,
            column_names=column_names,
            names_to=names_to,
            names_sep=names_sep,
            names_transform=names_transform,
        )
    return _pivot_longer_names_pattern_dot_value(
        df=df,
        index=index,
        column_names=column_names,
        names_to=names_to,
        names_pattern=names_pattern,
        names_transform=names_transform,
    )


def _pivot_longer_names_sep_no_dot_value(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple,
    values_to: str,
    names_sep: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    flip polars Frame to long form,
    if names_sep and no .value in names_to.
    """
    variable_name = "".join(df.columns)
    # the implode approach is used here
    # for efficiency
    # it is much faster to extract the relevant strings
    # on a smaller set and then explode
    # than to melt into the full data and then extract
    outcome = (
        df.select(pl.all().implode())
        .melt(
            id_vars=index,
            value_vars=column_names,
            variable_name=variable_name,
            value_name=values_to,
        )
        .with_columns(
            pl.col(variable_name)
            .str.split(by=names_sep)
            .list.to_struct(n_field_strategy="max_width"),
        )
    )
    if isinstance(df, pl.LazyFrame):
        extract = outcome.select(variable_name).collect().to_series(0)
    else:
        extract = outcome.get_column(variable_name)

    len_names_to = len(names_to)

    len_fields = len(extract.struct.fields)

    if len_names_to != len_fields:
        raise ValueError(
            f"The length of names_to does not match "
            "the number of fields extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of fields extracted is "
            f"{len_fields}."
        )

    expression = pl.col(variable_name).struct.rename_fields(names=names_to)
    outcome = outcome.with_columns(expression)

    if isinstance(df, pl.LazyFrame):
        # to ensure the unnested columns are available downstream
        # in a LazyFrame, a workaround is to reintroduce
        # the variable_name column via with_columns
        series = outcome.select(variable_name).collect()
        outcome = outcome.with_columns(series)

    outcome = outcome.unnest(variable_name)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)

    columns = [name for name in outcome.columns if name not in names_to]
    outcome = outcome.explode(columns=columns)
    return outcome


def _pivot_longer_names_pattern_no_dot_value(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple,
    values_to: str,
    names_pattern: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    flip polars Frame to long form,
    if names_pattern and no .value in names_to.
    """
    variable_name = "".join(df.columns)
    outcome = df.select(pl.all().implode())
    outcome = outcome.melt(
        id_vars=index,
        value_vars=column_names,
        variable_name=variable_name,
        value_name=values_to,
    )
    alias = outcome.columns
    alias = "".join(alias)
    alias = f"{alias}_"
    expression = pl.col(variable_name)
    expression = expression.str.extract_groups(pattern=names_pattern)
    expression = expression.alias(alias)
    outcome = outcome.with_columns(expression)
    extract = outcome.select(alias, variable_name)
    is_a_lazyframe = isinstance(df, pl.LazyFrame)
    if is_a_lazyframe:
        extract = extract.collect()
    len_fields = len(extract.get_column(alias).struct.fields)
    len_names_to = len(names_to)

    if len_names_to != len_fields:
        raise ValueError(
            f"The length of names_to does not match "
            "the number of fields extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of fields extracted is "
            f"{len_fields}."
        )
    expression = pl.exclude(variable_name).is_null().any()
    expression = pl.any_horizontal(expression)
    null_check = (
        extract.unnest(alias).filter(expression).get_column(variable_name)
    )
    if null_check.len():
        column_name = null_check.gather(0).item()
        raise ValueError(
            f"Column label '{column_name}' "
            "could not be matched with any of the groups "
            "in the provided regex. Kindly provide a regular expression "
            "(with the correct groups) that matches all labels in the columns."
        )

    expression = pl.col(alias).struct.rename_fields(names=names_to)
    outcome = outcome.with_columns(expression)

    outcome = outcome.select(pl.exclude(variable_name))
    if is_a_lazyframe:
        series = outcome.select(alias).collect()
        outcome = outcome.with_columns(series)
    outcome = outcome.unnest(alias)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)

    columns = [name for name in outcome.columns if name not in names_to]
    outcome = outcome.explode(columns=columns)
    return outcome


def _pivot_longer_names_sep_dot_value(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple,
    names_sep: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    flip polars Frame to long form,
    if names_sep and .value in names_to.
    """

    variable_name = "".join(df.columns)
    value_name = f"{''.join(df.columns)}_"
    outcome = _names_sep_reshape(
        df=df,
        index=index,
        variable_name=variable_name,
        column_names=column_names,
        names_to=names_to,
        value_name=value_name,
        names_sep=names_sep,
        names_transform=names_transform,
    )

    others = [name for name in names_to if name != ".value"]
    if others:
        return _pivot_longer_dot_value_others(
            df=df,
            outcome=outcome,
            value_name=value_name,
            others=others,
        )
    return _pivot_longer_dot_value_only(
        df=df,
        outcome=outcome,
        variable_name=variable_name,
        value_name=value_name,
    )


def _pivot_longer_names_pattern_dot_value(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple,
    names_pattern: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    """
    flip polars Frame to long form,
    if names_pattern and .value in names_to.
    """

    variable_name = "".join(df.columns)
    value_name = f"{''.join(df.columns)}_"
    outcome = _names_pattern_reshape(
        df=df,
        index=index,
        variable_name=variable_name,
        column_names=column_names,
        names_to=names_to,
        value_name=value_name,
        names_pattern=names_pattern,
        names_transform=names_transform,
    )

    others = [name for name in names_to if name != ".value"]
    if others:
        return _pivot_longer_dot_value_others(
            df=df,
            outcome=outcome,
            value_name=value_name,
            others=others,
        )
    return _pivot_longer_dot_value_only(
        df=df,
        outcome=outcome,
        value_name=value_name,
    )


def _pivot_longer_dot_value_only(
    df: pl.DataFrame | pl.LazyFrame,
    outcome: pl.DataFrame | pl.LazyFrame,
    value_name: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Pivot to long form if '.value' only
    """
    # for .value reshaping, each sub Frame
    # should have the same columns
    # the code below creates a DataFrame of unique values
    # (here we use cumcount to ensure uniqueness)
    alias = "".join(outcome.columns)
    expression = pl.cum_count(".value").over(".value").alias(alias)
    outcome = outcome.with_columns(expression)
    expr1 = pl.col(".value").unique().sort().implode()
    expr2 = pl.col(alias).unique().sort().implode()
    uniqs = outcome.select(expr1, expr2)
    uniqs = uniqs.explode(".value")
    uniqs = uniqs.explode(alias)
    # uniqs is then joined to `outcome`
    # to ensure all groups have the labels in .value
    # this may introduce nulls if not all groups
    # shared the same labels in .value prior to the join -
    # the null check below handles that
    outcome = uniqs.join(outcome, on=uniqs.columns, how="left")
    # patch to deal with nulls
    expression = pl.col(value_name).is_null().any()
    null_check = outcome.select(expression)
    is_a_lazyframe = isinstance(df, pl.LazyFrame)
    if is_a_lazyframe:
        null_check = null_check.collect()
    null_check = null_check.item()
    if null_check:
        variable_name = "".join(outcome.columns)
        expr1 = pl.lit(None).alias(variable_name)
        expr2 = pl.implode(variable_name)
        nulls = df.with_columns(expr1).select(expr2)
        if is_a_lazyframe:
            nulls = nulls.collect()
        nulls = nulls.to_series(0)
        expression = pl.col(value_name).fill_null(nulls)
        outcome = outcome.with_columns(expression)

    index = [
        label
        for label in outcome.columns
        if label not in {alias, value_name, ".value"}
    ]
    # due to the implodes, index, if present is repeated
    # however, we need index to be unique,
    # hence the selection of only the first entry
    # from the duplicated(repeated) index values in the list
    agg_ = [pl.first(index), pl.col(".value"), pl.col(value_name)]
    outcome = outcome.group_by(alias, maintain_order=True).agg(agg_)
    # since all groups have the same labels in '.value'
    # and order is assured in the group_by operation
    # we just grab only the first row
    # which will serve as headers of the new columns with values
    fields = outcome.select(pl.first(".value"))
    if is_a_lazyframe:
        fields = fields.collect()
    fields = fields.item().to_list()

    outcome = outcome.select(pl.exclude(".value"))
    expression = pl.col(value_name).list.to_struct(
        n_field_strategy="max_width", fields=fields
    )
    outcome = outcome.with_columns(expression)
    if is_a_lazyframe:
        # to ensure the unnested columns are available downstream
        # in a LazyFrame, a workaround is to reintroduce
        # the value_name column via with_columns
        series = outcome.select(value_name).collect()
        outcome = outcome.with_columns(series)
    outcome = (
        outcome.unnest(value_name)
        .explode([*index, *fields])
        .select(pl.exclude(alias))
    )
    return outcome


def _pivot_longer_dot_value_others(
    df: pl.DataFrame | pl.LazyFrame,
    outcome: pl.DataFrame | pl.LazyFrame,
    value_name: str,
    others: list,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Pivot to long form if '.value'
    and `others`.
    """
    # logic breakdown is similar to _pivot_longer_dot_value_only
    expr1 = pl.struct(others).unique().sort().implode()
    expr2 = pl.col(".value").unique().sort().implode()
    uniqs = outcome.select(expr1, expr2)
    uniqs = uniqs.explode(others[0])
    uniqs = uniqs.explode(".value")
    uniqs = uniqs.unnest(others[0])

    outcome = uniqs.join(outcome, on=uniqs.columns, how="left")

    expression = pl.col(value_name).is_null().any()
    null_check = outcome.select(expression)
    is_a_lazyframe = isinstance(df, pl.LazyFrame)
    if is_a_lazyframe:
        null_check = null_check.collect()
    null_check = null_check.item()
    if null_check:
        variable_name = "".join(outcome.columns)
        expr1 = pl.lit(None).alias(variable_name)
        expr2 = pl.implode(variable_name)
        nulls = df.with_columns(expr1).select(expr2)
        if is_a_lazyframe:
            nulls = nulls.collect()
        nulls = nulls.to_series(0)
        expression = pl.col(value_name).fill_null(nulls)
        outcome = outcome.with_columns(expression)

    index = [
        label
        for label in outcome.columns
        if label not in {*others, value_name, ".value"}
    ]
    agg_ = [pl.first(index), pl.col(".value"), pl.col(value_name)]
    outcome = outcome.group_by(others, maintain_order=True).agg(agg_)

    fields = outcome.select(pl.first(".value"))
    if is_a_lazyframe:
        fields = fields.collect()
    fields = fields.item().to_list()

    outcome = outcome.select(pl.exclude(".value"))
    expression = pl.col(value_name).list.to_struct(
        n_field_strategy="max_width", fields=fields
    )

    outcome = outcome.with_columns(expression)
    if is_a_lazyframe:
        series = outcome.select(value_name).collect()
        outcome = outcome.with_columns(series)
    outcome = outcome.unnest(value_name).explode([*index, *fields])

    return outcome


def _names_sep_reshape(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple,
    variable_name: str,
    value_name: str,
    names_sep: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    # the implode approach is used here
    # for efficiency
    # it is much faster to extract the relevant strings
    # on a smaller set and then explode
    # than to melt into the full data and then extract
    outcome = (
        df.select(pl.all().implode())
        .melt(
            id_vars=index,
            value_vars=column_names,
            variable_name=variable_name,
            value_name=value_name,
        )
        .with_columns(
            pl.col(variable_name)
            .str.split(by=names_sep)
            .list.to_struct(n_field_strategy="max_width"),
        )
    )

    if isinstance(df, pl.LazyFrame):
        extract = outcome.select(variable_name).collect().to_series(0)
    else:
        extract = outcome.get_column(variable_name)

    len_names_to = len(names_to)

    len_fields = len(extract.struct.fields)

    if len_names_to != len_fields:
        raise ValueError(
            f"The length of names_to does not match "
            "the number of fields extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of fields extracted is "
            f"{len_fields}."
        )

    if names_to.count(".value") > 1:
        _fields = extract.struct.fields
        fields = [
            extract.struct.field(label)
            for label, name in zip(_fields, names_to)
            if name == ".value"
        ]
        _value = pl.concat_str(fields).alias(".value")
        fields = [
            extract.struct.field(label).alias(name)
            for label, name in zip(_fields, names_to)
            if name != ".value"
        ]
        fields.append(_value)
        extract = pl.struct(fields).alias(variable_name)
        outcome = outcome.with_columns(extract)
    else:
        expression = pl.col(variable_name).struct.rename_fields(names=names_to)
        outcome = outcome.with_columns(expression)
    if isinstance(df, pl.LazyFrame):
        # to ensure the unnested columns are available downstream
        # in a LazyFrame, a workaround is to reintroduce
        # the variable_name column via with_columns
        series = outcome.select(variable_name).collect()
        outcome = outcome.with_columns(series)
    outcome = outcome.unnest(variable_name)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)
    return outcome


def _names_pattern_reshape(
    df: pl.DataFrame | pl.LazyFrame,
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: list | tuple,
    variable_name: str,
    value_name: str,
    names_pattern: str,
    names_transform: pl.Expr,
) -> pl.DataFrame | pl.LazyFrame:
    outcome = df.select(pl.all().implode())
    outcome = outcome.melt(
        id_vars=index,
        value_vars=column_names,
        variable_name=variable_name,
        value_name=value_name,
    )
    alias = outcome.columns
    alias = "".join(alias)
    alias = f"{alias}_"
    outcome = outcome.with_columns(
        pl.col(variable_name)
        .str.extract_groups(pattern=names_pattern)
        .alias(alias)
    )
    extract = outcome.select(alias, variable_name)
    is_a_lazyframe = isinstance(df, pl.LazyFrame)
    if is_a_lazyframe:
        extract = extract.collect()
    len_fields = len(extract.get_column(alias).struct.fields)
    len_names_to = len(names_to)

    if len_names_to != len_fields:
        raise ValueError(
            f"The length of names_to does not match "
            "the number of fields extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of fields extracted is "
            f"{len_fields}."
        )
    expression = pl.exclude(variable_name).is_null().any()
    expression = pl.any_horizontal(expression)
    null_check = (
        extract.unnest(alias).filter(expression).get_column(variable_name)
    )
    if null_check.len():
        column_name = null_check.gather(0).item()
        raise ValueError(
            f"Column label '{column_name}' "
            "could not be matched with any of the groups "
            "in the provided regex. Kindly provide a regular expression "
            "(with the correct groups) that matches all labels in the columns."
        )

    if names_to.count(".value") > 1:
        extract = extract.get_column(alias)
        _fields = extract.struct.fields
        fields = [
            extract.struct.field(label)
            for label, name in zip(_fields, names_to)
            if name == ".value"
        ]
        _value = pl.concat_str(fields).alias(".value")
        fields = [
            extract.struct.field(label).alias(name)
            for label, name in zip(_fields, names_to)
            if name != ".value"
        ]
        fields.append(_value)
        extract = pl.struct(fields).alias(alias)
        outcome = outcome.with_columns(extract)
    else:
        expression = pl.col(alias).struct.rename_fields(names=names_to)
        outcome = outcome.with_columns(expression)

    outcome = outcome.select(pl.exclude(variable_name))
    if is_a_lazyframe:
        series = outcome.select(alias).collect()
        outcome = outcome.with_columns(series)
    outcome = outcome.unnest(alias)
    if names_transform is not None:
        outcome = outcome.with_columns(names_transform)
    return outcome
