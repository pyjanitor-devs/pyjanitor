"""pivot_longer implementation for polars."""

from collections import defaultdict
from itertools import chain
from typing import Any, Iterable, Optional, Union

from janitor.utils import check, import_message

try:
    import polars as pl
    import polars.selectors as cs
    from polars.datatypes.classes import DataTypeClass
    from polars.type_aliases import IntoExpr, PolarsDataType
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def _pivot_longer(
    df: pl.DataFrame,
    index: Union[IntoExpr, Iterable[IntoExpr], None],
    column_names: Union[IntoExpr, Iterable[IntoExpr], None],
    names_to: Optional[Union[list, str]],
    values_to: Optional[str],
    names_sep: Optional[Union[str, None]],
    names_pattern: Optional[Union[list, tuple, str, None]],
    names_transform: Optional[Union[PolarsDataType, dict]],
) -> pl.DataFrame:
    """
    Unpivots a DataFrame to long form.
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
            variable_name=names_to[0],
            value_name=values_to,
        )

    # the core idea is to do the transformation on the columns
    # before flipping into long form
    # typically less work is done this way
    # compared to flipping and then processing the columns

    if names_sep is not None:
        return _pivot_longer_names_sep(
            df=df,
            index=index,
            column_names=column_names,
            names_to=names_to,
            names_sep=names_sep,
            values_to=values_to,
            names_transform=names_transform,
        )

    if isinstance(names_pattern, str):
        return _pivot_longer_names_pattern_str(
            df=df,
            index=index,
            column_names=column_names,
            names_to=names_to,
            names_pattern=names_pattern,
            values_to=values_to,
            names_transform=names_transform,
        )
    if isinstance(values_to, (list, tuple)):
        return _pivot_longer_values_to_sequence(
            df=df,
            index=index,
            column_names=column_names,
            names_to=names_to,
            names_pattern=names_pattern,
            values_to=values_to,
            names_transform=names_transform,
        )

    return _pivot_longer_names_pattern_sequence(
        df=df,
        index=index,
        column_names=column_names,
        names_to=names_to,
        names_pattern=names_pattern,
    )


def _pivot_longer_names_sep(
    df: pl.DataFrame,
    index: Iterable,
    column_names: Iterable,
    names_to: Iterable,
    names_sep: str,
    values_to: str,
    names_transform: dict,
) -> pl.DataFrame:
    """
    This takes care of unpivoting scenarios where
    names_sep is provided.
    """
    columns = df.select(column_names).columns
    outcome = (
        pl.Series(columns)
        .str.split(by=names_sep)
        .list.to_struct(n_field_strategy="max_width")
    )
    if ".value" not in names_to:
        return _pivot_longer_no_dot_value(
            df=df,
            outcome=outcome,
            names_to=names_to,
            names_transform=names_transform,
            values_to=values_to,
            index=index,
            columns=columns,
        )

    if all(label == ".value" for label in names_to):
        return _pivot_longer_dot_value_only(
            df=df,
            names_to=names_to,
            columns=columns,
            index=index,
            outcome=outcome,
        )
    return _pivot_longer_dot_value(
        df=df,
        names_to=names_to,
        columns=columns,
        index=index,
        outcome=outcome,
        names_transform=names_transform,
    )


def _pivot_longer_names_pattern_str(
    df: pl.DataFrame,
    index: Iterable,
    column_names: Iterable,
    names_to: Iterable,
    names_pattern: str,
    values_to: str,
    names_transform: dict,
) -> pl.DataFrame:
    """
    This takes care of unpivoting scenarios where
    names_pattern is a string.
    """

    columns = df.select(column_names).columns
    outcome = pl.Series(columns).str.extract_groups(names_pattern)

    if ".value" not in names_to:
        return _pivot_longer_no_dot_value(
            df=df,
            outcome=outcome,
            names_to=names_to,
            names_transform=names_transform,
            values_to=values_to,
            index=index,
            columns=columns,
        )

    if all(label == ".value" for label in names_to):
        return _pivot_longer_dot_value_only(
            df=df,
            names_to=names_to,
            columns=columns,
            index=index,
            outcome=outcome,
        )
    return _pivot_longer_dot_value(
        df=df,
        names_to=names_to,
        columns=columns,
        index=index,
        outcome=outcome,
        names_transform=names_transform,
    )


def _pivot_longer_values_to_sequence(
    df: pl.DataFrame,
    index: Iterable,
    column_names: Iterable,
    names_to: Iterable,
    names_pattern: Iterable,
    values_to: Iterable,
    names_transform: dict,
) -> pl.DataFrame:
    """
    This takes care of unpivoting scenarios where
    values_to is a list/tuple.
    """
    expressions = [
        pl.col("cols").str.contains(pattern).alias(f"cols{num}")
        for num, pattern in enumerate(names_pattern)
    ]
    columns = df.select(column_names).columns
    outcome = pl.DataFrame({"cols": columns}).with_columns(expressions)
    booleans = outcome.select(pl.exclude("cols").any())
    for position in range(len(names_pattern)):
        if not booleans.to_series(position).item():
            raise ValueError(
                "No match was returned for the regex "
                f"at position {position} -> {names_pattern[position]}."
            )
    names_booleans = pl
    values_booleans = pl
    for boolean, repl_name, repl_value in zip(
        booleans.columns, names_to, values_to
    ):
        names_booleans = names_booleans.when(pl.col(boolean)).then(
            pl.lit(repl_name)
        )
        values_booleans = values_booleans.when(pl.col(boolean)).then(
            pl.lit(repl_value)
        )
    names_booleans = names_booleans.alias("value")
    values_booleans = values_booleans.alias(".value")
    filter_expr = pl.col(".value").is_not_null()
    cum_expr = pl.col(".value").cum_count().over(".value").sub(1).alias("idx")
    outcome = (
        outcome.select(names_booleans, values_booleans, pl.col("cols"))
        .filter(filter_expr)
        .with_columns(cum_expr)
    )
    mapping = defaultdict(list)
    for num, col_name, value_header, name_header in zip(
        outcome.get_column("idx"),
        outcome.get_column("cols"),
        outcome.get_column(".value"),
        outcome.get_column("value"),
    ):
        expression = (
            pl.col(col_name).alias(value_header),
            pl.lit(col_name, dtype=names_transform[name_header]).alias(
                name_header
            ),
        )
        mapping[num].append(expression)
    mapping = (zip(*entry) for entry in mapping.values())
    if index:
        mapping = (
            ((pl.col(index), *cols_to_select), cols_to_append)
            for cols_to_select, cols_to_append in mapping
        )
    contents = (
        df.select(cols_to_select).with_columns(cols_to_append)
        for cols_to_select, cols_to_append in mapping
    )
    columns_to_select = chain.from_iterable(zip(names_to, values_to))
    columns_to_select = [*index, *columns_to_select]
    return pl.concat(contents, how="diagonal_relaxed").select(
        columns_to_select
    )


def _pivot_longer_names_pattern_sequence(
    df: pl.DataFrame,
    index: Iterable,
    column_names: Iterable,
    names_to: Iterable,
    names_pattern: Iterable,
) -> pl.DataFrame:
    """
    This takes care of unpivoting scenarios where
    names_pattern is a list/tuple.
    """
    columns = df.select(column_names).columns
    outcome = pl.DataFrame({"cols": columns})
    expressions = [
        pl.col("cols").str.contains(pattern).alias(f"cols{num}")
        for num, pattern in enumerate(names_pattern)
    ]
    outcome = outcome.with_columns(expressions)
    booleans = outcome.select(pl.exclude("cols").any())
    for position in range(len(names_pattern)):
        if not booleans.to_series(position).item():
            raise ValueError(
                "No match was returned for the regex "
                f"at position {position} -> {names_pattern[position]}."
            )
    names_booleans = pl
    for boolean, repl_name in zip(booleans.columns, names_to):
        names_booleans = names_booleans.when(pl.col(boolean)).then(
            pl.lit(repl_name)
        )

    names_booleans = names_booleans.alias(".value")
    filter_expr = pl.col(".value").is_not_null()
    cum_expr = pl.col(".value").cum_count().over(".value").sub(1).alias("idx")
    outcome = (
        outcome.select(names_booleans, pl.col("cols"))
        .filter(filter_expr)
        .with_columns(cum_expr)
    )
    mapping = defaultdict(list)
    for num, col_name, repl_name in zip(
        outcome.get_column("idx"),
        outcome.get_column("cols"),
        outcome.get_column(".value"),
    ):
        expression = pl.col(col_name).alias(repl_name)
        mapping[num].append(expression)
    mapping = mapping.values()
    if index:
        mapping = ((pl.col(index), *entry) for entry in mapping)
    contents = map(df.with_columns, mapping)
    columns_to_select = [*index, *names_to]
    return pl.concat(contents, how="diagonal_relaxed").select(
        columns_to_select
    )


def _pivot_longer_no_dot_value(
    df: pl.DataFrame,
    outcome: pl.Series,
    names_to: Iterable,
    values_to: str,
    index: Iterable,
    columns: Iterable,
    names_transform: dict,
):
    if isinstance(df, pl.LazyFrame):
        height = df.select(pl.len()).collect().item()
    else:
        height = df.height
    values = (pl.col(col_name).alias(values_to) for col_name in columns)
    if index:
        values = ([pl.col(index), col_name] for col_name in values)
    values = map(df.select, values)
    idx = f"{names_to[0]}_" if len(names_to) == 1 else "".join(names_to)
    outcome = (
        outcome.struct.rename_fields(names=names_to)
        .struct.unnest()
        .cast(names_transform)
        .with_row_index(name=idx)
    )
    temp = (
        pl.int_range(height * len(columns), dtype=pl.UInt32)
        .floordiv(height)
        .alias(idx)
    )
    temp = pl.select(temp)
    if isinstance(df, pl.LazyFrame):
        temp = temp.lazy().join(outcome.lazy(), on=idx, how="left")
    else:
        temp = temp.join(outcome, on=idx, how="left")
    temp = temp.select(pl.exclude(idx))
    columns_to_select = [*index, *names_to, values_to]
    values = pl.concat(values)
    return pl.concat([values, temp], how="horizontal").select(
        columns_to_select
    )


def _pivot_longer_dot_value(
    df: pl.DataFrame,
    names_to: Iterable,
    outcome: pl.Series,
    index: Iterable,
    columns: Iterable,
    names_transform: Union[PolarsDataType, dict],
) -> pl.DataFrame:
    """
    Pivots the dataframe into the final form,
    for scenarios where .value is in names_to.
    """
    idx = "".join(names_to)
    booleans = (
        outcome.struct.unnest()
        .select(
            pl.Series(columns).alias(idx),
            pl.any_horizontal(pl.exclude(idx).is_null()),
        )
        .filter(pl.last())
    )
    if not booleans.is_empty():
        column_name = booleans.get_column(idx)[0]
        raise ValueError(
            f"Column label '{column_name}' "
            "could not be matched with any of the groups "
            "in the provided regex. Kindly provide a regular expression "
            "(with the correct groups) that matches all labels in the columns."
        )

    if names_to.count(".value") > 1:
        cols = outcome.struct.fields
        dot_value = [
            cols[num]
            for num, label in enumerate(names_to)
            if label == ".value"
        ]
        not_dot_value = [
            pl.col(field_name).alias(repl_name)
            for field_name, repl_name in zip(cols, names_to)
            if field_name not in dot_value
        ]

        outcome = outcome.struct.unnest().select(
            pl.concat_str(dot_value).alias(".value"), *not_dot_value
        )
    else:
        outcome = outcome.struct.rename_fields(names_to).struct.unnest()
    not_dot_value = [name for name in names_to if name != ".value"]
    outcome = outcome.with_row_index(idx).with_columns(
        pl.col(idx).first().over(not_dot_value).rank("dense").sub(1),
        pl.struct(not_dot_value),
    )
    mapping = defaultdict(list)
    for num, col_name, repl_name in zip(
        outcome.get_column(idx),
        columns,
        outcome.get_column(".value"),
    ):
        mapping[num].append(pl.col(col_name).alias(repl_name))
    if index:
        mapping = {
            num: [pl.col(index), *columns_to_select]
            for num, columns_to_select in mapping.items()
        }
    dot_value = outcome.get_column(".value").unique()
    outcome = outcome.select(idx, not_dot_value[0]).unique()
    outcome = dict(zip(outcome.to_series(0), outcome.to_series(1)))
    contents = []
    for key, columns_to_select in mapping.items():
        columns_to_append = outcome[key]
        columns_to_append = [
            pl.lit(stub_name, dtype=names_transform[repl_name]).alias(
                repl_name
            )
            for repl_name, stub_name in columns_to_append.items()
        ]
        contents.append(
            df.select(columns_to_select).with_columns(columns_to_append)
        )

    columns_to_select = [*index, *not_dot_value, *dot_value]
    return pl.concat(contents, how="diagonal_relaxed").select(
        columns_to_select
    )


def _pivot_longer_dot_value_only(
    df: pl.DataFrame,
    names_to: Iterable,
    outcome: pl.Series,
    index: Iterable,
    columns: Iterable,
) -> pl.DataFrame:
    """
    Pivots the dataframe into the final form,
    for scenarios where only '.value' is present in names_to.
    """

    if names_to.count(".value") > 1:
        outcome = outcome.struct.unnest().select(
            pl.concat_str(pl.all()).alias(".value")
        )
    else:
        outcome = outcome.struct.rename_fields(names_to).struct.unnest()
    outcome = outcome.with_columns(
        pl.col(".value").cum_count().over(".value").sub(1).alias("idx")
    )

    mapping = defaultdict(list)
    for num, col_name, repl_name in zip(
        outcome.get_column("idx"),
        columns,
        outcome.get_column(".value"),
    ):
        mapping[num].append(pl.col(col_name).alias(repl_name))
    if index:
        mapping = {
            num: [pl.col(index), *columns_to_select]
            for num, columns_to_select in mapping.items()
        }
    contents = map(df.with_columns, mapping.values())
    columns_to_select = [*index, *outcome.get_column(".value").unique()]
    return pl.concat(contents, how="diagonal_relaxed").select(
        columns_to_select
    )


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
                    "should be a string type, a ColumnSelector,  "
                    "an expression or a list/tuple that contains "
                    "a string and/or a ColumnSelector and/or an expression."
                )

        if isinstance(arg_value, (list, tuple)):
            for entry in arg_value:
                _check_type_single(entry=entry)
        else:
            _check_type_single(entry=arg_value)

    if (index is None) and (column_names is None):
        column_names = cs.expand_selector(df, pl.all())
        index = []
    elif (index is not None) and (column_names is not None):
        _check_type(arg_name="index", arg_value=index)
        index = cs.expand_selector(df, index)
        _check_type(arg_name="column_names", arg_value=column_names)
        column_names = cs.expand_selector(df, column_names)

    elif (index is None) and (column_names is not None):
        _check_type(arg_name="column_names", arg_value=column_names)
        column_names = cs.expand_selector(df, column_names)
        index = cs.expand_selector(df, pl.exclude(column_names))

    elif (index is not None) and (column_names is None):
        _check_type(arg_name="index", arg_value=index)
        index = cs.expand_selector(df, index)
        column_names = cs.expand_selector(df, pl.exclude(index))

    check("names_to", names_to, [list, tuple, str])
    if isinstance(names_to, (list, tuple)):
        uniques = set()
        for word in names_to:
            check(f"'{word}' in names_to", word, [str])
            if (word in uniques) and (word != ".value"):
                raise ValueError(f"'{word}' is duplicated in names_to.")
            uniques.add(word)
    names_to = [names_to] if isinstance(names_to, str) else names_to

    if names_sep and names_pattern:
        raise ValueError(
            "Only one of names_pattern or names_sep should be provided."
        )

    if names_sep is not None:
        check("names_sep", names_sep, [str])

    if names_pattern is not None:
        check("names_pattern", names_pattern, [str, list, tuple])
        if isinstance(names_pattern, (list, tuple)):
            for word in names_pattern:
                check(f"'{word}' in names_pattern", word, [str])
            if ".value" in names_to:
                raise ValueError(
                    ".value is not accepted in names_to "
                    "if names_pattern is a list/tuple."
                )
            if len(names_pattern) != len(names_to):
                raise ValueError(
                    f"The length of names_to does not match "
                    "the number of regexes in names_pattern. "
                    f"The length of names_to is {len(names_to)} "
                    f"while the number of regexes is {len(names_pattern)}."
                )

    check("values_to", values_to, [str, list, tuple])
    values_to_is_a_sequence = isinstance(values_to, (list, tuple))
    names_pattern_is_a_sequence = isinstance(names_pattern, (list, tuple))
    if values_to_is_a_sequence:
        if not names_pattern_is_a_sequence:
            raise TypeError(
                "values_to can be a list/tuple only "
                "if names_pattern is a list/tuple."
            )

        if len(names_pattern) != len(values_to):
            raise ValueError(
                f"The length of values_to does not match "
                "the number of regexes in names_pattern. "
                f"The length of values_to is {len(values_to)} "
                f"while the number of regexes is {len(names_pattern)}."
            )
        uniques = set()
        for word in values_to:
            check(f"{word} in values_to", word, [str])
            if word in uniques:
                raise ValueError(f"'{word}' is duplicated in values_to.")
            uniques.add(word)

    columns_to_append = any(label != ".value" for label in names_to)
    if values_to_is_a_sequence or columns_to_append:
        check("names_transform", names_transform, [DataTypeClass, dict])
        if isinstance(names_transform, dict):
            for _, dtype in names_transform.items():
                check(
                    "dtype in the names_transform mapping",
                    dtype,
                    [DataTypeClass],
                )
            names_transform = {
                label: names_transform.get(label, pl.Utf8)
                for label in names_to
            }
        else:
            names_transform = {label: names_transform for label in names_to}

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
