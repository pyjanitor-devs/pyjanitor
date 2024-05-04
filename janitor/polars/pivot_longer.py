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
<<<<<<< HEAD

=======
    if not isinstance(df, pl.LazyFrame) and (".value" not in names_to):
        columns_to_select = [*index, *names_to, values_to]
        # use names that are unlikely to cause a conflict
        variable_name = "".join(columns_to_select)
        explode_name = [*index, values_to]
        return (
            df.select(pl.all().implode())
            .melt(
                id_vars=index,
                value_vars=column_names,
                value_name=values_to,
                variable_name=variable_name,
            )
            .with_columns(
                pl.col(variable_name)
                .str.split(by=names_sep)
                .list.to_struct(n_field_strategy="max_width", fields=names_to)
            )
            .unnest(variable_name)
            .cast(names_transform)
            .explode(explode_name)
            .select(columns_to_select)
        )
>>>>>>> 2a9bc7c (add support for lazyframe)
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

<<<<<<< HEAD
    if ".value" not in names_to:
        outcome = outcome.struct.rename_fields(names_to)
        return _pivot_longer_no_dot_value(
            df=df,
            outcome=outcome,
            values_to=values_to,
            index=index,
            columns=columns,
            names_to=names_to,
            names_transform=names_transform,
        )
=======
>>>>>>> 2a9bc7c (add support for lazyframe)
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
<<<<<<< HEAD

    columns = df.select(column_names).columns
    outcome = pl.Series(columns).str.extract_groups(names_pattern)
    len_outcome = len(outcome.struct.fields)
    len_names_to = len(names_to)
    if len_names_to != len_outcome:
        raise ValueError(
            f"The length of names_to does not match "
            "the number of fields extracted. "
            f"The length of names_to is {len_names_to} "
            "while the number of fields extracted is "
            f"{len_outcome}."
        )
    if ".value" not in names_to:
        outcome = outcome.struct.rename_fields(names_to)
        return _pivot_longer_no_dot_value(
            df=df,
            outcome=outcome,
            values_to=values_to,
            index=index,
            columns=columns,
            names_to=names_to,
            names_transform=names_transform,
        )
=======
    if not isinstance(df, pl.LazyFrame) and (".value" not in names_to):
        columns_to_select = [*index, *names_to, values_to]
        # use names that are unlikely to cause a conflict
        variable_name = "".join(columns_to_select)
        explode_name = [*index, values_to]
        return (
            df.select(pl.all().implode())
            .melt(
                id_vars=index,
                value_vars=column_names,
                value_name=values_to,
                variable_name=variable_name,
            )
            .with_columns(
                pl.col(variable_name)
                .str.extract_groups(names_pattern)
                .struct.rename_fields(names_to)
            )
            .unnest(variable_name)
            .cast(names_transform)
            .explode(explode_name)
            .select(columns_to_select)
        )
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

>>>>>>> 2a9bc7c (add support for lazyframe)
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
    headers_dict = defaultdict(list)
    non_headers_dict = defaultdict(list)
    for num, col_name, value_header, name_header in zip(
        outcome.get_column("idx"),
        outcome.get_column("cols"),
        outcome.get_column(".value"),
        outcome.get_column("value"),
    ):
<<<<<<< HEAD
        non_headers_dict[num].append((col_name, name_header))
        headers_dict[num].append((col_name, value_header))
    contents = []
    for key, value in headers_dict.items():
        expression = [] if index is None else [pl.col(index)]
        columns_to_select = [
            pl.col(col_name).alias(repl_name) for col_name, repl_name in value
        ]
        expression.extend(columns_to_select)
        columns_to_append = [
            pl.lit(col_name, dtype=names_transform[repl_name]).alias(repl_name)
            for col_name, repl_name in non_headers_dict[key]
        ]

        contents.append(df.select(expression).with_columns(columns_to_append))
    columns_to_select = [] if not index else list(index)
    columns_to_select.extend(chain.from_iterable(zip(names_to, values_to)))
=======
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
>>>>>>> 2a9bc7c (add support for lazyframe)
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
    headers_dict = defaultdict(list)
    for num, col_name, name_header in zip(
        outcome.get_column("idx"),
        outcome.get_column("cols"),
        outcome.get_column(".value"),
    ):
<<<<<<< HEAD
        headers_dict[num].append((col_name, name_header))

    contents = []
    for _, value in headers_dict.items():
        expression = [] if index is None else [pl.col(index)]
        columns_to_select = [
            pl.col(col_name).alias(repl_name) for col_name, repl_name in value
        ]
        expression.extend(columns_to_select)

        contents.append(df.select(expression))
    return pl.concat(contents, how="diagonal_relaxed")


def _pivot_longer_no_dot_value(
    df: pl.DataFrame,
    outcome: pl.Series,
    names_to: Iterable,
    values_to: str,
    index: Iterable,
    columns: Iterable,
    names_transform: dict,
) -> pl.DataFrame:
    """
    Reshape the data for scenarios where .value
    is not present in names_to,
    or names_to is not a list/tuple.
    """
    contents = []
    for col_name, mapping in zip(columns, outcome):
        expression = (
            [pl.col(col_name)]
            if index is None
            else [pl.col(index), pl.col(col_name).alias(values_to)]
        )
        columns_to_append = [
            pl.lit(label, dtype=names_transform[header]).alias(header)
            for header, label in mapping.items()
        ]
        _frame = df.select(expression).with_columns(columns_to_append)
        contents.append(_frame)
    columns_to_select = [] if not index else list(index)
    columns_to_select.extend(names_to)
    columns_to_select.append(values_to)
    return pl.concat(contents, how="diagonal_relaxed").select(
        pl.col(columns_to_select)
=======
        expression = pl.col(col_name).alias(repl_name)
        mapping[num].append(expression)
    mapping = mapping.values()
    if index:
        mapping = ((pl.col(index), *entry) for entry in mapping)
    contents = map(df.with_columns, mapping)
    columns_to_select = [*index, *names_to]
    return pl.concat(contents, how="diagonal_relaxed").select(
        columns_to_select
>>>>>>> 2a9bc7c (add support for lazyframe)
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
    outcome = outcome.struct.rename_fields(names_to)
    columns_to_append = [
        [
            pl.lit(label, dtype=names_transform[header]).alias(header)
            for header, label in dictionary.items()
        ]
        for dictionary in outcome
    ]
    values = (pl.col(col_name).alias(values_to) for col_name in columns)
    if index:
        values = ([pl.col(index), col_name] for col_name in values)
    values = map(df.select, values)
    values = (
        entry.with_columns(col)
        for entry, col in zip(values, columns_to_append)
    )
    values = pl.concat(values, how="diagonal_relaxed")
    columns_to_select = [*index, *names_to, values_to]
    return values.select(columns_to_select)


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
    booleans = outcome.struct.unnest().select(pl.all().is_null().any())
    for position in range(len(names_to)):
        if booleans.to_series(position).item():
            raise ValueError(
                f"Column labels '{columns[position]}' "
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
    idx = "".join(names_to)
    not_dot_value = [name for name in names_to if name != ".value"]
    outcome = outcome.with_row_index(idx).with_columns(
        pl.col(idx).first().over(not_dot_value).rank("dense").sub(1),
        pl.struct(not_dot_value),
    )
    headers_dict = defaultdict(list)
    for num, col_name, repl_name in zip(
        outcome.get_column(idx),
        columns,
        outcome.get_column(".value"),
    ):
<<<<<<< HEAD
        headers_dict[num].append((col_name, repl_name))

    non_headers_dict = dict()
=======
        mapping[num].append(pl.col(col_name).alias(repl_name))
    if index:
        mapping = {
            num: [pl.col(index), *columns_to_select]
            for num, columns_to_select in mapping.items()
        }
    dot_value = outcome.get_column(".value").unique()
>>>>>>> 2a9bc7c (add support for lazyframe)
    outcome = outcome.select(idx, not_dot_value[0]).unique()

    for key, value in zip(outcome.to_series(0), outcome.to_series(1)):
        value = [
            pl.lit(stub_name, dtype=names_transform[repl_name]).alias(
                repl_name
            )
            for repl_name, stub_name in value.items()
        ]
<<<<<<< HEAD
        non_headers_dict[key] = value
    contents = []
    for key, value in headers_dict.items():
        expression = [] if index is None else [pl.col(index)]
        columns_to_select = [
            pl.col(col_name).alias(repl_name) for col_name, repl_name in value
        ]
        expression.extend(columns_to_select)
        _frame = df.select(expression).with_columns(non_headers_dict[key])
        contents.append(_frame)
    columns_to_select = [] if not index else list(index)
    columns_to_select.extend(not_dot_value)
=======
        contents.append(
            df.select(columns_to_select).with_columns(columns_to_append)
        )

    columns_to_select = [*index, *not_dot_value, *dot_value]
>>>>>>> 2a9bc7c (add support for lazyframe)
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
    headers_dict = defaultdict(list)
    for num, col_name, repl_name in zip(
        outcome.get_column("idx"),
        columns,
        outcome.get_column(".value"),
    ):
<<<<<<< HEAD
        headers_dict[num].append((col_name, repl_name))

    contents = []
    for _, value in headers_dict.items():
        expression = [] if index is None else [pl.col(index)]
        columns_to_select = [
            pl.col(col_name).alias(repl_name) for col_name, repl_name in value
        ]
        expression.extend(columns_to_select)
        contents.append(df.select(expression))

    return pl.concat(contents, how="diagonal_relaxed")
=======
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
>>>>>>> 2a9bc7c (add support for lazyframe)


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

"""pivot_longer implementation for polars."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

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
