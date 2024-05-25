"""pivot_longer implementation for polars."""

from collections import defaultdict
from typing import Any, Iterable, Union

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
    df: Union[pl.DataFrame, pl.LazyFrame],
    index: ColumnNameOrSelector,
    column_names: ColumnNameOrSelector,
    names_to: Union[list, tuple, str],
    values_to: str,
    names_sep: str,
    names_pattern: str,
    names_transform: dict,
) -> Union[pl.DataFrame, pl.LazyFrame]:
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
    names_sep: Union[str, None],
    names_pattern: Union[str, None],
    values_to: str,
    names_transform: dict,
) -> pl.DataFrame:
    """
    This is where the spec DataFrame is created,
    before the final reshape.
    """
    spec = pl.DataFrame({".name": column_names})
    if names_sep is not None:
        split_expr = (
            pl.col(".name")
            .str.split(by=names_sep)
            .list.to_struct(n_field_strategy="max_width")
            .alias("extract")
        )
        spec = spec.with_columns(split_expr)
    else:
        extract_expr = (
            pl.col(".name")
            .str.extract_groups(pattern=names_pattern)
            .alias("extract")
        )
        spec = spec.with_columns(extract_expr)
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
        null_check = (
            spec.unnest(columns="extract")
            .filter(pl.any_horizontal(pl.exclude(".name").is_null().any()))
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
        rename_expr = pl.col("extract").struct.rename_fields(names=names_to)
        spec = spec.with_columns(rename_expr).unnest(columns="extract")
    else:
        spec = _squash_multiple_dot_value(spec=spec, names_to=names_to)
    if ".value" not in names_to:
        spec = spec.with_columns(pl.lit(value=values_to).alias(".value"))

    spec = spec.select(
        pl.col([".name", ".value"]), pl.exclude([".name", ".value"])
    )
    if names_transform is not None:
        spec = spec.cast(dtypes=names_transform)
    return spec


def _pivot_longer_dot_value(
    df: Union[pl.DataFrame, pl.LazyFrame], spec: pl.DataFrame
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Reshape DataFrame to long form based on metadata in `spec`.
    """
    index = [column for column in df.columns if column not in spec[".name"]]
    not_dot_value = [
        column for column in spec.columns if column not in {".name", ".value"}
    ]
    idx = "".join(spec.columns)
    if len(spec.columns) == 2:
        # use a cumulative count to properly pair the columns
        spec = spec.with_columns(
            pl.cum_count(".value").over(".value").alias(idx)
        )
    else:
        # assign a number to each group
        ngroup_expr = pl.first(idx).over(not_dot_value).rank("dense").sub(1)
        spec = spec.with_row_index(name=idx).with_columns(ngroup_expr)
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
    df_is_a_lazyframe = isinstance(df, pl.LazyFrame)
    df = [
        df.select(columns_to_select).with_columns(position)
        for columns_to_select, position in mapping
    ]

    df = pl.concat(df, how="diagonal_relaxed")
    spec = spec.select(idx, *not_dot_value).unique()
    if df_is_a_lazyframe:
        spec = spec.lazy()
    df = df.join(spec, on=idx, how="inner")
    df = df.select(pl.exclude(idx))
    return df


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

    if names_transform is not None:
        check("names_transform", names_transform, [dict])

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
