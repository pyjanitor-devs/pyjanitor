"""expand implementation for polars."""

from __future__ import annotations

from janitor.utils import import_message

try:
    import polars as pl
except ImportError:
    import_message(
        submodule="polars",
        package="polars",
        conda_channel="conda-forge",
        pip_install=True,
    )


def cartesian_product(
    *inputs: tuple[pl.DataFrame, pl.LazyFrame, pl.Series]
) -> pl.DataFrame | pl.LazyFrame:
    """Creates a DataFrame from a cartesian combination of all inputs.

    Inspiration is from tidyr's expand_grid() function.

    The input argument should be a polars Series, DataFrame, or LazyFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor as jn
        >>> df = pd.DataFrame({"x": [1, 2], "y": [2, 1]})
        >>> data = pd.Series([1, 2, 3], name='z')
        >>> jn.cartesian_product(df, data)
           x  y  z
        0  1  2  1
        1  1  2  2
        2  1  2  3
        3  2  1  1
        4  2  1  2
        5  2  1  3

        `cartesian_product` also works with non-pandas objects:

        >>> data = {"x": [1, 2, 3], "y": [1, 2]}
        >>> cartesian_product(data)
           x  y
        0  1  1
        1  1  2
        2  2  1
        3  2  2
        4  3  1
        5  3  2

    Args:
        *inputs: Variable arguments. The argument should be
            a polars Series, DataFrame or LazyFrame.

    Returns:
        A polars DataFrame/LazyFrame.
    """  # noqa: E501

    unique_names = set()
    for entry in inputs:
        if isinstance(entry, (pl.LazyFrame, pl.DataFrame)):
            for column in entry.columns:
                if column in unique_names:
                    raise pl.exceptions.DuplicateError(
                        f"column with name '{column}' already exists."
                    )
                unique_names.add(column)
        elif isinstance(entry, pl.Series):
            column = entry.name
            if column in unique_names:
                raise pl.exceptions.DuplicateError(
                    f"column with name '{column}' already exists."
                )
            unique_names.add(column)
        else:
            raise TypeError(
                "Expected a polars LazyFrame, DataFrame, "
                f"or Series; instead got {type(entry).__name__}"
            )
    unique_names = []
    uniques = []
    for position, entry in enumerate(inputs):
        position = str(position)
        if isinstance(entry, (pl.LazyFrame, pl.DataFrame)):
            expression = pl.struct(pl.all()).alias(name=position)
            outcome = entry.select(expression.implode())
        else:
            outcome = pl.select(
                pl.struct(entry).alias(name=position).implode()
            )
        unique_names.append(position)
        uniques.append(outcome)
    cartesian = pl.concat(uniques, rechunk=True, how="horizontal")
    return cartesian
    for name in unique_names:
        cartesian = cartesian.unnest(name)
    return cartesian


def indices(dimensions):
    return dimensions
