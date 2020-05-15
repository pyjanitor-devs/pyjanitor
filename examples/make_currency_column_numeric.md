# df.make_currency_column_numeric()

## Description

This method allows one to take a column containing currency values, inadvertently imported as a string, and cast it as a float. This is usually the case when reading CSV files that were modified in Excel. Empty strings (i.e. `''`) are retained as `NaN` values.

## Parameters
### df
A Pandas DataFrame

### col_name
Name of the new column. Should be a string, in order for the column name to be compatible with the Feather binary format (this is a useful thing to have).

### type
Type of conversion to perform. If `None`, defaults to standard conversion. Other option is `'accounting'` which cleans values that use parentheses for negative numbers and a dash for zero.

### cast_non_numeric
A dict of how to coerce certain strings. For example, if there are values of 'REORDER' in the DataFrame, {'REORDER': 0} will cast all instances of 'REORDER' to 0.

### fill_all_non_numeric
Similar to `cast_non_numeric`, but fills all strings to the same value. For example, `fill_all_non_numeric=1`, will convert everything that does not coerce to a currency 1. May be used in tandem with `cast_non_numeric`.

### remove_non_numeric
Will remove rows of a DataFrame that contain non-numeric values in the `col_name` column. Defaults to `False`.

## Setup

```python
import pandas as pd
import janitor


data = {
    "a": ["-$1.00", "", "REPAY"] * 2 + ["$23.00", "", "Other Account"],
    "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
    "decorated-elephant": [1, 2, 3] * 3,
    "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
}
df = pd.DataFrame(data)
```


## Example 1: Coerce numeric values in column to float

```python
df.make_currency_column_numeric("a")
```

## Output
          a  Bell__Chart  decorated-elephant animals@#$%^     cities
    0  -1.0     1.234523                   1       rabbit  Cambridge
    1   NaN     2.456234                   2      leopard   Shanghai
    2   NaN     3.234612                   3         lion      Basel
    3  -1.0     1.234523                   1       rabbit  Cambridge
    4   NaN     2.456234                   2      leopard   Shanghai
    5   NaN     3.234612                   3         lion      Basel
    6  23.0     1.234523                   1       rabbit  Cambridge
    7   NaN     2.456234                   2      leopard   Shanghai
    8   NaN     3.234612                   3         lion      Basel

## Example 2: Coerce numeric values in column to float, and replace a string value with a specific value

```python
cast_non_numeric = {"REPAY": 22}
df.make_currency_column_numeric("a", cast_non_numeric=cast_non_numeric)
```

## Output
          a  Bell__Chart  decorated-elephant animals@#$%^     cities
    0  -1.0     1.234523                   1       rabbit  Cambridge
    1   NaN     2.456234                   2      leopard   Shanghai
    2  22.0     3.234612                   3         lion      Basel
    3  -1.0     1.234523                   1       rabbit  Cambridge
    4   NaN     2.456234                   2      leopard   Shanghai
    5  22.0     3.234612                   3         lion      Basel
    6  23.0     1.234523                   1       rabbit  Cambridge
    7   NaN     2.456234                   2      leopard   Shanghai
    8   NaN     3.234612                   3         lion      Basel

## Example 3: Coerce numeric values in column to float, and replace all string value with a specific value

```python
df.make_currency_column_numeric("a", fill_all_non_numeric=35)
```

## Output

          a  Bell__Chart  decorated-elephant animals@#$%^     cities
    0  -1.0     1.234523                   1       rabbit  Cambridge
    1   NaN     2.456234                   2      leopard   Shanghai
    2  35.0     3.234612                   3         lion      Basel
    3  -1.0     1.234523                   1       rabbit  Cambridge
    4   NaN     2.456234                   2      leopard   Shanghai
    5  35.0     3.234612                   3         lion      Basel
    6  23.0     1.234523                   1       rabbit  Cambridge
    7   NaN     2.456234                   2      leopard   Shanghai
    8  35.0     3.234612                   3         lion      Basel

## Example 4: Coerce numeric values in column to float, replace a string value with a specific value, and replace remaining string values with a specific value

```python
df.make_currency_column_numeric(
    "a",
    cast_non_numeric=cast_non_numeric,
    fill_all_non_numeric=35
)
```

## Output

          a  Bell__Chart  decorated-elephant animals@#$%^     cities
    0  -1.0     1.234523                   1       rabbit  Cambridge
    1   NaN     2.456234                   2      leopard   Shanghai
    2  22.0     3.234612                   3         lion      Basel
    3  -1.0     1.234523                   1       rabbit  Cambridge
    4   NaN     2.456234                   2      leopard   Shanghai
    5  22.0     3.234612                   3         lion      Basel
    6  23.0     1.234523                   1       rabbit  Cambridge
    7   NaN     2.456234                   2      leopard   Shanghai
    8  35.0     3.234612                   3         lion      Basel


## Example 5: Coerce numeric values in column to float, and remove string values

```python
df.make_currency_column_numeric("a", remove_non_numeric=True)
```

## Output

          a  Bell__Chart  decorated-elephant animals@#$%^     cities
    0  -1.0     1.234523                   1       rabbit  Cambridge
    1   NaN     2.456234                   2      leopard   Shanghai
    3  -1.0     1.234523                   1       rabbit  Cambridge
    4   NaN     2.456234                   2      leopard   Shanghai
    6  23.0     1.234523                   1       rabbit  Cambridge
    7   NaN     2.456234                   2      leopard   Shanghai


## Example 6: Coerce numeric values in column to float, replace a string value with a specific value, and remove remaining string values

```python
df.make_currency_column_numeric(
    "a",
    cast_non_numeric=cast_non_numeric,
    remove_non_numeric=True
)
```

## Output

          a  Bell__Chart  decorated-elephant animals@#$%^     cities
    0  -1.0     1.234523                   1       rabbit  Cambridge
    1   NaN     2.456234                   2      leopard   Shanghai
    2  22.0     3.234612                   3         lion      Basel
    3  -1.0     1.234523                   1       rabbit  Cambridge
    4   NaN     2.456234                   2      leopard   Shanghai
    5  22.0     3.234612                   3         lion      Basel
    6  23.0     1.234523                   1       rabbit  Cambridge
    7   NaN     2.456234                   2      leopard   Shanghai
