# df.add_column()

## Description

This method adds a column to the dataframe. It is intended to be the method-chaining alternative to: `df[colname] = value`.

## Parameters
### df
A pandas dataframe.

### colname
Name of the new column. Should be a string, in order for the column name to be compatible with the Feather binary format (this is a useful thing to have).

### value
Either a single value, or a list/tuple of values.

### fill_remaining
If value is a tuple or list that is smaller than the number of rows in the DataFrame, repeat the list or tuple (R-style) to the end of the DataFrame.

## Setup

```python
import pandas as pd
import janitor


data = {
    "a": [1, 2, 3] * 3,
    "Bell__Chart": [1, 2, 3] * 3,
    "decorated-elephant": [1, 2, 3] * 3,
    "animals": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
}

df = pd.DataFrame(data)
```


## Example 1: Create a new column with a single value

```python
df.add_column("city_pop", 100000)
```

## Output
       a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    0  1            1                   1   rabbit  Cambridge    100000
    1  2            2                   2  leopard   Shanghai    100000
    2  3            3                   3     lion      Basel    100000
    3  1            1                   1   rabbit  Cambridge    100000
    4  2            2                   2  leopard   Shanghai    100000
    5  3            3                   3     lion      Basel    100000
    6  1            1                   1   rabbit  Cambridge    100000
    7  2            2                   2  leopard   Shanghai    100000
    8  3            3                   3     lion      Basel    100000

## Example 2: Create a new column with an iterator which fills to the column size
```python
df.add_column("city_pop", range(3), fill_remaining=True)
```
## Output
       a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    0  1            1                   1   rabbit  Cambridge         0
    1  2            2                   2  leopard   Shanghai         1
    2  3            3                   3     lion      Basel         2
    3  1            1                   1   rabbit  Cambridge         0
    4  2            2                   2  leopard   Shanghai         1
    5  3            3                   3     lion      Basel         2
    6  1            1                   1   rabbit  Cambridge         0
    7  2            2                   2  leopard   Shanghai         1
    8  3            3                   3     lion      Basel         2

## Example 3: Add new column based on mutation of other columns
```python
df.add_column("city_pop", df.Bell__Chart - 2 * df.a)
```

## Output

       a  Bell__Chart  decorated-elephant  animals     cities  city_pop
    0  1            1                   1   rabbit  Cambridge        -1
    1  2            2                   2  leopard   Shanghai        -2
    2  3            3                   3     lion      Basel        -3
    3  1            1                   1   rabbit  Cambridge        -1
    4  2            2                   2  leopard   Shanghai        -2
    5  3            3                   3     lion      Basel        -3
    6  1            1                   1   rabbit  Cambridge        -1
    7  2            2                   2  leopard   Shanghai        -2
    8  3            3                   3     lion      Basel        -3
