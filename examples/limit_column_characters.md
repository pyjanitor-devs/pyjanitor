# df.limit_column_characters()

## Description
This method truncates column names to a given character length. In the case of duplicated column names, numbers are appended to the columns with a character separator (default is "_").

## Parameters
### df
A pandas dataframe.

### column_length
Character length for which to truncate all columns. The column separator value and number for duplicate column name does
    not contribute. Therefore, if all columns are truncated to 10
    characters, the first distinct column will be 10 characters and the
    remaining will be 12 characters (assuming a column separator of one
    character).

### col_separator
The separator to use for counting distinct column values. Default is "_". Supply an empty string (i.e. '') to remove the
    separator.

## Setup
```python
import pandas as pd
import janitor

data_dict = {
    "really_long_name_for_a_column": range(10),
    "another_really_long_name_for_a_column": [2 * item for item in range(10)],
    "another_really_longer_name_for_a_column": list("lllongname"),
    "this_is_getting_out_of_hand": list("longername"),
}
```

## Example1: Standard truncation
 ```python
example_dataframe = pd.DataFrame(data_dict)

example_dataframe.limit_column_characters(7)
```

### Output

       really_  another another_1 this_is
    0        0        0         l       l
    1        1        2         l       o
    2        2        4         l       n
    3        3        6         o       g
    4        4        8         n       e
    5        5       10         g       r
    6        6       12         n       n
    7        7       14         a       a
    8        8       16         m       m
    9        9       18         e       e

## Example2: Standard truncation with different separator character

```python

example_dataframe2 = pd.DataFrame(data_dict)

example_dataframe2.limit_column_characters(7, ".")
```

### Output

       really_  another another.1 this_is
    0        0        0         l       l
    1        1        2         l       o
    2        2        4         l       n
    3        3        6         o       g
    4        4        8         n       e
    5        5       10         g       r
    6        6       12         n       n
    7        7       14         a       a
    8        8       16         m       m
    9        9       18         e       e
