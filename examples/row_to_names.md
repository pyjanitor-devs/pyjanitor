# df.row_to_names()

## Description
This method elevates a row to be the column names of a DataFrame. It contains parameters to remove the elevated row from the DataFrame along with removing the rows above the selected row.

    :param df: A pandas DataFrame.
    :param row_number: The row containing the variable names
    :param remove_row: Should the row be removed from the DataFrame?
    :param remove_rows_above: Should the rows above row_number be removed from the resulting DataFrame?

## Parameters
### df
A pandas dataframe.

### row_number
The number of the row containing the variable names. Remember, indexing starts at zero!

### remove_row (Default: False)
Remove the row that is now the headers from the DataFrame.

### remove_rows_above (Default: False)
Remove the rows from the index above `row_number`.


## Setup

```python
import pandas as pd
import janitor


data_dict = {
    "a": [1, 2, 3] * 3,
    "Bell__Chart": [1, 2, 3] * 3,
    "decorated-elephant": [1, 2, 3] * 3,
    "animals": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
}
```



## Example1: Move first row to column names
 ```python
example_dataframe = pd.DataFrame(data_dict)

example_dataframe.row_to_names(0)
```

### Output

       1  1  1   rabbit  Cambridge
    0  1  1  1   rabbit  Cambridge
    1  2  2  2  leopard   Shanghai
    2  3  3  3     lion      Basel
    3  1  1  1   rabbit  Cambridge
    4  2  2  2  leopard   Shanghai
    5  3  3  3     lion      Basel
    6  1  1  1   rabbit  Cambridge
    7  2  2  2  leopard   Shanghai

## Example2: Move first row to column names and remove row

```python
example_dataframe = pd.DataFrame(data_dict)

example_dataframe.row_to_names(0, remove_row=True)
```

### Output

       1  1  1   rabbit  Cambridge
    1  2  2  2  leopard   Shanghai
    2  3  3  3     lion      Basel
    3  1  1  1   rabbit  Cambridge
    4  2  2  2  leopard   Shanghai
    5  3  3  3     lion      Basel
    6  1  1  1   rabbit  Cambridge
    7  2  2  2  leopard   Shanghai
    8  3  3  3     lion      Basel

## Example3: Move first row to column names, remove row, and remove rows above selected row

```python
example_dataframe = pd.DataFrame(data_dict)

example_dataframe.row_to_names(2, remove_row=True, remove_rows_above=True)
```

### Output

       3  3  3     lion      Basel
    3  1  1  1   rabbit  Cambridge
    4  2  2  2  leopard   Shanghai
    5  3  3  3     lion      Basel
    6  1  1  1   rabbit  Cambridge
    7  2  2  2  leopard   Shanghai
    8  3  3  3     lion      Basel
