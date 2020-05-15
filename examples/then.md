# df.then()

## Description
This method allows for arbitrary functions to be used in the pyJanitor method-chaining fluent API.

## Parameters
### df
A pandas dataframe.

### func
A function that takes one parameter and returns that same parameter. For consistency it should be `df` to indicate that it's the Pandas DataFrame.  After that, do whatever you want in the middle. Go crazy.


## Setup
```python
import pandas as pd
import janitor

data_dict = {
    "a": [1.23, 2.45, 3.23] * 3,
    "Bell__Chart": [1, 2, 3] * 3,
    "decorated-elephant": [1, 2, 3] * 3,
    "animals": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
}
```

## Example 1
 ```python
def remove_first_two_letters_from_col_names(df):
    col_names = df.columns
    col_names = [name[2:] for name in col_names]
    df.columns = col_names
    return df

example_dataframe = pd.DataFrame(data_dict)

example_dataframe.then(remove_first_two_letters_from_col_names)
```

### Output

             ll__Chart  corated-elephant    imals       ties
    0  1.23          1                 1   rabbit  Cambridge
    1  2.45          2                 2  leopard   Shanghai
    2  3.23          3                 3     lion      Basel
    3  1.23          1                 1   rabbit  Cambridge
    4  2.45          2                 2  leopard   Shanghai
    5  3.23          3                 3     lion      Basel
    6  1.23          1                 1   rabbit  Cambridge
    7  2.45          2                 2  leopard   Shanghai
    8  3.23          3                 3     lion      Basel

## Example 2

```python
def remove_rows_3_and_4(df):
    df = df.drop(3, axis=0)
    df = df.drop(4, axis=0)
    return df


example_dataframe2 = pd.DataFrame(data_dict)

example_dataframe2.then(remove_rows_3_and_4)
```

### Output

          a  Bell__Chart  decorated-elephant  animals     cities
    0  1.23            1                   1   rabbit  Cambridge
    1  2.45            2                   2  leopard   Shanghai
    2  3.23            3                   3     lion      Basel
    5  3.23            3                   3     lion      Basel
    6  1.23            1                   1   rabbit  Cambridge
    7  2.45            2                   2  leopard   Shanghai
    8  3.23            3                   3     lion      Basel

## Example 3

```python
example_dataframe = pd.DataFrame(data_dict)
example_dataframe = (
    example_dataframe
    .then(remove_first_two_letters_from_col_names)
    .then(remove_rows_3_and_4)
)
```

### Output

             ll__Chart  corated-elephant    imals       ties
    0  1.23          1                 1   rabbit  Cambridge
    1  2.45          2                 2  leopard   Shanghai
    2  3.23          3                 3     lion      Basel
    5  3.23          3                 3     lion      Basel
    6  1.23          1                 1   rabbit  Cambridge
    7  2.45          2                 2  leopard   Shanghai
    8  3.23          3                 3     lion      Basel
