# df.round_to_fraction()

## Description
This method modifies a column of floats to a nearest fraction, given a certain denominator.

## Parameters
### df
A pandas dataframe.

### colname
The column which to apply the fraction transformation.

### denominator
The denominator of the fraction to round to. For example, `2`  would values to the nearest half (i.e. `1/2`).

## Setup
```python
import pandas as pd
import janitor

data_dict = {
    "a": [1.23452345, 2.456234, 3.2346125] * 3,
    "Bell__Chart": [1/3, 2/7, 3/2] * 3,
    "decorated-elephant": [1/234, 2/13, 3/167] * 3,
    "animals": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
}
```

## Example 1: Rounding the first column to the nearest half
 ```python
example_dataframe = pd.DataFrame(data_dict)

example_dataframe.round_to_fraction('a', 2)
```

### Output

         a  Bell__Chart  decorated-elephant  animals     cities
    0  1.0     0.333333            0.004274   rabbit  Cambridge
    1  2.5     0.285714            0.153846  leopard   Shanghai
    2  3.0     1.500000            0.017964     lion      Basel
    3  1.0     0.333333            0.004274   rabbit  Cambridge
    4  2.5     0.285714            0.153846  leopard   Shanghai
    5  3.0     1.500000            0.017964     lion      Basel
    6  1.0     0.333333            0.004274   rabbit  Cambridge
    7  2.5     0.285714            0.153846  leopard   Shanghai
    8  3.0     1.500000            0.017964     lion      Basel

## Example 2: Rounding the first column to nearest third

```python
example_dataframe2 = pd.DataFrame(data_dict)

example_dataframe2.limit_column_characters('a', 3)
```

### Output

              a  Bell__Chart  decorated-elephant  animals     cities
    0  1.333333     0.333333            0.004274   rabbit  Cambridge
    1  2.333333     0.285714            0.153846  leopard   Shanghai
    2  3.333333     1.500000            0.017964     lion      Basel
    3  1.333333     0.333333            0.004274   rabbit  Cambridge
    4  2.333333     0.285714            0.153846  leopard   Shanghai
    5  3.333333     1.500000            0.017964     lion      Basel
    6  1.333333     0.333333            0.004274   rabbit  Cambridge
    7  2.333333     0.285714            0.153846  leopard   Shanghai
    8  3.333333     1.500000            0.017964     lion      Basel

## Example 3: Rounding the first column to the nearest third and rounding each value to the 10,000th place

```python
example_dataframe2 = pd.DataFrame(data_dict)

example_dataframe2.limit_column_characters('a', 3, 4)
```

### Output

            a  Bell__Chart  decorated-elephant  animals     cities
    0  1.3333     0.333333            0.004274   rabbit  Cambridge
    1  2.3333     0.285714            0.153846  leopard   Shanghai
    2  3.3333     1.500000            0.017964     lion      Basel
    3  1.3333     0.333333            0.004274   rabbit  Cambridge
    4  2.3333     0.285714            0.153846  leopard   Shanghai
    5  3.3333     1.500000            0.017964     lion      Basel
    6  1.3333     0.333333            0.004274   rabbit  Cambridge
    7  2.3333     0.285714            0.153846  leopard   Shanghai
    8  3.3333     1.500000            0.017964     lion      Basel
