# df.filter_date()

## Description
This method modifies a column of floats to a nearest fraction, given a certain denominator.

## Parameters
### df
A pandas dataframe.

### column
The column which to apply the fraction transformation.

### start
The beginning date to use to filter the DataFrame.

### end
The end date to use to filter the DataFrame.

### years
The years to use to filter the DataFrame (it expects an iterable).

### months
The months to use to filter the DataFrame (it expects an iterable).

### days
The days to use to filter the DataFrame (it expects an iterable).

### column_date_options
Special options to use when parsing the date column in the original DataFrame. The options may be found [at the official Pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html).

### format
It you're using a format for `start` or `end` that is not recognized natively by pandas' `to_datetime` function, you may supply the format yourself. Python date and time formats [may be found here](http://strftime.org/).
__Note__: This only affects the format of the `start` and `end` parameters. If there's an issue with the format of the DataFrame being parsed, you would pass `{'format': your_format}` to `column_date_options`.

## Setup
```python
import pandas as pd
import janitor

date_list = [
    [1, "01/28/19"],
    [2, "01/29/19"],
    [3, "01/30/19"],
    [4, "01/31/19"],
    [5, "02/01/19"],
    [6, "02/02/19"],
    [7, "02/03/19"],
    [8, "02/04/19"],
    [9, "02/05/19"],
    [10, "02/06/19"],
    [11, "02/07/20"],
    [12, "02/08/20"],
    [13, "02/09/20"],
    [14, "02/10/20"],
    [15, "02/11/20"],
    [16, "02/12/20"],
    [17, "02/07/20"],
    [18, "02/08/20"],
    [19, "02/09/20"],
    [20, "02/10/20"],
    [21, "02/11/20"],
    [22, "02/12/20"],
    [23, "03/08/20"],
    [24, "03/09/20"],
    [25, "03/10/20"],
    [26, "03/11/20"],
    [27, "03/12/20"]]

example_dataframe = pd.DataFrame(date_list, columns=['AMOUNT', 'DATE'])
```

## Example 1: Filter dataframe between two dates
 ```python
start = "01/29/19"
end = "01/30/19"

example_dataframe.filter_date('DATE', start=start, end=end)
```

### Output

       AMOUNT       DATE
    1       2 2019-01-29
    2       3 2019-01-30

## Example 2: Using a different date format for filtering
 ```python
end = "01$$$30$$$19"
format = "%m$$$%d$$$%y"

example_dataframe.filter_date('DATE', end=end, format=format)
```

### Output

       AMOUNT       DATE
    0       1 2019-01-28
    1       2 2019-01-29
    2       3 2019-01-30

## Example 3: Filtering by year

```python
years = [2019]

example_dataframe.filter_date('DATE', years=years)
```

### Output

       AMOUNT       DATE
    0       1 2019-01-28
    1       2 2019-01-29
    2       3 2019-01-30
    3       4 2019-01-31
    4       5 2019-02-01
    5       6 2019-02-02
    6       7 2019-02-03
    7       8 2019-02-04
    8       9 2019-02-05
    9      10 2019-02-06

## Example 4: Filtering by year and month

```python
years = [2020]
months = [3]

example_dataframe.filter_date('DATE', years=years, months=months)
```

### Output

        AMOUNT       DATE
    22      23 2020-03-08
    23      24 2020-03-09
    24      25 2020-03-10
    25      26 2020-03-11
    26      27 2020-03-12

## Example 5: Filtering by year and day

```python
years = [2020]
days = range(10, 12)

example_dataframe.filter_date('DATE', years=years, days=days)
```

### Output

        AMOUNT       DATE
    13      14 2020-02-10
    14      15 2020-02-11
    19      20 2020-02-10
    20      21 2020-02-11
    24      25 2020-03-10
    25      26 2020-03-11
