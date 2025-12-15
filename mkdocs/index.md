# pyjanitor

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ericmjl/pyjanitor/dev)

<!-- pypi-doc -->
`pyjanitor` is a Python implementation of the R package [`janitor`][janitor].
It provides a clean user-friendly API for extending pandas with powerful and readable data-cleaning functions .

[janitor]: https://github.com/sfirke/janitor

## Quick start

- Installation: `conda install -c conda-forge pyjanitor`. Read more installation instructions [here](https://pyjanitor-devs.github.io/pyjanitor/#installation).
- Check out the collection of [general functions](https://pyjanitor-devs.github.io/pyjanitor/api/functions/).

## Why janitor?

Originally a port of the R package,
`pyjanitor` has evolved from a set of convenient data cleaning routines
into an experiment with the [`method chaining`][mc] paradigm.

[mc]: https://towardsdatascience.com/the-unreasonable-effectiveness-of-method-chaining-in-pandas-15c2109e3c69

Data preprocessing usually consists of a series of steps
that involve transforming raw data into an understandable/usable format.
These series of steps need to be run in a certain sequence to achieve success.
We take a base data file as the starting point,
and perform actions on it,
such as removing null/empty rows,
replacing them with other values,
adding/renaming/removing columns of data,
filtering rows and others.
More formally, these steps along with their relationships
and dependencies are commonly referred to as a Directed Acyclic Graph (DAG).

The `pandas` API has been invaluable for the Python data science ecosystem,
and implements method chaining of a subset of methods as part of the API.
For example, resetting indexes (`.reset_index()`),
dropping null values (`.dropna()`), and more,
are accomplished via the appropriate `pd.DataFrame` method calls.

Inspired by the ease-of-use
and expressiveness of the `dplyr` package
of the R statistical language ecosystem,
we have evolved `pyjanitor` into a language
for expressing the data processing DAG for `pandas` users.
<!-- pypi-doc -->

To accomplish this,
actions for which we would need to invoke imperative-style statements,
can be replaced with method chains
that allow one to read off the logical order of actions taken.
Let us see the annotated example below.
First off, here is the textual description of a data cleaning pathway:

1. Create a `DataFrame`.
2. Delete one column.
3. Drop rows with empty values in two particular columns.
4. Rename another two columns.
5. Add a new column.

Let's import some libraries
and begin with some sample data for this example:

```python
# Libraries
import numpy as np
import pandas as pd
import janitor

# Sample Data curated for this example
company_sales = {
    'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
    'Company1': [150.0, 200.0, 300.0, 400.0],
    'Company2': [180.0, 250.0, np.nan, 500.0],
    'Company3': [400.0, 500.0, 600.0, 675.0]
}
```

In `pandas` code, most users might type something like this:

```python
# The Pandas Way

# 1. Create a pandas DataFrame from the company_sales dictionary
df = pd.DataFrame.from_dict(company_sales)

# 2. Delete a column from the DataFrame. Say 'Company1'
del df['Company1']

# 3. Drop rows that have empty values in columns 'Company2' and 'Company3'
df = df.dropna(subset=['Company2', 'Company3'])

# 4. Rename 'Company2' to 'Amazon' and 'Company3' to 'Facebook'
df = df.rename(
    {
        'Company2': 'Amazon',
        'Company3': 'Facebook',
    },
    axis=1,
)

# 5. Let's add some data for another company. Say 'Google'
df['Google'] = [450.0, 550.0, 800.0]

# Output looks like this:
# Out[15]:
#   SalesMonth  Amazon  Facebook  Google
# 0        Jan   180.0     400.0   450.0
# 1        Feb   250.0     500.0   550.0
# 3      April   500.0     675.0   800.0
```

Slightly more advanced users might take advantage of the functional API:

```python
df = (
    pd.DataFrame(company_sales)
    .drop(columns="Company1")
    .dropna(subset=["Company2", "Company3"])
    .rename(columns={"Company2": "Amazon", "Company3": "Facebook"})
    .assign(Google=[450.0, 550.0, 800.0])
)

# The output is the same as before, and looks like this:
# Out[15]:
#   SalesMonth  Amazon  Facebook  Google
# 0        Jan   180.0     400.0   450.0
# 1        Feb   250.0     500.0   550.0
# 3      April   500.0     675.0   800.0
```


With `pyjanitor`, we enable method chaining with method names
that are *explicitly named verbs*, which describe the action taken.

```python
df = (
    pd.DataFrame.from_dict(company_sales)
    .remove_columns(["Company1"])
    .dropna(subset=["Company2", "Company3"])
    .rename_column("Company2", "Amazon")
    .rename_column("Company3", "Facebook")
    .add_column("Google", [450.0, 550.0, 800.0])
)

# Output looks like this:
# Out[15]:
#   SalesMonth  Amazon  Facebook  Google
# 0        Jan   180.0     400.0   450.0
# 1        Feb   250.0     500.0   550.0
# 3      April   500.0     675.0   800.0
```

As such,
`pyjanitor`'s etymology has a two-fold relationship to "cleanliness".
Firstly, it's about extending Pandas with convenient data cleaning routines.
Secondly, it's about providing a cleaner, method-chaining, verb-based API
for common pandas routines.


<!-- pypi-doc -->
## Installation

`pyjanitor` is currently installable from PyPI:

```bash
pip install pyjanitor
```

`pyjanitor` also can be installed by the conda package manager:

```bash
conda install pyjanitor -c conda-forge
```

`pyjanitor` can be installed by the pipenv environment manager too. This requires enabling prerelease dependencies:

```bash
pipenv install --pre pyjanitor
```

`pyjanitor` requires Python 3.6+.

## Functionality

Current functionality includes:

- Cleaning columns name (multi-indexes are possible!)
- Removing empty rows and columns
- Identifying duplicate entries
- Encoding columns as categorical
- Splitting your data into features and targets (for machine learning)
- Adding, removing, and renaming columns
- Coalesce multiple columns into a single column
- Date conversions (from matlab, excel, unix) to Python datetime format
- Expand a single column that has delimited, categorical values
  into dummy-encoded variables
- Concatenating and deconcatenating columns, based on a delimiter
- Syntactic sugar for filtering the dataframe based on queries on a column
- Experimental submodules for finance, biology, chemistry, engineering, and pyspark
<!-- pypi-doc -->

## API

The idea behind the API is two-fold:

- Copy the R package function names,
  but enable Pythonic use with method chaining or `pandas` piping.
- Add other utility functions
  that make it easy to do data cleaning/preprocessing in `pandas`.

Continuing with the company_sales dataframe previously used:

```python
import pandas as pd
import numpy as np
company_sales = {
    'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
    'Company1': [150.0, 200.0, 300.0, 400.0],
    'Company2': [180.0, 250.0, np.nan, 500.0],
    'Company3': [400.0, 500.0, 600.0, 675.0]
}
```

As such, there are three ways to use the API.
The first, and most strongly recommended one, is to use `pyjanitor`'s functions
as if they were native to pandas.

```python
import janitor  # upon import, functions are registered as part of pandas.

# This cleans the column names as well as removes any duplicate rows
df = pd.DataFrame.from_dict(company_sales).clean_names().remove_empty()
```

The second is the functional API.

```python
from janitor import clean_names, remove_empty

df = pd.DataFrame.from_dict(company_sales)
df = clean_names(df)
df = remove_empty(df)
```

The final way is to use the [`pipe()`][pipe] method:

[pipe]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html

```python
from janitor import clean_names, remove_empty
df = (
    pd.DataFrame.from_dict(company_sales)
    .pipe(clean_names)
    .pipe(remove_empty)
)
```

## Contributing

Follow the [development guide](https://pyjanitor-devs.github.io/pyjanitor/devguide/) for a full description of the process of contributing to `pyjanitor`.

## Adding new functionality

Keeping in mind the etymology of pyjanitor,
contributing a new function to pyjanitor
is a task that is not difficult at all.

### Define a function

First off, you will need to define the function
that expresses the data processing/cleaning routine,
such that it accepts a dataframe as the first argument,
and returns a modified dataframe:

```python
import pandas_flavor as pf

@pf.register_dataframe_method
def my_data_cleaning_function(df, arg1, arg2, ...):
    # Put data processing function here.
    return df
```

We use [`pandas_flavor`](https://github.com/Zsailer/pandas_flavor) to register the function natively on a `pandas.DataFrame`.


### Add a test case

Secondly, we ask that you contribute a test case,
to ensure that the function works as intended.
Follow the [contribution] docs for further details.

[contribution]: https://pyjanitor-devs.github.io/pyjanitor/contributing.html#unit-test-guidelines

### Feature requests

If you have a feature request,
please post it as an issue on the GitHub repository issue tracker.
Even better, put in a PR for it!
We are more than happy to guide you through the codebase
so that you can put in a contribution to the codebase.

Because `pyjanitor` is currently maintained by volunteers
and has no fiscal support,
any feature requests will be prioritized according to
what maintainers encounter as a need in our day-to-day jobs.
Please temper expectations accordingly.

## API Policy

`pyjanitor` only extends or aliases the `pandas` API
(and other dataframe APIs),
but will never fix or replace them.

Undesirable `pandas` behaviour should be reported upstream
in the `pandas` [issue tracker](https://github.com/pandas-dev/pandas/issues).
We explicitly do not fix the `pandas` API.
If at some point the `pandas` devs
decide to take something from `pyjanitor`
and internalize it as part of the official `pandas` API,
then we will deprecate it from `pyjanitor`,
while acknowledging the original contributors' contribution
as part of the official deprecation record.


## Contributors

Thanks goes to these wonderful people who have contributed to pyjanitor:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/anzelpwj"><img src="?s=100" width="100px;" alt="anzelpwj"/><br /><sub><b>anzelpwj</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=anzelpwj" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aopisco"><img src="?s=100" width="100px;" alt="aopisco"/><br /><sub><b>aopisco</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=aopisco" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/apatao"><img src="?s=100" width="100px;" alt="apatao"/><br /><sub><b>apatao</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aapatao" title="Bug reports">🐛</a> <a href="#question-apatao" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/asearfos"><img src="?s=100" width="100px;" alt="asearfos"/><br /><sub><b>asearfos</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=asearfos" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ashenafiyb"><img src="?s=100" width="100px;" alt="ashenafiyb"/><br /><sub><b>ashenafiyb</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aashenafiyb" title="Bug reports">🐛</a> <a href="#question-ashenafiyb" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/asmirnov69"><img src="?s=100" width="100px;" alt="asmirnov69"/><br /><sub><b>asmirnov69</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aasmirnov69" title="Bug reports">🐛</a> <a href="#question-asmirnov69" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BaritoneBeard"><img src="?s=100" width="100px;" alt="BaritoneBeard"/><br /><sub><b>BaritoneBeard</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3ABaritoneBeard" title="Bug reports">🐛</a> <a href="#question-BaritoneBeard" title="Answering Questions">💬</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bdice"><img src="?s=100" width="100px;" alt="bdice"/><br /><sub><b>bdice</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=bdice" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/benjaminjack"><img src="?s=100" width="100px;" alt="benjaminjack"/><br /><sub><b>benjaminjack</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=benjaminjack" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bhallaY"><img src="?s=100" width="100px;" alt="bhallaY"/><br /><sub><b>bhallaY</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=bhallaY" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/catherinedevlin"><img src="?s=100" width="100px;" alt="catherinedevlin"/><br /><sub><b>catherinedevlin</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=catherinedevlin" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cduvallet"><img src="?s=100" width="100px;" alt="cduvallet"/><br /><sub><b>cduvallet</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=cduvallet" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cjmayers"><img src="?s=100" width="100px;" alt="cjmayers"/><br /><sub><b>cjmayers</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=cjmayers" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/CWen001"><img src="?s=100" width="100px;" alt="CWen001"/><br /><sub><b>CWen001</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=CWen001" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dave-frazzetto"><img src="?s=100" width="100px;" alt="dave-frazzetto"/><br /><sub><b>dave-frazzetto</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=dave-frazzetto" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dendrondal"><img src="?s=100" width="100px;" alt="dendrondal"/><br /><sub><b>dendrondal</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=dendrondal" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/derekpowell"><img src="?s=100" width="100px;" alt="derekpowell"/><br /><sub><b>derekpowell</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aderekpowell" title="Bug reports">🐛</a> <a href="#question-derekpowell" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DollofCuty"><img src="?s=100" width="100px;" alt="DollofCuty"/><br /><sub><b>DollofCuty</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=DollofCuty" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DSNortsev"><img src="?s=100" width="100px;" alt="DSNortsev"/><br /><sub><b>DSNortsev</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=DSNortsev" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dsouzadaniel"><img src="?s=100" width="100px;" alt="dsouzadaniel"/><br /><sub><b>dsouzadaniel</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=dsouzadaniel" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dwgoltra"><img src="?s=100" width="100px;" alt="dwgoltra"/><br /><sub><b>dwgoltra</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=dwgoltra" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Econundrums"><img src="?s=100" width="100px;" alt="Econundrums"/><br /><sub><b>Econundrums</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3AEconundrums" title="Bug reports">🐛</a> <a href="#question-Econundrums" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Eidhagen"><img src="?s=100" width="100px;" alt="Eidhagen"/><br /><sub><b>Eidhagen</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=Eidhagen" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/emmanuel-ferdman"><img src="?s=100" width="100px;" alt="emmanuel-ferdman"/><br /><sub><b>emmanuel-ferdman</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aemmanuel-ferdman" title="Bug reports">🐛</a> <a href="#question-emmanuel-ferdman" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/emnemnemnem"><img src="?s=100" width="100px;" alt="emnemnemnem"/><br /><sub><b>emnemnemnem</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=emnemnemnem" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ericclessantostv"><img src="?s=100" width="100px;" alt="ericclessantostv"/><br /><sub><b>ericclessantostv</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aericclessantostv" title="Bug reports">🐛</a> <a href="#question-ericclessantostv" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ericmjl"><img src="?s=100" width="100px;" alt="ericmjl"/><br /><sub><b>ericmjl</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=ericmjl" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ethompsy"><img src="?s=100" width="100px;" alt="ethompsy"/><br /><sub><b>ethompsy</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aethompsy" title="Bug reports">🐛</a> <a href="#question-ethompsy" title="Answering Questions">💬</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/evan-anderson"><img src="?s=100" width="100px;" alt="evan-anderson"/><br /><sub><b>evan-anderson</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Aevan-anderson" title="Bug reports">🐛</a> <a href="#question-evan-anderson" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eyaltrabelsi"><img src="?s=100" width="100px;" alt="eyaltrabelsi"/><br /><sub><b>eyaltrabelsi</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=eyaltrabelsi" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fireddd"><img src="?s=100" width="100px;" alt="fireddd"/><br /><sub><b>fireddd</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Afireddd" title="Bug reports">🐛</a> <a href="#question-fireddd" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gahjelle"><img src="?s=100" width="100px;" alt="gahjelle"/><br /><sub><b>gahjelle</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Agahjelle" title="Bug reports">🐛</a> <a href="#question-gahjelle" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gaworecki5"><img src="?s=100" width="100px;" alt="gaworecki5"/><br /><sub><b>gaworecki5</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=gaworecki5" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gcamargo2"><img src="?s=100" width="100px;" alt="gcamargo2"/><br /><sub><b>gcamargo2</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=gcamargo2" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gddcunh"><img src="?s=100" width="100px;" alt="gddcunh"/><br /><sub><b>gddcunh</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=gddcunh" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gjlynx"><img src="?s=100" width="100px;" alt="gjlynx"/><br /><sub><b>gjlynx</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=gjlynx" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hectormz"><img src="?s=100" width="100px;" alt="hectormz"/><br /><sub><b>hectormz</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=hectormz" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jcvall"><img src="?s=100" width="100px;" alt="jcvall"/><br /><sub><b>jcvall</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=jcvall" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jekwatt"><img src="?s=100" width="100px;" alt="jekwatt"/><br /><sub><b>jekwatt</b></sub></a><br /><a href="#tool-jekwatt" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jiafengkevinchen"><img src="?s=100" width="100px;" alt="jiafengkevinchen"/><br /><sub><b>jiafengkevinchen</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=jiafengkevinchen" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jk3587"><img src="?s=100" width="100px;" alt="jk3587"/><br /><sub><b>jk3587</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=jk3587" title="Code">💻</a> <a href="#tool-jk3587" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jonnybazookatone"><img src="?s=100" width="100px;" alt="jonnybazookatone"/><br /><sub><b>jonnybazookatone</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=jonnybazookatone" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joranbeasley"><img src="?s=100" width="100px;" alt="joranbeasley"/><br /><sub><b>joranbeasley</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Ajoranbeasley" title="Bug reports">🐛</a> <a href="#question-joranbeasley" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JoshuaC3"><img src="?s=100" width="100px;" alt="JoshuaC3"/><br /><sub><b>JoshuaC3</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=JoshuaC3" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/keoghdata"><img src="?s=100" width="100px;" alt="keoghdata"/><br /><sub><b>keoghdata</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=keoghdata" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Kevin-Smith77"><img src="?s=100" width="100px;" alt="Kevin-Smith77"/><br /><sub><b>Kevin-Smith77</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3AKevin-Smith77" title="Bug reports">🐛</a> <a href="#question-Kevin-Smith77" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kianmeng"><img src="?s=100" width="100px;" alt="kianmeng"/><br /><sub><b>kianmeng</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=kianmeng" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kimt33"><img src="?s=100" width="100px;" alt="kimt33"/><br /><sub><b>kimt33</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=kimt33" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kulini"><img src="?s=100" width="100px;" alt="kulini"/><br /><sub><b>kulini</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=kulini" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kurtispinkney"><img src="?s=100" width="100px;" alt="kurtispinkney"/><br /><sub><b>kurtispinkney</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=kurtispinkney" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lbeltrame"><img src="?s=100" width="100px;" alt="lbeltrame"/><br /><sub><b>lbeltrame</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=lbeltrame" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/loganthomas"><img src="?s=100" width="100px;" alt="loganthomas"/><br /><sub><b>loganthomas</b></sub></a><br /><a href="#tool-loganthomas" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lphk92"><img src="?s=100" width="100px;" alt="lphk92"/><br /><sub><b>lphk92</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=lphk92" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mdini"><img src="?s=100" width="100px;" alt="mdini"/><br /><sub><b>mdini</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=mdini" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MinchinWeb"><img src="?s=100" width="100px;" alt="MinchinWeb"/><br /><sub><b>MinchinWeb</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3AMinchinWeb" title="Bug reports">🐛</a> <a href="#question-MinchinWeb" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MollyCroke"><img src="?s=100" width="100px;" alt="MollyCroke"/><br /><sub><b>MollyCroke</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3AMollyCroke" title="Bug reports">🐛</a> <a href="#question-MollyCroke" title="Answering Questions">💬</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mphirke"><img src="?s=100" width="100px;" alt="mphirke"/><br /><sub><b>mphirke</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Amphirke" title="Bug reports">🐛</a> <a href="#question-mphirke" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mralbu"><img src="?s=100" width="100px;" alt="mralbu"/><br /><sub><b>mralbu</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=mralbu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NapsterInBlue"><img src="?s=100" width="100px;" alt="NapsterInBlue"/><br /><sub><b>NapsterInBlue</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=NapsterInBlue" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nvamsikrishna05"><img src="?s=100" width="100px;" alt="nvamsikrishna05"/><br /><sub><b>nvamsikrishna05</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Anvamsikrishna05" title="Bug reports">🐛</a> <a href="#question-nvamsikrishna05" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/OdinTech3"><img src="?s=100" width="100px;" alt="OdinTech3"/><br /><sub><b>OdinTech3</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3AOdinTech3" title="Bug reports">🐛</a> <a href="#question-OdinTech3" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/portc13"><img src="?s=100" width="100px;" alt="portc13"/><br /><sub><b>portc13</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=portc13" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/puruckertom"><img src="?s=100" width="100px;" alt="puruckertom"/><br /><sub><b>puruckertom</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=puruckertom" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/qtson"><img src="?s=100" width="100px;" alt="qtson"/><br /><sub><b>qtson</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=qtson" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rahosbach"><img src="?s=100" width="100px;" alt="rahosbach"/><br /><sub><b>rahosbach</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=rahosbach" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Rajat-181"><img src="?s=100" width="100px;" alt="Rajat-181"/><br /><sub><b>Rajat-181</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=Rajat-181" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ram-N"><img src="?s=100" width="100px;" alt="Ram-N"/><br /><sub><b>Ram-N</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=Ram-N" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rebeccawperry"><img src="?s=100" width="100px;" alt="rebeccawperry"/><br /><sub><b>rebeccawperry</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=rebeccawperry" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/richardqiu"><img src="?s=100" width="100px;" alt="richardqiu"/><br /><sub><b>richardqiu</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Arichardqiu" title="Bug reports">🐛</a> <a href="#question-richardqiu" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ricky-lim"><img src="?s=100" width="100px;" alt="ricky-lim"/><br /><sub><b>ricky-lim</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=ricky-lim" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/robertmitchellv"><img src="?s=100" width="100px;" alt="robertmitchellv"/><br /><sub><b>robertmitchellv</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Arobertmitchellv" title="Bug reports">🐛</a> <a href="#question-robertmitchellv" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sallyhong"><img src="?s=100" width="100px;" alt="sallyhong"/><br /><sub><b>sallyhong</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=sallyhong" title="Code">💻</a> <a href="#tool-sallyhong" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/samukweku"><img src="?s=100" width="100px;" alt="samukweku"/><br /><sub><b>samukweku</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=samukweku" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/samwalkow"><img src="?s=100" width="100px;" alt="samwalkow"/><br /><sub><b>samwalkow</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=samwalkow" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sauln"><img src="?s=100" width="100px;" alt="sauln"/><br /><sub><b>sauln</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Asauln" title="Bug reports">🐛</a> <a href="#question-sauln" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shandou"><img src="?s=100" width="100px;" alt="shandou"/><br /><sub><b>shandou</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=shandou" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shantanuo"><img src="?s=100" width="100px;" alt="shantanuo"/><br /><sub><b>shantanuo</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=shantanuo" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/smu095"><img src="?s=100" width="100px;" alt="smu095"/><br /><sub><b>smu095</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Asmu095" title="Bug reports">🐛</a> <a href="#question-smu095" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SorenFrohlich"><img src="?s=100" width="100px;" alt="SorenFrohlich"/><br /><sub><b>SorenFrohlich</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=SorenFrohlich" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sousa8697"><img src="?s=100" width="100px;" alt="Sousa8697"/><br /><sub><b>Sousa8697</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3ASousa8697" title="Bug reports">🐛</a> <a href="#question-Sousa8697" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/StephenSchroed"><img src="?s=100" width="100px;" alt="StephenSchroed"/><br /><sub><b>StephenSchroed</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=StephenSchroed" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/szuckerman"><img src="?s=100" width="100px;" alt="szuckerman"/><br /><sub><b>szuckerman</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=szuckerman" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thatlittleboy"><img src="?s=100" width="100px;" alt="thatlittleboy"/><br /><sub><b>thatlittleboy</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Athatlittleboy" title="Bug reports">🐛</a> <a href="#question-thatlittleboy" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/thomasjpfan"><img src="?s=100" width="100px;" alt="thomasjpfan"/><br /><sub><b>thomasjpfan</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=thomasjpfan" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tomjemmett"><img src="?s=100" width="100px;" alt="tomjemmett"/><br /><sub><b>tomjemmett</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Atomjemmett" title="Bug reports">🐛</a> <a href="#question-tomjemmett" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TomMonks"><img src="?s=100" width="100px;" alt="TomMonks"/><br /><sub><b>TomMonks</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=TomMonks" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/UGuntupalli"><img src="?s=100" width="100px;" alt="UGuntupalli"/><br /><sub><b>UGuntupalli</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3AUGuntupalli" title="Bug reports">🐛</a> <a href="#question-UGuntupalli" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/VPerrollaz"><img src="?s=100" width="100px;" alt="VPerrollaz"/><br /><sub><b>VPerrollaz</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3AVPerrollaz" title="Bug reports">🐛</a> <a href="#question-VPerrollaz" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xujiboy"><img src="?s=100" width="100px;" alt="xujiboy"/><br /><sub><b>xujiboy</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/issues?q=author%3Axujiboy" title="Bug reports">🐛</a> <a href="#question-xujiboy" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zbarry"><img src="?s=100" width="100px;" alt="zbarry"/><br /><sub><b>zbarry</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=zbarry" title="Code">💻</a> <a href="#talk-zbarry" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Zeroto521"><img src="?s=100" width="100px;" alt="Zeroto521"/><br /><sub><b>Zeroto521</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=Zeroto521" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zjpoh"><img src="?s=100" width="100px;" alt="zjpoh"/><br /><sub><b>zjpoh</b></sub></a><br /><a href="https://github.com/pyjanitor-devs/pyjanitor/commits?author=zjpoh" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

As of 11/05/2025, the test data for the chemistry submodule
is unavailable.
