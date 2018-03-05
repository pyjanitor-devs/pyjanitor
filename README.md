# `pyjanitor`

Python implementation of the R package [janitor](https://github.com/sfirke/janitor).

## why janitor?

It's well-explained in the R package documentation, but the high level summary is this:

1. If all column names are **lowercase** and **underscored**, this makes it much, easier for data scientists to do their coding. No more dealing with crazy spaces and upper/lower-case mismatches. (I pull my hair out over this all the time!)

## installation

`pyjanitor` is currently only installable from GitHub:

```bash
pip install git+https://github.com/ericmjl/pyjanitor
```

## functionality

As of 4 March 2018, this is a super new project. The continually updated list of functions are:

- Cleaning columns name (`clean_names(df)` or `df.clean_names()`)

## apis

There are three ways to use the API. The first is the functional API.

```python
from janitor import clean_names
import pandas as pd

df = pd.DataFrame(...)
df = clean_names(df)
```

The second is the wrapped `pandas` DataFrame.

```python
from janitor import clean_names
import pandas as pd

df = pd.DataFrame(...)
df = janitor.DataFrame(df)
df.clean_names().... # method chaining possible
```

The third is to use the `pipe()` method.

```python
from janitor import clean_names
import pandas as pd

df = pd.DataFrame(...)
(df.pipe(clean_names)
   .pipe(...))
```

## feature requests

If you have a feature request, please post it as an issue on the GitHub repository issue tracker. Even better, put in a PR for it! I am more than happy to guide you through the codebase so that you can put in a contribution to the codebase.

Because `pyjanitor` is currently maintained by volunteers and has no fiscal support, any feature requests will be prioritized according to what maintainers encounter as a need in our day-to-day jobs. Please temper expectations accordingly.
