pyjanitor
===========

Python implementation of the R package `janitor`_.

.. _janitor: https://github.com/sfirke/janitor

why janitor?
------------

It's well-explained in the R package documentation, but the high level summary is this:

- If all column names are **lowercase** and **underscored**, this makes it much, easier for data scientists to do their coding. No more dealing with crazy spaces and upper/lower-case mismatches. (I pull my hair out over this all the time!)
- If all empty column names and rows were removed prior to data analysis, a lot of hurt w.r.t. checking shapes could be eliminated!

installation
------------

`pyjanitor` is currently only installable from GitHub:

.. code-block:: bash

    pip install git+https://github.com/ericmjl/pyjanitor


functionality
-------------

As of 4 March 2018, this is a super new project. The continually updated list of functions are:

- Cleaning columns name
- Removing empty rows and columns
- Identifying duplicate entries
- Encoding columns as categorical
- Splitting your data into features and targets (for machine learning)
- Easily renaming individual columns

apis
----

The idea behind the API is two-fold:

- Copy the R package function names, but enable Pythonic use with method chaining or `pandas` piping.
- Add other utility functions that make it easy to do data cleaning in `pandas`.

As such, there are three ways to use the API. The first is the functional API.

.. code-block:: python

    from janitor import clean_names, remove_empty
    import pandas as pd

    df = pd.DataFrame(...)
    df = clean_names(df)
    df = remove_empty(df)


The second is the wrapped `pandas` DataFrame.

.. code-block:: python

  import janitor as jn
  import pandas as pd

  df = pd.DataFrame(...)
  df = jn.DataFrame(df)
  df.clean_names().remove_empty()... # method chaining possible

The third is to use the `pipe()` method.

.. code-block:: python

  from janitor import clean_names, remove_empty
  import pandas as pd

  df = pd.DataFrame(...)
  (df.pipe(clean_names)
     .pipe(remove_empty)
     .pipe(...))


feature requests
----------------

If you have a feature request, please post it as an issue on the GitHub repository issue tracker. Even better, put in a PR for it! I am more than happy to guide you through the codebase so that you can put in a contribution to the codebase.

Because `pyjanitor` is currently maintained by volunteers and has no fiscal support, any feature requests will be prioritized according to what maintainers encounter as a need in our day-to-day jobs. Please temper expectations accordingly.
