pyjanitor
===========

.. image:: https://travis-ci.org/ericmjl/pyjanitor.svg?branch=master
    :target: https://travis-ci.org/ericmjl/pyjanitor

Python implementation of the R package `janitor`_, and more.

.. _janitor: https://github.com/sfirke/janitor

why janitor?
------------

Originally a port of the R package, ``pyjanitor`` has evolved from a set of convenient data cleaning routines into an experiment with the method chaining paradigm. 

Data preprocessing is best expressed as a directed acyclic graph (DAG) of actions taken on data. We take a base data file as the starting point, and perform actions on it, such as removing null/empty rows, replacing them with other values, adding/renaming/removing columns of data, filtering rows and more.

The `pandas` API has been invaluable for the Python data science ecosystem, and implements method chaining of a subset of methods as part of the API. For example, resetting indexes (``.reset_index()``), dropping null values (``.dropna()``), and more, are accomplished via the appropriate ``pd.DataFrame`` method calls.

Inspired by the R statistical language ecosystem, where consistent and good API design in the ``dplyr`` package enables end-users, who are not necessarily developers, to concisely express data processing code, I have evolved ``pyjanitor`` into a language for expressing the data processing DAG for ``pandas`` users.

To accomplish this, actions for which we would need to invoke imperative-style statements, can be replaced with method chains that allow one to read off the logical order of actions taken. Let us see the annotated example below. First, off, here's the textual description of a data cleaning pathway:

1. Create dataframe.
1. Delete one column.
1. Drop rows with empty values in two particular columns.
1. Rename another two columns.
1. Add a new column.

In ``pandas`` code, this would look as such:

.. code-block:: python

    df = pd.DataFrame(...)  # create a pandas DataFrame somehow.
    del df['column1']  # delete a column from the dataframe.
    df = df.dropna(subset=['column2', 'column3'])  # drop rows that have empty values in column 2 and 3.
    df = df.rename({'column2': 'unicorns', 'column3': 'dragons'})  # rename column2 and column3
    df['newcolumn'] = ['iterable', 'of', 'items']  # add a new column.

With ``pyjanitor``, we enable method chaining with method names that are *verbs*, which describe the action taken.

.. code-block:: python

    df = (
        pd.DataFrame(...)
        .remove_column('column1')
        .dropna(subset=['column2', 'column3'])
        .rename_column('column2', 'unicorns')
        .rename_column('column3', 'dragons')
        .add_column('newcolumn', ['iterable', 'of', 'items'])
    )

As such, the pyjanitor's etymology has a two-fold relationship to "cleanliness". Firstly, it's about extending Pandas with convenient data cleaning routines. Secondly, it's about providing a cleaner, method-chaining, verb-based API for common pandas routines.



installation
------------

`pyjanitor` is currently installable from PyPI:

.. code-block:: bash

    pip install pyjanitor

`pyjanitor` also can be installed by the conda package manager:

..code-block:: bash

    conda install pyjanitor -c conda-forge

functionality
-------------

Current functionality includes:

- Cleaning columns name (multi-indexes are possible!)
- Removing empty rows and columns
- Identifying duplicate entries
- Encoding columns as categorical
- Splitting your data into features and targets (for machine learning)
- Adding, removing, and renaming columns
- Coalesce multiple columns into a single column
- Convert excel date (serial format) into a Python datetime format
- Expand a single column that has delimited, categorical values into dummy-encoded variables

apis
----

The idea behind the API is two-fold:

- Copy the R package function names, but enable Pythonic use with method chaining or `pandas` piping.
- Add other utility functions that make it easy to do data cleaning/preprocessing in `pandas`.

As such, there are three ways to use the API. The first, and most strongly recommended one, is to use janitor's functions as if they were native to pandas.

.. code-block:: python

    import pandas as pd
    import janitor  # upon import, functions are registered as part of pandas.

    df = pd.DataFrame(...)
    df = df.clean_names().remove_empty()  # further method chaining possible.

The second is the functional API.

.. code-block:: python

    from janitor import clean_names, remove_empty
    import pandas as pd

    df = pd.DataFrame(...)
    df = clean_names(df)
    df = remove_empty(df)

The final way is to use the `pipe()` method.

.. code-block:: python

  from janitor import clean_names, remove_empty
  import pandas as pd

  df = pd.DataFrame(...)
  (df.pipe(clean_names)
     .pipe(remove_empty)
     .pipe(...))


contributing
------------

adding new functionality
~~~~~~~~~~~~~~~~~~~~~~~~

Keeping in mind the etymology of pyjanitor, contributing a new function to pyjanitor is a task that is not difficult at all.

define a function
^^^^^^^^^^^^^^^^^

First off, you will need to define the function that expresses the data processing/cleaning routine, such that it accepts a dataframe as the first argument, and returns a modified dataframe:

.. code-block:: python

    import pandas_flavor as pf

    @pf.register_dataframe_function
    def my_data_cleaning_function(df, arg1, arg2, ...):
        # Put data processing function here.
        return df

We use ``pandas_flavor`` to register the function natively on a ``pandas.DataFrame``.

add a test case 
^^^^^^^^^^^^^^^

Secondly, we ask that you contribute an test case, to ensure that it works as intended. This should go inside the ``tests/test_functions.py`` file.

feature requests
----------------

If you have a feature request, please post it as an issue on the GitHub repository issue tracker. Even better, put in a PR for it! I am more than happy to guide you through the codebase so that you can put in a contribution to the codebase.

Because `pyjanitor` is currently maintained by volunteers and has no fiscal support, any feature requests will be prioritized according to what maintainers encounter as a need in our day-to-day jobs. Please temper expectations accordingly.

