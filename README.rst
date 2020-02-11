=========
pyjanitor
=========

.. image:: https://dev.azure.com/ericmjl/Open%20Source%20Packages/_apis/build/status/ericmjl.pyjanitor?branchName=dev
    :target: https://dev.azure.com/ericmjl/Open%20Source%20Packages/_build/latest?definitionId=2&branchName=dev

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/ericmjl/pyjanitor/dev

.. pypi-doc

``pyjanitor`` is a Python implementation of the R package `janitor`_, and
provides a clean API for cleaning data.

.. _janitor: https://github.com/sfirke/janitor

Why janitor?
------------

Originally a port of the R package,
``pyjanitor`` has evolved from a set of convenient data cleaning routines
into an experiment with the `method chaining`__ paradigm.

.. _chaining: https://towardsdatascience.com/the-unreasonable-effectiveness-of-method-chaining-in-pandas-15c2109e3c69

__ chaining_

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
For example, resetting indexes (``.reset_index()``),
dropping null values (``.dropna()``), and more,
are accomplished via the appropriate ``pd.DataFrame`` method calls.

Inspired by the ease-of-use
and expressiveness of the ``dplyr`` package
of the R statistical language ecosystem,
we have evolved ``pyjanitor`` into a language
for expressing the data processing DAG for ``pandas`` users.

.. pypi-doc

To accomplish this,
actions for which we would need to invoke imperative-style statements,
can be replaced with method chains
that allow one to read off the logical order of actions taken.
Let us see the annotated example below.
First off, here is the textual description of a data cleaning pathway:

1. Create a ``DataFrame``.
2. Delete one column.
3. Drop rows with empty values in two particular columns.
4. Rename another two columns.
5. Add a new column.

Let's import some libraries
and begin with some sample data for this example :

.. code-block:: python

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


In ``pandas`` code, most users might type something like this:

.. code-block:: python

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
    # SalesMonth  Amazon  Facebook  Google
    # 0        Jan   180.0     400.0   450.0
    # 1        Feb   250.0     500.0   550.0
    # 3      April   500.0     675.0   800.0

Slightly more advanced users might take advantage of the functional API:

.. code-block:: python

    df = (
        pd.DataFrame(company_sales)
        .drop(columns="Company1")
        .dropna(subset=['Company2', 'Company3'])
        .rename(columns={"Company2": "Amazon", "Company3": "Facebook"})
        .assign(Google=[450.0, 550.0, 800.0])
        )

    # Output looks like this:
    # Out[15]:
    # SalesMonth  Amazon  Facebook  Google
    # 0        Jan   180.0     400.0   450.0
    # 1        Feb   250.0     500.0   550.0
    # 3      April   500.0     675.0   800.0



With ``pyjanitor``, we enable method chaining with method names
that are *verbs*, which describe the action taken.

.. code-block:: python


    df = (
        pd.DataFrame.from_dict(company_sales)
        .remove_columns(['Company1'])
        .dropna(subset=['Company2', 'Company3'])
        .rename_column('Company2', 'Amazon')
        .rename_column('Company3', 'Facebook')
        .add_column('Google', [450.0, 550.0, 800.0])
    )

    # Output looks like this:
    # Out[15]:
    # SalesMonth  Amazon  Facebook  Google
    # 0        Jan   180.0     400.0   450.0
    # 1        Feb   250.0     500.0   550.0
    # 3      April   500.0     675.0   800.0


As such,
pyjanitor's etymology has a two-fold relationship to "cleanliness".
Firstly, it's about extending Pandas with convenient data cleaning routines.
Secondly, it's about providing a cleaner, method-chaining, verb-based API
for common pandas routines.


Installation
------------

``pyjanitor`` is currently installable from PyPI:

.. code-block:: bash

    pip install pyjanitor


``pyjanitor`` also can be installed by the conda package manager:

.. code-block:: bash

    conda install pyjanitor -c conda-forge

``pyjanitor`` requires Python 3.6+.

.. pypi-doc

Functionality
-------------

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

.. pypi-doc

API
---

The idea behind the API is two-fold:

- Copy the R package function names,
  but enable Pythonic use with method chaining or `pandas` piping.
- Add other utility functions
  that make it easy to do data cleaning/preprocessing in `pandas`.

Continuing with the company_sales dataframe previously used:

.. code-block:: python

    import pandas as pd
    import numpy as np
    company_sales = {
        'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
        'Company1': [150.0, 200.0, 300.0, 400.0],
        'Company2': [180.0, 250.0, np.nan, 500.0],
        'Company3': [400.0, 500.0, 600.0, 675.0]
    }

As such, there are three ways to use the API.
The first, and most strongly recommended one, is to use ``pyjanitor``'s functions
as if they were native to pandas.

.. code-block:: python

    import janitor  # upon import, functions are registered as part of pandas.

    # This cleans the column names as well as removes any duplicate rows
    df = pd.DataFrame.from_dict(company_sales).clean_names().remove_empty()

The second is the functional API.

.. code-block:: python

    from janitor import clean_names, remove_empty

    df = pd.DataFrame.from_dict(company_sales)
    df = clean_names(df)
    df = remove_empty(df)

The final way is to use the `pipe()`_ method:

.. _pipe(): https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html

.. code-block:: python

    from janitor import clean_names, remove_empty
    df = (
        pd.DataFrame.from_dict(company_sales)
        .pipe(clean_names)
        .pipe(remove_empty)
    )

Contributing
------------

Follow `contribution docs
<https://ericmjl.github.io/pyjanitor/contributing.html>`_ for a full description of the process of contributing to ``pyjanitor``.

Adding new functionality
~~~~~~~~~~~~~~~~~~~~~~~~

Keeping in mind the etymology of pyjanitor,
contributing a new function to pyjanitor
is a task that is not difficult at all.

Define a function
^^^^^^^^^^^^^^^^^

First off, you will need to define the function
that expresses the data processing/cleaning routine,
such that it accepts a dataframe as the first argument,
and returns a modified dataframe:

.. code-block:: python

    import pandas_flavor as pf

    @pf.register_dataframe_method
    def my_data_cleaning_function(df, arg1, arg2, ...):
        # Put data processing function here.
        return df

We use `pandas_flavor`_ to register the function natively on a ``pandas.DataFrame``.

.. _pandas_flavor: https://github.com/Zsailer/pandas_flavor

Add a test case
^^^^^^^^^^^^^^^

Secondly, we ask that you contribute a test case,
to ensure that it works as intended.
Follow the `contribution`_ docs for further details.

.. _contribution: https://ericmjl.github.io/pyjanitor/contributing.html#unit-test-guidelines

Feature requests
~~~~~~~~~~~~~~~~

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

API Policy
~~~~~~~~~~

``pyjanitor`` only extends or aliases the ``pandas`` API
(and other dataframe APIs),
but will never fix or replace them.

Undesirable ``pandas`` behaviour should be reported upstream
in the ``pandas`` `issue tracker <https://github.com/pandas-dev/pandas/issues>`_.
We explicitly do not fix the ``pandas`` API.
If at some point the ``pandas`` devs
decide to take something from ``pyjanitor``
and internalize it as part of the official ``pandas`` API,
then we will deprecate it from ``pyjanitor``,
while acknowledging the original contributors' contribution
as part of the official deprecation record.


Credits
~~~~~~~

Test data for chemistry submodule can be found at `Predictive Toxicology`__ .

.. _predtox: https://www.predictive-toxicology.org/data/ntp/corrected_smiles.txt

__ predtox_
