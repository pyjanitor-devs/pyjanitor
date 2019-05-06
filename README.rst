pyjanitor
=========

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

1. Create a dataframe.
2. Delete one column.
3. Drop rows with empty values in two particular columns.
4. Rename another two columns.
5. Add a new column.

Let's import some libraries and begin with some sample data for this example :

.. code-block:: python

    # Libraries
    import numpy as np
    import pandas as pd
    import janitor

    # Sample Data curated for this example
    company_sales = {'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
                     'Company1': [150.0, 200.0, 300.0, 400.0],
                     'Company2': [180.0, 250.0, np.nan, 500.0],
                     'Company3': [400.0, 500.0, 600.0, 675.0]}

In ``pandas`` code, this would look as such:

.. code-block:: python

    # The Pandas Way
    df = pd.DataFrame.from_dict(company_sales) # create a pandas DataFrame from the company_sales dictionary
    del df['Company1']  # delete a column from the DataFrame. Say 'Company1'
    df = df.dropna(subset=['Company2', 'Company3'])  # drop rows that have empty values in columns 'Company2' and 'Company3'
    df = df.rename({'Company2': 'Amazon', 'Company3': 'Facebook'}, axis=1)  # rename 'Company2' to 'Amazon' and 'Company3' to 'Facebook'
    df['Google'] = [450.0, 550.0, 800.0]  # Let's add some data for another company. Say 'Google'

With ``pyjanitor``, we enable method chaining with method names that are *verbs*, which describe the action taken.

.. code-block:: python

    # The PyJanitor Way
    df = (
        pd.DataFrame.from_dict(company_sales)
        .remove_columns(['Company1'])
        .dropna(subset=['Company2', 'Company3'])
        .rename_column('Company2', 'Amazon')
        .rename_column('Company3', 'Facebook')
        .add_column('Google', [450.0, 550.0, 800.0])
    )

As such, the pyjanitor's etymology has a two-fold relationship to "cleanliness". Firstly, it's about extending Pandas with convenient data cleaning routines. Secondly, it's about providing a cleaner, method-chaining, verb-based API for common pandas routines.


installation
------------

``pyjanitor`` is currently installable from PyPI:

.. code-block:: bash

    pip install pyjanitor


``pyjanitor`` also can be installed by the conda package manager:

.. code-block:: bash

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
- Date conversions (from matlab, excel, unix) to Python datetime format
- Expand a single column that has delimited, categorical values into dummy-encoded variables
- Concatenating and deconcatenating columns, based on a delimiter
- Syntactic sugar for filtering the dataframe based on queries on a column
- Experimental submodules for finance and biology

apis
----

The idea behind the API is two-fold:

- Copy the R package function names, but enable Pythonic use with method chaining or `pandas` piping.
- Add other utility functions that make it easy to do data cleaning/preprocessing in `pandas`.

As such, there are three ways to use the API. The first, and most strongly recommended one, is to use janitor's functions as if they were native to pandas.

Continuing with the company_sales dataframe previously used
.. code-block:: python
	import pandas as pd
	import numpy as np
    
	company_sales = {'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
					 'Company1': [150.0, 200.0, 300.0, 400.0],
					 'Company2': [180.0, 250.0, np.nan, 500.0],
					 'Company3': [400.0, 500.0, 600.0, 675.0]}
	

.. code-block:: python
	import janitor  # upon import, functions are registered as part of pandas.
	
    df = pd.DataFrame.from_dict(company_sales)
    df = df.clean_names().remove_empty()  # further method chaining possible.

The second is the functional API.

.. code-block:: python

    from janitor import clean_names, remove_empty
    import pandas as pd

    df = pd.DataFrame.from_dict(company_sales)
    df = clean_names(df)
    df = remove_empty(df)

The final way is to use the `pipe()` method.

.. code-block:: python

  from janitor import clean_names, remove_empty
  import pandas as pd

  ddf = pd.DataFrame.from_dict(company_sales)
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

    @pf.register_dataframe_method
    def my_data_cleaning_function(df, arg1, arg2, ...):
        # Put data processing function here.
        return df

We use `pandas_flavor`_ to register the function natively on a ``pandas.DataFrame``.

.. _pandas_flavor: https://github.com/Zsailer/pandas_flavor

add a test case
^^^^^^^^^^^^^^^

Secondly, we ask that you contribute an test case, to ensure that it works as intended. This should go inside the ``tests/test_functions.py`` file.

feature requests
~~~~~~~~~~~~~~~~

If you have a feature request, please post it as an issue on the GitHub repository issue tracker. Even better, put in a PR for it! I am more than happy to guide you through the codebase so that you can put in a contribution to the codebase.

Because `pyjanitor` is currently maintained by volunteers and has no fiscal support, any feature requests will be prioritized according to what maintainers encounter as a need in our day-to-day jobs. Please temper expectations accordingly.

credits
~~~~~~~

Test data for chemistry submodule can be found at `Predictive Toxicology`__ .

.. _predtox: https://www.predictive-toxicology.org/data/ntp/corrected_smiles.txt

__ predtox_
