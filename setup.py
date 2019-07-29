from setuptools import setup


def requirements():
    with open("requirements.txt", "r+") as f:
        return f.read()


long_description = """
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
and expressiveness of the ``dplyr`` package of the R statistical language
ecosystem, we have evolved ``pyjanitor`` into a language for expressing the
data processing DAG for ``pandas`` users.

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
- Experimental submodules for finance and biology
"""

setup(
    name="pyjanitor",
    version="0.18.0",
    description="Tools for cleaning pandas DataFrames",
    author="Eric J. Ma",
    author_email="ericmajinglong@gmail.com",
    url="https://github.com/ericmjl/pyjanitor",
    packages=["janitor"],
    install_requires=requirements(),
    python_requires=">=3.6",
    long_description=long_description,
    long_description_content_type="text/x-rst",
)
