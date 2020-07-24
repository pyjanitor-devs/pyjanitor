.. _contribution_types:

Types of Contributions
=======================

Write Documentation
--------------------

``pyjanitor`` could always use more documentation,
whether as part of the official pyjanitor docs, in docstrings, or the examples gallery.

During sprints, we require newcomers to the project to
first contribute a documentation fix before contributing a code fix.
Doing so has numerous benefits:

1. You become familiar with the project by first reading through the docs.
2. Your documentation contribution will be a pain point that you have full context on.
3. Your contribution will be impactful because documentation is the project's front-facing interface.
4. Your first contribution will be simpler, because you won't have to wrestle with build systems.
5. You can choose between getting set up locally first (recommended),
   or instead directly making edits on the GitHub web UI (also not a problem).
6. Every newcomer is equal in our eyes, and it's the most egalitarian way to get started (regardless of experience).

Remote contributors outside of sprints and prior contributors
who are joining us at the sprints need not adhere to this rule,
as a good prior assumption is that you are a motivated user of the library already.
If you have made a prior pull request to the library,
we would like to encourage you to mentor newcomers in lieu of coding contributions.

Documentation can come in many forms. For example, you might want to contribute:

- Fixes for a typographical, grammatical, or spelling error.
- Changes for a docstring that was unclear.
- Clarifications for installation/setup instructions that are unclear.
- Corrections to a sentence/phrase/word choice that didn't make sense.
- New example/tutorial notebooks using the library.
- Edits to existing tutorial notebooks with better code style.

In particular, contributing new tutorial notebooks and
improving the clarity of existing ones are great ways to
get familiar with the library and find pain points that
you can propose as fixes or enhancements to the library.

Report Bugs
------------

Report bugs at https://github.com/ericmjl/pyjanitor/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
---------

Look through the GitHub issues for bugs.
Anything tagged with ``bug`` and ``available to hack on`` is open to
whoever wants to implement it.

Do be sure to claim the issue for yourself by indicating,
"I would like to work on this issue."
If you would like to discuss it further before going forward,
you are more than welcome to discuss on the GitHub issue tracker.

Implement Features
-------------------

Look through the GitHub issues for features. Anything tagged with ``enhancement``
and ``available to hack on`` are open to whoever wants to implement it.


Key Ingredients
^^^^^^^^^^^^^^^^^

To ensure ``pyjanitor``'s code quality and long-term maintainability
(both aspects lay the groundwork for growing community engagement),
we ask our contributors to

1. Include good documentation
(:ref:`docstring_guidelines`)

2. Align code patterns with existing ``pyjanitor`` functions
(:ref:`code_pattern_guidelines`)

3. Set up and carry out unit tests
(:ref:`unit_test_guidelines`)

These three elements consitute the minimum requirement for feature contributions.
In the next three sections, we will dive deeper into these requirements.

.. _docstring_guidelines:

Docstring Guidelines
^^^^^^^^^^^^^^^^^^^^

For a python function, its docstring is a string that
documents what the function does and what the inputs/outputs should be.
Docstring conventions have `various flavors <http://www.sphinx-doc.org/en/1.8/usage/extensions/napoleon.html#confval-napoleon_numpy_docstring>`_.
In ``pyjanitor``, we use the `Sphinx style docstring <https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html>`_ that
is built on top of the `reStructuredText (reST) markup <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html#headings>`_ language.

reST syntax, like python, is sensitive to **indentation**,
and all the text units (e.g., heading, paragraph, and code block) must be
separated by **blank lines** from each other.
Proper rendering of the web docs you are browsing through right now
relies on our contributors' adherence to these reST syntax rules.

Below is a docstring example from ``pyjanitor``'s ``rename_columns`` function.
For illustration purposes,
we also add annotations (surrounded by square brackets) to
highlight all the key sections
(Note: These annotations are NOT part of the docstring).

.. code-block:: python

    @pf.register_dataframe_method
    def rename_columns(df: pd.DataFrame, new_column_names: Dict) -> pd.DataFrame:
        """
        [short summary]
        Rename columns in place.

        [examples]
        Functional usage example:

        .. code-block:: python

            df = rename_columns({"old_column_name": "new_column_name"})

        Method chaining example:

        .. code-block:: python

            import pandas as pd
            import janitor
            df = pd.DataFrame(...).rename_columns({"old_column_name":
            "new_column_name"})

        [notes]
        This is just syntactic sugar/a convenience function for renaming one column
        at a time. If you are convinced that there are multiple columns in need of
        changing, then use the :py:meth:`pandas.DataFrame.rename` method.

        [parameters]
        :param df: The pandas DataFrame object.
        :param new_column_names: A dictionary of old and new column names.
        [returns]
        :returns: A pandas DataFrame with renamed columns.
        """
        check_column(df, list(new_column_names.keys()))
        return df.rename(columns=new_column_names)

Let's expand on this example:

Docstrings should always be surrounded by **triple double quotes**. i.e.,

.. code-block:: python

    """
    I am a docstring

    I can take up several lines
    """

The key sections of a docstring are:

1. *short summary*: A concise one-line summary about what the function does.
It should NOT include variable names or function names.

.. code-block:: python

    """
    Rename columns in place.

    """

2. *examples*: Examples play an important role in
ensuring user-friendliness of the API.
For ``pyjanitor``, ideal examples should demonstrate both the functional and
the method chaining usages of the function.

Each usage example should have a short text description and
a code block (marked by the `.. code-block:: python` `reST directive <http://docutils.sourceforge.net/docs/ref/rst/directives.html#code>`_).
The text description, the `.. code-block:: python` directive, and
the content of the code block must be separated by
**blank lines** from one another
(see :ref:`docstring_guidelines`).

.. code-block:: python

    """
    Functional usage example:

    .. code-block:: python

        df = rename_columns({"old_column_name": "new_column_name"})

    """

3. *notes*: Notes should provide additional information that
users and maintainers should be aware of. e.g.,

.. code-block:: python

    """
    This is just syntactic sugar/a convenience function for renaming one column
    at a time. If you are convinced that there are multiple columns in need of
    changing, then use the :py:meth:`pandas.DataFrame.rename` method.
    """

4. *parameters*: Itemized description of the function's arguments and
keyword arguments. Each item should follow the format of

``:param <arg name>: <arg description>, default to <default value>``.

.. code-block:: python

    """
    :param df: The pandas DataFrame object.
    :param new_column_names: A dictionary of old and new column names.
    """

5. *returns*: Itemized description of returned values.
Each item should follow the format of

``:returns: <return description>.``

For ``pyjanitor`` functions,
the returned values typically are pandas ``DataFrame``. e.g.,

.. code-block:: python

    """
    :returns: A pandas DataFrame with renamed columns.
    """

.. _docstring_notes:

.. note::

   We may go back-and-forth a few times on the docstring.
   The docstring is a particularly important part of developer documentation;
   therefore, we may want much more detail than you are used to providing.
   This is for maintenance purposes:
   Contributions from new contributors frequently end up being maintained by
   the maintainers, and hence we would err on the side of
   providing more contextual information than less,
   especially regarding design choices.

.. _code_pattern_guidelines:

Code Pattern Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^

Let's continue using the same code example and
shift our focus to the code patterns:

.. code-block:: python
   :linenos:
   :emphasize-lines: 1,2,7

    @pf.register_dataframe_method
    def rename_columns(df: pd.DataFrame, new_column_names: Dict) -> pd.DataFrame:
        """
        Docstring as shown above; Omitted here
        """
        check_column(df, list(new_column_names.keys()))
        return df.rename(columns=new_column_names)

The three highlighted lines (line 1, 2, and 7) constitute
the code pattern that
is frequently used in ``pyjanitor`` implementations:

* Line 1: ``@pf.register_dataframe_method``

This `decorator <https://realpython.com/primer-on-python-decorators/>`_ comes from
`pandas-flavor <https://pypi.org/project/pandas-flavor/>`_.
It is the "magic" that allows ``pyjanitor`` code to
use just one set of implementations (like this `rename_columns` function) for
both the functional and the method chaining usages of the API.
In your new feature or feature enhancement,
you are highly likely to start your function with this decorator line
(or see it in the function that you are enhancing).

* Line 2 and 7: The *dataframe in, dataframe out* function signature

.. code-block:: python

    def rename_columns(df: pd.DataFrame, new_column_names: Dict) -> pd.DataFrame:
        ...
        return df.rename(columns=new_column_names)

The function signature should take a pandas ``DataFrame`` as
the input and return a pandas ``DataFrame`` as the output.
Any manipulations to the dataframe should be implemented inside the function.
The standard functionality of pyjanitor methods that
we are moving towards is to mutate the input ``DataFrame`` itself.

.. note::

   ``pyjanitor`` code uses `type hints <https://docs.python.org/3/library/typing.html>`_
   in function definitions.
   Even though Python--a dynamic typing language--by default does not do
   any type checking at runtime,
   adding type hints helps simplify code documentation
   (otherwise we would need to use docstrings to
   document argument types and return types) and over time,
   could help build and maintain a clearner code architecture
   (forces us to think about types as we write the code).
   Moreover, with type hints,
   type checkers such as `Mypy <http://mypy-lang.org/>`_ could be used as
   part of the code testing.
   For these reasons, we ask our contributors to use type hints.

.. _unit_test_guidelines:

Unit Test Guidelines
^^^^^^^^^^^^^^^^^^^^^

Unit tests form the basic immune system for a code base.
For this reason, all ``pyjanitor`` features,
regardless of being a brand-new function or an enhancement to an existing function,
should have accompanying tests.

``pyjanitor`` uses the `pytest <https://docs.pytest.org/en/latest/index.html>`_ framework
to carry out unit tests.
Any code contributions should at least have `example-based tests <https://www.freecodecamp.org/news/intro-to-property-based-testing-in-python-6321e0c2f8b/>`_.
Contributors who have experiences in `property-based tests <https://www.freecodecamp.org/news/intro-to-property-based-testing-in-python-6321e0c2f8b/>`_
can use the `Hypothesis <https://hypothesis.readthedocs.io/en/latest/>`_ framework to
automatically generate example dataframes
(We provide a number of dataframe-generating strategies in `janitor.testing_utils.strategies`).

But *where should we put the tests?* To answer this question,
let's look at the structure of the current ``pyjanitor`` codes:

.. code-block:: bash
   :emphasize-lines: 12

    pyjanitor/janitor
    ├── __init__.py
    ├── biology.py
    ├── chemistry.py
    ├── errors.py
    ├── finance.py
    ├── functions.py
    ├── io.py
    ├── testing_utils
    │   ├── __init__.py
    │   ├── date_data.py
    │   └── strategies.py  # contains dataframe-generating strategies
    └── utils.py

In this tree diagram, all the top level ``*.py`` files are
the **modules** of the ``pyjanitor`` library.
The accompanying tests files are in the ``pyjanitor/tests`` directory and
has a structure as shown below:

.. code-block:: bash
   :emphasize-lines: 8

    pyjanitor/tests
    ├── biology
    │   └── test_join_fasta.py
    ├── chemistry
    │   ├── test_maccs_keys_fingerprint.py
    │   ├── test_molecular_descriptors.py
    │   ├── ...
    ├── conftest.py  # contains test dataframes
    ├── finance
    │   └── test_convert_currency.py
    ├── functions
    │   ├── test_add_column.py
    │   ├── test_add_columns.py
    │   ├── ...
    ├── io
    │   └── test_read_csvs.py
    ├── test_data
    │   ├── corrected_smiles.txt
    │   ├── sequences.fasta
    │   └── sequences.tsv
    ├── test_df_registration.py
    └── utils
        ├── test_check_column.py
        ├── test_clean_accounting_column.py
        ├── ...

You can see that the naming and organization convention for unit tests:
That is, unit tests for a **function** inside a **module** should be in

.. code-block:: bash

    tests/<module_name>/test_<function_name>.py

The highlighted ``conftest.py`` contains **test dataframes** that
are implemented as `pytest fixtures <http://doc.pytest.org/en/latest/fixture.html>`_.

To make this more concrete, let's return to the ``rename_columns`` example.

1. *Where is the test data?*

The test ``dataframe`` in ``pyjanitor/tests/confest.py``:

.. code-block:: python
   :emphasize-lines: 5

    import pandas as pd
    import pytest


    @pytest.fixture
    def dataframe():
        data = {
            "a": [1, 2, 3] * 3,
            "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
            "decorated-elephant": [1, 2, 3] * 3,
            "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
            "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
        }
        df = pd.DataFrame(data)
        return df

The highlighted pytest `decorator <https://realpython.com/primer-on-python-decorators/>`_
``@pytest.fixture`` turns ``dataframe`` from a regular function to a pytest fixture.
This then allows us to
**inject** the ``dataframe`` into our test function as shown below.

2. *How should the test look like?*

Now let's look at the test for the ``rename_columns``
(in ``pyjanitor/janitor/functions.py``)

.. code-block:: python
   :linenos:
   :emphasize-lines: 4,5,9-12

    import pytest


    @pytest.mark.functions
    def test_rename_columns(dataframe):
        df = dataframe.clean_names().rename_columns(
            {"a": "index", "bell_chart": "chart"}
        )
        assert set(df.columns) == set(
            ["index", "chart", "decorated_elephant", "animals@#$%^", "cities"]
        )
        assert "a" not in set(df.columns)


The highlighted lines denote the pattern for testing:

* Line 4: ``@pytest.mark.functions``

This decorator is a `custom pytest mark <http://doc.pytest.org/en/latest/example/markers.html>`_.
You will often see it at the top of each test function following the convention of
``@pytest.mark.<module_name>``.
This mark allows us to restrict a test run to only run tests marked with `<module_name>`.
For example, we can run all the test with the ``pytest.mark.functions`` mark:

.. code-block:: bash

    # run `pytest -h` in your terminal to check all available options
    $ pytest -v -m functions --cov

Or conversely, we can run all the tests *except* the ``pytest.mark.functions`` ones:

.. code-block:: bash

    # run `pytest -h` in your terminal to check all available options
    $ pytest -v -m "not functions" --cov

* Line 5: ``dataframe`` injection

Upon test function definition, the test ``dataframe`` fixture is injected.

* Line 9-12: assertions for example-based tests

After using the function in the test (line 6-8),
we use ``assert`` statements to carry out example-based tests using fixed inputs
and fixed, expected outputs.

3. *How do we run the test?*

* To run only your test:

.. code-block:: bash

    # In `pyjanitor/tests/<module_name>`
    $ pytest -v test_<function_name>.py --cov

* To run all the tests:

.. code-block:: bash

    # Under `pyjanitor` top level directory (i.e., where `Makefile` is)
    $ make test

This is the basic structure of unit tests.
For your own tests, you can either use existing test data in ``conftest.py``,
or add your own test data to that file by following the same fixture pattern.

Submit Feedback
-----------------

The best way to send feedback is to file an issue at https://github.com/ericmjl/pyjanitor/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)
