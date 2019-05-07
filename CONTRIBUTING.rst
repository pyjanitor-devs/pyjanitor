.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ericmjl/pyjanitor/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Implementing a feature generally means writing a function that has the
following form:

.. code-block:: python

    @pf.register_dataframe_method
    def function(df, *args, **kwargs):
        # stuff done here
        return df

The function signature should take a pandas dataframe as the input, and return
a pandas dataframe as the output. Any manipulation to the dataframe should be
implemented inside the function.

This function should be implemented in `functions.py`, and should have a test
accompanying it in `tests/functions/test_<function_name_here>.py`.

When writing a test, the minimum acceptable test is an "example-based test".
Under ``janitor.testing_utils.fixtures``, you will find a suite of example
dataframes that can be imported and used in the test.

If you are more experienced with testing, you can use Hypothesis to
automatically generate example dataframes. We provide a number of
dataframe-generating strategies in ``janitor.testing_utils.strategies``.

If you're wondering why we don't have to implement the method chaining
portion, it's because we use pandas-flavor's `register_dataframe_method`,
which registers the function as a pandas dataframe method.

Write Documentation
~~~~~~~~~~~~~~~~~~~

``pyjanitor`` could always use more documentation, whether as part of the
official pyjanitor docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ericmjl/pyjanitor/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `pyjanitor` for local development.

1. Fork the `pyjanitor` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pyjanitor.git

3. Install your local copy into a conda environment. Assuming you have conda installed, this is how you set up your fork for local development::

    $ cd pyjanitor/
    $ conda env create -f environment-dev.yml
    $ python setup.py develop
    $ conda install -c conda-forge --yes --file requirements-dev.txt

4. Create a branch for local development::

New features added to pyjanitor should be done in a new branch you have based off of the latest version of the `dev` branch. The protocol for pyjanitor branches for new development is that the `master` branch mirrors the current version of pyjanitor on PyPI, whereas `dev` branch is for additional features for an eventual new official version of the package which might be deemed slightly less stable. Once more confident in the reliability / suitability for introducing a batch of changes into the official version, the `dev` branch is then merged into `master` and the PyPI package is subsequently updated.

To base a branch directly off of `dev` instead of `master`, create a new one as follows:

    $ git checkout -b name-of-your-bugfix-or-feature dev

   Now you can make your changes locally.

5. When you're done making changes, check that your changes are properly formatted and that all tests still pass::

    $ make lint
    $ make format
    $ py.test

   All of these commands are available when you create the development environment.

   When you run the test locally, the tests in ``chemistry.py`` are automatically skipped if you don't have the optional dependencies (e.g. ``rdkit``) installed.
        1. test_maccs_keys_fingerprint
        2. test_molecular_descriptors
        3. test_morgan_fingerprint_counts
        4. test_morgan_fingerprint_bits
        5. test_smiles2mol [None]
        6. test_smiles2mol [terminal]

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website where when you are picking out which branch to merge into, you select `dev` instead of `master`.


PyCharm Users
~~~~~~~~~~~~~
Currently, PyCharm doesn't support the generation of Conda environments via a
YAML file as prescribed above. To get around this issue you would simply set up
your environment as described above and within PyCharm point your interpreter
to the predefined conda environment.

1. Complete steps 1-3 under the Getting Started section.
2. Determine the location of the newly created conda environment::

    conda info --env

3. Open up the location of the cloned pyjanitor directory in PyCharm.
4. Navigate to the Preferences location.

    .. image:: /images/preferences.png

5. Navigate to the Project Interpreter tab.

    .. image:: /images/project_interpreter.png

6. Click the cog at the top right and select Add.

    .. image:: /images/click_add.png

7. Select Conda Environment on the left and select existing environment. Click
on the three dots and copy the location of your newly created conda environment
and append bin/python to the end of the path.

    .. image:: /images/add_env.png

Click Ok and you should be good to go!


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.

Tips
----

To run a subset of tests::

    $ py.test tests.test_functions
