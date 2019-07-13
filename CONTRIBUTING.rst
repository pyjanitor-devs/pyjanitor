.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Write Documentation
~~~~~~~~~~~~~~~~~~~

``pyjanitor`` could always use more documentation, whether as part of the
official pyjanitor docs, in docstrings, the examples gallery.

During sprints, we require newcomers to the project to first contribute a
documentation fix before contributing a code fix. Doing so has manyfold benefits:

1. You become familiar with the project by first reading through the docs.
2. Your documentation contribution will be a pain point that you have full context on.
3. Your contribution will be impactful because documentation is the project's front-facing interface.
4. Your first contribution will be simpler, because you won't have to wrestle with build systems.
5. You can choose between getting set up locally first (recommended), or instead directly making edits on the GitHub web UI (also not a problem).
6. Every newcomer is equal in our eyes, and it's the most egalitarian way to get started, regardless of experience.

Remote contributors outside of sprints, and prior contributors who are joining
us at the sprints need not adhere to this rule, as a good prior
assumption is that you are a motivated user of the library already. If you have
made a prior PR to the library, we would like to encourage you to mentor newcomers
in lieu of coding contributions.

Documentation can come in many forms. For example, you might want to contribute:

- Fixes for a typographical, grammatical, or spelling error.
- Changes for a docstring that was unclear.
- Clarifications for installation/setup instructions that are unclear.
- Rephrasing of a sentence/phrase/word choice that didn't make sense.
- New example/tutorial notebooks using the library.
- Reworking of existing tutorial notebooks with better code style.

In particular, contributing new tutorial notebooks and improving the clarity of existing ones
are a great ways to get familiar with the library and find pain points that you can
propose as fixes or enhancements to the library.

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ericmjl/pyjanitor/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with ``bug``
and ``available to hack on`` is open to whoever wants to implement it.

Do be sure to claim the issue for yourself by indicating, "I would like to
work on this issue." If you would like to discuss it further before going forward,
you are more than welcome to discuss on the GitHub issue tracker.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with ``enhancement``
and ``available to hack on`` are open to whoever wants to implement it.

Implementing a feature generally means writing a function that has the
following form:

.. code-block:: python

    @pf.register_dataframe_method
    def function(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Very detailed docstring here. Look to other functions for examples.
        """
        # stuff done here
        return df

The function signature should take a pandas dataframe as the input, and return
a pandas ``DataFrame`` as the output. Any manipulation to the dataframe should be
implemented inside the function. The standard functionality of ``pyjanitor`` methods that we're moving towards is to not modify the input ``DataFrame``.

This function should be implemented in ``functions.py``, and should have a test
accompanying it in ``tests/functions/test_<function_name_here>.py``.

When writing a test, the minimum acceptable test is an "example-based test".
Under ``tests/conf.py``, you will find a suite of example dataframes that can be
imported and used in the test.

If you are more experienced with testing, you can use Hypothesis to
automatically generate example dataframes. We provide a number of
dataframe-generating strategies in ``janitor.testing_utils.strategies``.

We may go back-and-forth a few times on the docstring. The docstring is a very
important part of developer documentation, and we may want much more detail than
you are used to providing. This is for maintenance purposes; more often than not,
contributions from new contributors will end up being maintained by the
maintainers, and hence we would err on the side of providing more contextual
information than less, especially on design choices made.

If you're wondering why we don't have to implement the method chaining
portion, it's because we use pandas-flavor's ``register_dataframe_method``,
which registers the function as a pandas dataframe method.

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
    $ conda activate pyjanitor-dev
    $ python setup.py develop
    $ conda install -c conda-forge --yes --file requirements-dev.txt

4. Build the documentation locally, from the main `pyjanitor` directory:
    $ cd docs
    $ make html

    **Note:** If you get an error when building docs for a Jupyter notebook saying that the module `janitor` is not available (the specific error is `ModuleNotFoundError: No module named 'janitor'`), install an `ipykernel` in the current environment with the following steps:
        $ python3 -m ipykernel install --name pyjanitor-dev --user

    This should allow Jupyter to run correctly inside the environment and you should be able to build the docs locally

5. Create a branch for local development:

New features added to ``pyjanitor`` should be done in a new branch you have based off of the latest version of the `dev` branch. The protocol for ``pyjanitor`` branches for new development is that the ``master`` branch mirrors the current version of ``pyjanitor`` on PyPI, whereas ``dev`` branch is for additional features for an eventual new official version of the package which might be deemed slightly less stable. Once more confident in the reliability / suitability for introducing a batch of changes into the official version, the ``dev`` branch is then merged into ``master`` and the PyPI package is subsequently updated.

To base a branch directly off of ``dev`` instead of ``master``, create a new one as follows:

    $ git checkout -b name-of-your-bugfix-or-feature dev

Now you can make your changes locally.

6. When you're done making changes, check that your changes are properly formatted and that all tests still pass::

    $ make format
    $ make lint
    $ py.test

``format`` will apply code style formatting, ``lint`` checks for styling problems (which must be resolved before the PR can be accepted), and ``py.test`` runs all of ``pyjanitor``'s unit tests to probe for whether changes to the source code have potentially introduced bugs. These tests must also pass before the PR is accepted.

All of these commands are available when you create the development environment.

When you run the test locally, the tests in ``chemistry.py`` & ``biology.py`` are automatically skipped if you don't have the optional dependencies (e.g. ``rdkit``) installed.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website where when you are picking out which branch to merge into, you select ``dev`` instead of ``master``.


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
