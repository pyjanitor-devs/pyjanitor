============
Contributing
============

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

The following sections detail a variety of ways to contribute,
as well as how to get started.

.. note:: Please take a look at the `types of Contributions  <CONTRIBUTION_TYPES.html>`__  that we welcome, along with the guidelines.

Get Started!
------------

Ready to contribute? Here's how to setup ``pyjanitor`` for local development.

1. Fork the ``pyjanitor`` repo on GitHub: https://github.com/ericmjl/pyjanitor.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pyjanitor.git

3. Install your local copy into a conda environment. Assuming you have conda installed, this is how you set up your fork for local development::

    $ cd pyjanitor/
    $ make install

This also installs your new conda environment as a Jupyter-accessible kernel. To run correctly inside the environment, make sure you select the correct kernel from the top right corner of JupyterLab!

.. note :: If you are on Windows, you may need to install ``make`` before you can run the install. You can get it from ``conda-forge``::

    $ conda install -c defaults -c conda-forge make

    You should be able to run `make` now. The command above installs `make` to the `~/Anaconda3/Library/bin` directory.

.. note:: For PyCharm users, here are some `instructions <PYCHARM_USERS.html>`__  to get your Conda environment set up.

4. (Optional) Install the pre-commit hooks.

As of 29 October 2019, pre-commit hooks are available to run code formatting checks automagically
before git commits happen. If you did not have these installed before, run the following commands::

    # Update your environment to install pre-commit
    $ conda env update -f environment-dev.yml
    # Install pre-commit hooks
    $ pre-commit install-hooks

5. You should also be able to build the docs locally. To do this, from the main ``pyjanitor`` directory::

    $ make docs

The command above allows you to view the documentation locally in your browser. `Sphinx (a python documentation generator) <http://www.sphinx-doc.org/en/stable/usage/quickstart.html>`_ builds and renders the html for you, and you can find the html files by navigating to ``pyjanitor/docs/_build``, and then you can find the correct html file. To see the main pyjanitor page, open the ``index.html`` file.

.. note:: If you get any errors related to Importing modules when running `make docs`, first activate the development environment::

    $ source activate pyjanitor-dev

    or by typing::

    $ conda activate pyjanitor-dev


Sphinx uses `rst files (restructured text) <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ as its markdown language. To edit documentation, go to the rst file that corresponds to the html file you would like to edit. Make the changes directly in the rst file with the correct markup. Save the file, and rebuild the html pages using the same commands as above to see what your changes look like in html.

6. Submit an issue to the ``pyjanitor`` GitHub issue tracker describing your planned changes: https://github.com/ericmjl/pyjanitor/issues

This helps us keep track of who is working on what.

7. Create a branch for local development:

New features added to ``pyjanitor`` should be done in a new branch you have based off of the latest version of the ``dev`` branch. The protocol for ``pyjanitor`` branches for new development is that the ``master`` branch mirrors the current version of ``pyjanitor`` on PyPI, whereas the ``dev`` branch is for additional features for an eventual new official version of the package which might be deemed slightly less stable. Once more confident in the reliability/suitability for introducing a batch of changes into the official version, the ``dev`` branch is then merged into ``master`` and the PyPI package is subsequently updated.

To base a branch directly off of ``dev`` instead of ``master``, create a new one as follows::

    $ git checkout -b name-of-your-bugfix-or-feature dev

Now you can make your changes locally.

8. When you're done making changes, check that your changes are properly formatted and that all tests still pass::

    $ make check

If any of the checks fail, you can apply the checks individually (to save time):

* Automated code formatting: ``make style``
* Code styling problems check: ``make lint``
* Code unit testing: ``make test``

Styling problems must be resolved before the pull request can be accepted.

``make test`` runs all of ``pyjanitor``'s unit tests to probe for whether changes to the source code have potentially introduced bugs. These tests must also pass before the pull request is accepted.

All of these commands are available when you create the development environment.

When you run the test locally, the tests in ``chemistry.py``, ``biology.py``, ``spark.py`` are automatically skipped if you don't have the optional dependencies (e.g. ``rdkit``) installed.

9. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

10. Submit a pull request through the GitHub website. When you are picking out which branch to merge into, be sure to select ``dev`` (not ``master``).



Code Compatibility
------------------

pyjanitor supports Python 3.6+, so all contributed code must maintain this compatibility.


Tip
----

To run a subset of tests::

    $ py.test tests.test_functions


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
