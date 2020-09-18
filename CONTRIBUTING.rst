============
Contributing
============

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

The following sections detail a variety of ways to contribute,
as well as how to get started.

.. note:: Please take a look at the `types of Contributions  <CONTRIBUTION_TYPES.html>`__  that we welcome,
    along with the guidelines.

Get Started!
------------

Ready to contribute? Here's how to setup ``pyjanitor`` for local development.

Development Containers with VSCode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of 29 May 2020, development containers are supported!
For me, this is the preferred way to get you started up and running,
as I have experience with it.
Here, you are provided with a pre-built and clean
runtime and build environment.
You don't have to wrestle with conda wait times if you don't want to!

To get started:

1. Fork the repository.
2. Ensure you have Docker running on your local machine.
3. Ensure you have VSCode running on your local machine.
4. In Visual Studio Code,
    click on the quick actions Status Bar item in the lower left corner.
5. Then select "Remote Containers: Clone Repository In Container Volume".
6. Enter in the URL of your fork of ``pyjanitor``.

VSCode will pull down the prebuilt Docker container,
git clone the repository for you inside an isolated Docker volume,
and mount the repository directory inside your Docker container.

Follow best practices to submit a pull request by making a feature branch.
Now, hack away, and submit in your pull request!

You shouln't be able to access the cloned repo
on your local hard drive.
If you do want local access, then clone the repo locally first
before selecting "Remote Containers: Open Folder In Container".

If you find something is broken because a utility is missing in the container,
submit a PR with the appropriate build command inserted in the Dockerfile.
Care has been taken to document what each step does,
so please read the in-line documentation in the Dockerfile carefully.

Manual Setup
~~~~~~~~~~~~

1. Fork the ``pyjanitor`` repo on GitHub: https://github.com/ericmjl/pyjanitor.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pyjanitor.git

3. Install your local copy into a conda environment.
    Assuming you have conda installed,
    this is how you set up your fork for local development::

    $ cd pyjanitor/
    $ make install

This also installs your new conda environment as a Jupyter-accessible kernel.
To run correctly inside the environment,
make sure you select the correct kernel from the top right corner of JupyterLab!

.. note:: If you are on Windows,
    you may need to install ``make`` before you can run the install.
    You can get it from ``conda-forge``::

    $ conda install -c defaults -c conda-forge make

    You should be able to run `make` now. The command above installs `make` to the `~/Anaconda3/Library/bin` directory.

.. note:: For PyCharm users,
    here are some `instructions <PYCHARM_USERS.html>`__  to get your Conda environment set up.

4. (Optional) Install the pre-commit hooks.

As of 29 October 2019,
``pre-commit`` hooks are available to run code formatting checks automagically before git commits happen.
If you did not have these installed before,
run the following commands::

    # Update your environment to install pre-commit
    $ conda env update -f environment-dev.yml
    # Install pre-commit hooks
    $ pre-commit install-hooks

5. You should also be able to build the docs locally.
    To do this, from the main ``pyjanitor`` directory::

    $ make docs

The command above allows you to view the documentation locally in your browser.
`Sphinx (a python documentation generator) <http://www.sphinx-doc.org/en/stable/usage/quickstart.html>`_ builds and renders the html for you,
and you can find the html files by navigating to ``pyjanitor/docs/_build``,
and then you can find the correct html file.
To see the main pyjanitor page,
open the ``index.html`` file.

.. note:: If you get any errors related to Importing modules when running ``make docs``,
    first activate the development environment::

    $ source activate pyjanitor-dev

    or by typing::

    $ conda activate pyjanitor-dev


Sphinx uses `rst files (restructured text) <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ as its markdown language.
To edit documentation,
go to the rst file that corresponds to the html file you would like to edit.
Make the changes directly in the rst file with the correct markup.
Save the file and rebuild the html pages using the same commands as above to see what your changes look like in html.

6. Submit an issue to the ``pyjanitor`` GitHub issue tracker describing your planned changes: https://github.com/ericmjl/pyjanitor/issues

This helps us keep track of who is working on what.

7. Create a branch for local development:

New features added to ``pyjanitor`` should be done in a new branch you have based off the latest version of the ``dev`` branch.

Releases are made off the ``dev`` branch.

To create a new branch::

    $ git checkout -b name-of-your-bugfix-or-feature dev

Now you can make your changes locally.

8. When you're done making changes,
    check that your changes are properly formatted and that all tests still pass::

    $ make check

If any of the checks fail, you can apply the checks individually (to save time):

* Automated code formatting: ``make style``
* Code styling problems check: ``make lint``
* Code unit testing: ``make test``

Styling problems must be resolved before the pull request can be accepted.

``make test`` runs all ``pyjanitor``'s unit tests to probe whether changes to the source code have potentially introduced bugs.
These tests must also pass before the pull request is accepted.

All these commands are available when you create the development environment.

When you run the test locally,
the tests in ``chemistry.py``, ``biology.py``, ``spark.py`` are automatically skipped if you don't have the optional dependencies (e.g. ``rdkit``) installed.

9. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

10. Submit a pull request through the GitHub website.
    When you are picking out which branch to merge into,
    be sure to select ``dev`` (not ``master``).

11. Let the continuous integration (CI) system on Azure Pipelines check your code.

If there are any issues, the pipeline will fail out.
We check for code style, docstring coverage, test coverage, and doc discovery.
If you're comfortable looking at the pipeline logs, feel free to do so;
they are open to all to view.
Otherwise, one of the dev team members can help you with reviewing the code checks.

Code Compatibility
------------------

pyjanitor supports Python 3.6+,
so all contributed code must maintain this compatibility.

Tips
----

To run a subset of tests::

    $ py.test tests.test_functions
