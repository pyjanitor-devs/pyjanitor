import os
import sys
from subprocess import CalledProcessError


import pandas as pd
import pytest

import janitor.io
import tempfile


def test_read_commandline(dataframe):
    """
    Test asserts that the dataframe made
        from the read_commandline function is
        identical to the test dataframe from
        which the .csv file was created.

    """
    # create a temporary .csv file from test data
    temp_dir = tempfile.gettempdir()

    dataframe.to_csv(f"{temp_dir}/dataframe.csv", index=0)

    # create a new dataframe from the temporary .csv using
    #   the cat command from the bash commandline

    if sys.platform in ["win32"]:
        # cat is not an operable command for Windows command line
        # "type" is a similar call
        df = janitor.io.read_commandline(f"type {temp_dir}\\dataframe.csv")
    else:
        df = janitor.io.read_commandline(f"cat {temp_dir}/dataframe.csv")

    # Make assertion that new dataframe created with read_commandline
    #   is equal to the test dataframe
    assert df.equals(dataframe)

    # clean up after the test
    os.unlink(f"{temp_dir}/dataframe.csv")


def test_read_commandline_bad_cmd(dataframe):
    """
    Test 1 raises a TypeError if read_commandline
        is given an input that is not a string.

    Test 2 raises a CalledProcessError if
        read_commandline is given a string
        which is not a valid bash command.

    Test 3 raises an EmptyDataError if
        read_commandline is given a string which
        is a valid bash command, however results
        in the shell not creating a dataframe.
    """
    temp_dir = tempfile.gettempdir()

    # create a temporary .csv file
    dataframe.to_csv(f"{temp_dir}/dataframe.csv")

    # Test 1
    with pytest.raises(TypeError):
        janitor.io.read_commandline(6)

    # Test 2
    with pytest.raises(CalledProcessError):
        janitor.io.read_commandline("bad command")

    # Test 3
    # windows does not support "cat" in commandline
    # "type" command must be used and it returns a different error
    cmd = "cat"

    ExpectedError = pd.errors.EmptyDataError
    if sys.platform in ["win32"]:
        cmd = "type"
        ExpectedError = CalledProcessError

    with pytest.raises(ExpectedError):
        janitor.io.read_commandline(cmd)

    # clean up after the tests
    os.unlink(f"{temp_dir}/dataframe.csv")
