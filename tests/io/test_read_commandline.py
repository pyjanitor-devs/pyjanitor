import os
from subprocess import CalledProcessError

import pandas as pd
import pytest

import janitor.io


def test_read_commandline(dataframe):
    """
    Test asserts that the dataframe made
        from the read_commandline function is
        identical to the test dataframe from
        which the .csv file was created.

    """
    # create a temporary .csv file from test data
    dataframe.to_csv("/tmp/dataframe.csv", index=0)

    # create a new dataframe from the temporary .csv using
    #   the cat command from the bash commandline
    df = janitor.io.read_commandline("cat /tmp/dataframe.csv")

    # Make assertion that new dataframe created with read_commandline
    #   is equal to the test dataframe
    assert df.equals(dataframe)

    # clean up after the test
    os.unlink("/tmp/dataframe.csv")


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
    # create a temporary .csv file
    dataframe.to_csv("/tmp/dataframe.csv")

    # Test 1
    with pytest.raises(TypeError):
        janitor.io.read_commandline(6)

    # Test 2
    with pytest.raises(CalledProcessError):
        janitor.io.read_commandline("bad command")

    # Test 3
    with pytest.raises(pd.errors.EmptyDataError):
        janitor.io.read_commandline("cat")

    # clean up after the tests
    os.unlink("/tmp/dataframe.csv")
