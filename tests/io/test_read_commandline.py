import os
import pandas as pd
import janitor.io


def test_read_commandline(dataframe):
    """
    Tests seek to assert that a dataframe made
        from the read_commandline function
        will create a dataframe identical
        to a test dataframe


    Test 1 asserts that the dataframe made
        from the read_commandline function is
        identical to the test dataframe from
        which the .csv file was created.

    """
    # create a temporary .csv file from test data
    dataframe.to_csv("/tmp/dataframe.csv", index = 0)

    # create a new dataframe from the temporary .csv using
    #   the cat command from the bash commandline
    df = janitor.io.read_commandline("cat /tmp/dataframe.csv")

    # Make assertion that new dataframe created with read_commandline
    #   is equal to the test dataframe
    assert df.equals(dataframe)

    # clean up after the test
    os.unlink("/tmp/dataframe.csv")
