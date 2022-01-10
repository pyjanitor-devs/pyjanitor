import os
import pandas as pd
import janitor.io


"""
Tests seek to assert that a dataframe made
    from the read_commandline function
    will create a dataframe identical
    to a test dataframe

Tests 1 and 2 assert are admittedly trivial,
    they effectively prove the validity of
    the pd.testing library by proving the
    test data is equal to itself. Note the
    requirement of 'assert not' for
    assert_frame_equal.

Tests 3 and 4 assert that the dataframe made
    from the read_commandline function is
    identical to the test dataframe from
    which the .csv file was created.

"""


def test_read_commandline(dataframe):
    # create a temporary .csv file from test data
    dataframe.to_csv("/tmp/dataframe.csv")

    # create a new dataframe from the temporary .csv using
    #   the cat command from the bash commandline
    df = janitor.io.read_commandline("cat /tmp/dataframe.csv")

    # The user may be expected to process some of the df themself
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # Make assertion dataframe is equal to itself
    assert not pd.testing.assert_frame_equal(dataframe, dataframe)
    assert dataframe.equals(dataframe)

    # Make assertion that new dataframe created with read_commandline
    #   is equal to the test dataframe
    assert not pd.testing.assert_frame_equal(dataframe, df)
    assert df.equals(dataframe)

    # clean up after the test
    os.unlink("/tmp/dataframe.csv")
