import pandas_flavor as pf
import pandas as pd
from subprocess import Popen, PIPE
import sys


@pf.register_dataframe_method
def read_commandline() -> pd.DataFrame:
    """
    Read command from commandline.

    argv[0] should be program name
    argv[1:] will be the a list of commands available to the shell
    such as:
    cat sep-quarter.csv | grep .SEA1AA

    calling
    df = read_commandline()
    will create a new dataframe from the commandline arguments given.

    :returns: a dataframe which has been created based on the arguments
        given in the commandline
    """
    command = __preprocess_command()
    df = query_df(command)
    return df


def manual_read_command(string: str) -> pd.DataFrame:
    """
    Can be used for testing, will take a hard-coded
    string in place of argv[1:]

    :param string: said string which takes places of commandline arguments.
    :returns: a dataframe which has been created based on then string
        given as a parameter in the function
    """
    df = query_df(string)
    return df


def __preprocess_command():
    """
    Transforms argv[1:] into a single string
    which is more easily parsed by PIPE and Popen above
    """
    command = ""
    for word in sys.argv[1:]:  # must be a single string to be used in Popen
        command += word + " "
    command = str(command)
    print(command)
    return command


def query_df(command: str) -> pd.DataFrame:
    with Popen(command, shell=True, stdout=PIPE) as process:
        df = pd.read_csv(process.stdout)
        return df


# currently not working as intended:
# planned to be used as a QOL tool which grabs only column names

# def __preprocess_file(command: str):
#     df = pd.DataFrame()
#     cmd_list = command.split()
#     if ".csv" in command:
#         for i in cmd_list:
#             if ".csv" in i:
#                 df = pd.read_csv(i, nrows=1)  # grab column names only
#                 break
#     else:
#         print("unable to preprocess file.")
#         print("file should have a csv file extension")
#         # df = pd.DataFrame()
#
#     return df
