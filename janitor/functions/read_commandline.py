import pandas as pd
import subprocess
from io import StringIO


def read_commandline(cmd: str) -> pd.DataFrame:
    """
    Read a CSV file based on a command-line command.

    For example, you may wish to run the following command on `sep-quarter.csv`
    before reading it into a pandas DataFrame:

    ```bash
    cat sep-quarter.csv | grep .SEA1AA
    ```

    In this case, you can use the following Python code to load the dataframe:

    ```python
    import janitor as jn
    df = jn.read_commandline("cat sep-quarter.csv | grep .SEA1AA")
    ```

    :param cmd: Shell command to preprocess a file on disk.
    :returns: A pandas DataFrame parsed from the stdout of the underlying
        shell.
    :raises EmptyDataError: If there is no data to parse, this often happens
        because the cmd param is either an invalid bash command, thus
        nothing happens in the shell , or if cmd param is not a string,
        thus creating an invalid shell command.
    """
    # cmd = cmd.split(" ")
    try:
        outcome = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )
        outcome = outcome.stdout
    except pd.EmptyDataError:
        msg = (
            "Empty Data Error: Be sure your parameter"
            " is both a valid shell command and a string"
        )
        raise pd.EmptyDataError(msg)

    return pd.read_csv(StringIO(outcome))
