
def read_csvs(
    directory: Union[str, Path], 
    pattern: str = pattern, 
    seperate_df: bool = seperate_df,
    **kwargs
):
    """
    :param directory: The directory that contains the CSVs.
    :param pattern: The pattern of CSVs to match.
    :param seperate_df: Returns a dictionary of seperate dataframes for each CSV file read
    :param kwargs: Keyword arguments to pass into pandas original `read_csv`.
    """
    if seperate_df:
        dfs = {
            os.path.basename(f): pd.read_csv(f, **kwargs) 
            for f 
            in glob.glob(os.path.join(path, "*.csv"))
        }
        return dfs
    else: 
        df = pd.concat(
            [
                pd.read_csv(f, **kwargs).assign(filename = os.path.basename(f)) 
                for f 
                in glob.glob(os.path.join(path, pattern))
            ]
            ignore_index=True, 
            sort=False)
        return df