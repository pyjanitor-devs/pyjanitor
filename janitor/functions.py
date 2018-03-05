def clean_names(df):
    columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = columns
    return df
