import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Processing Anime Data

    ## Background

    We will use pyjanitor to showcase how to conveniently chain methods together to perform data cleaning in one shot. We
    We first define and register a series of dataframe methods with pandas_flavor. Then we chain the dataframe methods together with pyjanitor methods to complete the data cleaning process. The below example shows a one-shot script followed by a step-by-step detail of each part of the method chain.

    We have adapted a [TidyTuesday analysis](https://github.com/rfordatascience/tidytuesday/blob/master/data/2019/2019-04-23/readme.md) that was originally performed in R. The original text from TidyTuesday will be shown in blockquotes.

    Note: TidyTuesday is based on the principles discussed and made popular by Hadley Wickham in his paper [Tidy Data](https://www.jstatsoft.org/index.php/jss/article/view/v059i10/v59i10.pdf).

    *The original text from TidyTuesday will be shown in blockquotes.*
    Here is a description of the Anime data set that we will use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    >This week's data comes from [Tam Nguyen](https://github.com/tamdrashtri) and [MyAnimeList.net via Kaggle](https://www.kaggle.com/aludosan/myanimelist-anime-dataset-as-20190204). [According to Wikipedia](https://en.wikipedia.org/wiki/MyAnimeList) - "MyAnimeList, often abbreviated as MAL, is an anime and manga social networking and social cataloging application website. The site provides its users with a list-like system to organize and score anime and manga. It facilitates finding users who share similar tastes and provides a large database on anime and manga. The site claims to have 4.4 million anime and 775,000 manga entries. In 2015, the site received 120 million visitors a month."
    >
    >Anime without rankings or popularity scores were excluded. Producers, genre, and studio were converted from lists to tidy observations, so there will be repetitions of shows with multiple producers, genres, etc. The raw data is also uploaded.
    >
    >Lots of interesting ways to explore the data this week!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Import libraries and load data
    """)
    return


@app.cell
def _():
    # Import pyjanitor and pandas
    import pandas as pd
    import pandas_flavor as pf
    return pd, pf


@app.cell
def _():
    # Suppress user warnings when we try overwriting our custom pandas flavor functions
    import warnings

    warnings.filterwarnings("ignore")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One-Shot
    """)
    return


@app.cell
def _(pd, pf):
    _filename = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-04-23/raw_anime.csv'
    df = pd.read_csv(_filename)

    @pf.register_dataframe_method
    def _str_remove(df, column_name: str, pat: str, *args, **kwargs):
        """Wrapper around df.str.replace"""
        df[column_name] = df[column_name].str.replace(pat, '', *args, **kwargs)
        return df

    @pf.register_dataframe_method
    def _str_trim(df, column_name: str, *args, **kwargs):
        """Wrapper around df.str.strip"""
        df[column_name] = df[column_name].str.strip(*args, **kwargs)
        return df

    @pf.register_dataframe_method
    def _explode(df: pd.DataFrame, column_name: str, sep: str):
        """
        For rows with a list of values, this function will create new
        rows for each value in the list
        """
        df['id'] = df.index
        wdf = pd.DataFrame(df[column_name].str.split(sep).fillna('').tolist()).stack().reset_index()
        wdf.columns = ['id', 'depth', column_name]
        wdf.drop('depth', axis=1, inplace=True)
        return pd.merge(df, wdf, on='id', suffixes=('_drop', '')).drop(columns=['id', column_name + '_drop'])

    @pf.register_dataframe_method
    def _str_word(df, column_name: str, start: int=None, stop: int=None, pat: str=' ', *args, **kwargs):
        """
        Wrapper around `df.str.split`
        with additional `start` and `end` arguments
        to select a slice of the list of words.
        """  # exploded_column = column_name
        df[column_name] = df[column_name].str.split(pat).str[start:stop]  # plural form to singular form
        return df  # wdf[column_name] = wdf[column_name].apply(lambda x: x.strip())  # trim

    @pf.register_dataframe_method
    def _str_join(df, column_name: str, sep: str, *args, **kwargs):
        """
        Wrapper around `df.str.join`
        Joins items in a list.
        """
        df[column_name] = df[column_name].str.join(sep)
        return df

    @pf.register_dataframe_method
    def _str_slice(df, column_name: str, start: int=None, stop: int=None, *args, **kwargs):
        """
        Wrapper around `df.str.slice
        """
        df[column_name] = df[column_name].str[start:stop]
        return df
    clean_df = df.str_remove(column_name='producers', pat='\\[|\\]').explode(column_name='producers', sep=',').str_remove(column_name='producers', pat="'").str_trim('producers').str_remove(column_name='genre', pat='\\[|\\]').explode(column_name='genre', sep=',').str_remove(column_name='genre', pat="'").str_trim(column_name='genre').str_remove(column_name='studio', pat='\\[|\\]').explode(column_name='studio', sep=',').str_remove(column_name='studio', pat="'").str_trim(column_name='studio').str_remove(column_name='aired', pat="\\{|\\}|'from':\\s*|'to':\\s*").str_word(column_name='aired', start=0, stop=2, pat=',').str_join(column_name='aired', sep=',').deconcatenate_column(column_name='aired', new_column_names=['start_date', 'end_date'], sep=',').remove_columns(column_names=['aired']).str_remove(column_name='start_date', pat="'").str_slice(column_name='start_date', start=0, stop=10).str_remove(column_name='end_date', pat="'").str_slice(column_name='end_date', start=0, stop=11).to_datetime('start_date', format='%Y-%m-%d', errors='coerce').to_datetime('end_date', format='%Y-%m-%d', errors='coerce').fill_empty(columns=['rank', 'popularity'], value=0).filter_on('rank != 0 & popularity != 0')
    return (clean_df,)


@app.cell
def _(clean_df):
    clean_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Step
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    >Data Dictionary
    >
    >Heads up the dataset is about 97 mb - if you want to free up some space, drop the synopsis and background, they are long strings, or broadcast, premiered, related as they are redundant or less useful.
    >
    >|variable       |class     |description |
    |:--------------|:---------|:-----------|
    |animeID        |double    | Anime ID (as in https://myanimelist.net/anime/animeID)          |
    |name           |character |anime title - extracted from the site.           |
    |title_english  |character | title in English (sometimes is different, sometimes is missing)          |
    |title_japanese |character | title in Japanese (if Anime is Chinese or Korean, the title, if available, in the respective language)          |
    |title_synonyms |character | other variants of the title         |
    |type           |character | anime type (e.g. TV, Movie, OVA)          |
    |source         |character | source of anime (i.e original, manga, game, music, visual novel etc.)         |
    |producers      |character | producers          |
    |genre          |character | genre         |
    |studio         |character | studio           |
    |episodes       |double    | number of episodes           |
    |status         |character | Aired or not aired      |
    |airing         |logical   | True/False is still airing          |
    |start_date     |double    | Start date (ymd)        |
    |end_date       |double    | End date (ymd)        |
    |duration       |character | Per episode duration or entire duration, text string        |
    |rating         |character | Age rating         |
    |score          |double    | Score (higher = better)       |
    |scored_by      |double    | Number of users that scored          |
    |rank           |double    | Rank - weight according to MyAnimeList formula          |
    |popularity     |double    |  based on how many members/users have the respective anime in their list          |
    |members        |double    | number members that added this anime in their list         |
    |favorites      |double    | number members that favorites these in their list          |
    |synopsis       |character | long string with anime synopsis          |
    |background     |character | long string with production background and other things          |
    |premiered      |character | anime premiered on season/year          |
    |broadcast      |character | when is (regularly) broadcasted         |
    |related        |character | dictionary: related animes, series, games etc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 0: Load data
    """)
    return


@app.cell
def _(pd):
    _filename = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-04-23/raw_anime.csv'
    df_1 = pd.read_csv(_filename)
    return (df_1,)


@app.cell
def _(df_1):
    df_1.head(3).T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: Clean `producers` column
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first step tries to clean up the `producers` column by removing some brackets ('[]') and trim off some empty spaces

    >```
    >clean_df <- raw_df %>%
    >  # Producers
    >  mutate(producers = str_remove(producers, "\\["),
             producers = str_remove(producers, "\\]"))
    >```

    What is mutate? This [link](https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html) compares R's `mutate` to be similar to pandas' `df.assign`.
    However, `df.assign` returns a new DataFrame whereas `mutate` adds a new variable while preserving the previous ones.
    Therefore, for this example, I will compare `mutate` to be similar to `df['col'] = X`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, this is looks like a list of items but in string form
    """)
    return


@app.cell
def _(df_1):
    # Let's see what we trying to remove
    df_1.loc[df_1['producers'].str.contains('\\[', na=False), 'producers'].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's use pandas flavor to create a custom method for just removing some strings so we don't have to use str.replace so many times.
    """)
    return


@app.cell
def _(pf):
    @pf.register_dataframe_method
    def _str_remove(df, column_name: str, pat: str, *args, **kwargs):  # noqa: F811
        """
        Wrapper around df.str.replace
        The function will loop through regex patterns
        and remove them from the desired column.

        :param df: A pandas DataFrame.
        :param column_name: A `str` indicating which column
            the string removal action is to be made.
        :param pat: A regex pattern to match and remove.
        """
        if not isinstance(pat, str):
            raise TypeError(f'Pattern should be a valid regex pattern. Received pattern: {pat} with dtype: {type(pat)}')
        df[column_name] = df[column_name].str.replace(pat, '', *args, **kwargs)
        return df
    return


@app.cell
def _(df_1):
    clean_df_1 = df_1.str_remove(column_name='producers', pat='\\[|\\]')
    return (clean_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With brackets removed.
    """)
    return


@app.cell
def _(clean_df_1):
    clean_df_1['producers'].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brackets are removed. Now the next part
    >```
    >  separate_rows(producers, sep = ",") %>%
    >```

    It seems like separate rows will go through each value of the column, and if the value is a list, will create a new row for each value in the list with the remaining column values being the same. This is commonly known as an `explode` method but it is not yet implemented in pandas. We will need a function for this (code adopted from [here](https://qiita.com/rikima/items/c10e27d8b7495af4c159)).
    """)
    return


@app.cell
def _(pd, pf):
    @pf.register_dataframe_method
    def _explode(df: pd.DataFrame, column_name: str, sep: str):  # noqa: F811
        """
        For rows with a list of values,
        this function will create new rows
        for each value in the list

        :param df: A pandas DataFrame.
        :param column_name: A `str` indicating which column
            the string removal action is to be made.
        :param sep: The delimiter.
            Example delimiters include `|`, `, `, `,` etc.
        """
        df['id'] = df.index
        wdf = pd.DataFrame(df[column_name].str.split(sep).fillna('').tolist()).stack().reset_index()
        wdf.columns = ['id', 'depth', column_name]
        wdf.drop('depth', axis=1, inplace=True)
        return pd.merge(df, wdf, on='id', suffixes=('_drop', '')).drop(columns=['id', column_name + '_drop'])  # exploded_column = column_name  # plural form to singular form  # wdf[column_name] = wdf[column_name].apply(lambda x: x.strip())  # trim
    return


@app.cell
def _(clean_df_1):
    clean_df_2 = clean_df_1.explode(column_name='producers', sep=',')
    return (clean_df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now every producer is its own row.
    """)
    return


@app.cell
def _(clean_df_2):
    clean_df_2['producers'].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now remove single quotes and a bit of trimming
    >```
      mutate(producers = str_remove(producers, "\\'"),
             producers = str_remove(producers, "\\'"),
             producers = str_trim(producers)) %>%
    ```
    """)
    return


@app.cell
def _(clean_df_2):
    clean_df_3 = clean_df_2.str_remove(column_name='producers', pat="'")
    return (clean_df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll make another custom function for trimming whitespace.
    """)
    return


@app.cell
def _(pf):
    @pf.register_dataframe_method
    def _str_trim(df, column_name: str, *args, **kwargs):  # noqa: F811
        """Remove trailing and leading characters, in a given column"""
        df[column_name] = df[column_name].str.strip(*args, **kwargs)
        return df
    return


@app.cell
def _(clean_df_3):
    clean_df_4 = clean_df_3.str_trim('producers')
    return (clean_df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, here is our cleaned `producers` column.
    """)
    return


@app.cell
def _(clean_df_4):
    clean_df_4['producers'].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: Clean `genre` and `studio` Columns
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's do the same process for columns `Genre` and `Studio`

    >```
    >  # Genre
      mutate(genre = str_remove(genre, "\\["),
             genre = str_remove(genre, "\\]")) %>%
      separate_rows(genre, sep = ",") %>%
      mutate(genre = str_remove(genre, "\\'"),
             genre = str_remove(genre, "\\'"),
             genre = str_trim(genre)) %>%
    >  # Studio
      mutate(studio = str_remove(studio, "\\["),
             studio = str_remove(studio, "\\]")) %>%
      separate_rows(studio, sep = ",") %>%
      mutate(studio = str_remove(studio, "\\'"),
             studio = str_remove(studio, "\\'"),
             studio = str_trim(studio)) %>%
    ```
    """)
    return


@app.cell
def _(clean_df_4):
    clean_df_5 = clean_df_4.str_remove(column_name='genre', pat='\\[|\\]').explode(column_name='genre', sep=',').str_remove(column_name='genre', pat="'").str_trim(column_name='genre').str_remove(column_name='studio', pat='\\[|\\]').explode(column_name='studio', sep=',').str_remove(column_name='studio', pat="'").str_trim(column_name='studio')  # Perform operation for genre.  # Now do it for studio
    return (clean_df_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Resulting cleaned columns.
    """)
    return


@app.cell
def _(clean_df_5):
    clean_df_5[['genre', 'studio']].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: Clean `aired` column
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `aired` column has something a little different. In addition to the usual removing some strings and whitespace trimming, we want to separate the values into two separate columns `start_date` and `end_date`

    >```r
    >  # Aired
      mutate(aired = str_remove(aired, "\\{"),
             aired = str_remove(aired, "\\}"),
             aired = str_remove(aired, "'from': "),
             aired = str_remove(aired, "'to': "),
             aired = word(aired, start = 1, 2, sep = ",")) %>%
      separate(aired, into = c("start_date", "end_date"), sep = ",") %>%
      mutate(start_date = str_remove_all(start_date, "'"),
             start_date = str_sub(start_date, 1, 10),
             end_date = str_remove_all(start_date, "'"),
             end_date = str_sub(end_date, 1, 10)) %>%
      mutate(start_date = lubridate::ymd(start_date),
             end_date = lubridate::ymd(end_date)) %>%
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will create some custom wrapper functions to emulate R's `word` and use pyjanitor's `deconcatenate_column`.
    """)
    return


@app.cell
def _(clean_df_5):
    # Currently looks like this
    clean_df_5['aired'].head()
    return


@app.cell
def _(pf):
    @pf.register_dataframe_method
    def _str_word(df, column_name: str, start: int=None, stop: int=None, pat: str=' ', *args, **kwargs):  # noqa: F811
        """
        Wrapper around `df.str.split`,
        with additional `start` and `end` arguments
        to select a slice of the list of words.

        :param df: A pandas DataFrame.
        :param column_name: A `str` indicating which column
            the split action is to be made.  # noqa: F811
        :param start: optional An `int` for the start index of the slice
        :param stop: optional  An `int` for the end index of the slice
        :param pat: String or regular expression to split on.
            If not specified, split on whitespace.

        """
        df[column_name] = df[column_name].str.split(pat).str[start:stop]
        return df

    @pf.register_dataframe_method
    def _str_join(df, column_name: str, sep: str, *args, **kwargs):
        """
        Wrapper around `df.str.join`
        Joins items in a list.

        :param df: A pandas DataFrame.
        :param column_name: A `str` indicating which column
            the split action is to be made.
        :param sep: The delimiter. Example delimiters
            include `|`, `, `, `,` etc.  # noqa: F811
        """
        df[column_name] = df[column_name].str.join(sep)
        return df

    @pf.register_dataframe_method
    def _str_slice(df, column_name: str, start: int=None, stop: int=None, *args, **kwargs):
        """
        Wrapper around `df.str.slice
        Slices strings.

        :param df: A pandas DataFrame.
        :param column_name: A `str` indicating which column
            the split action is to be made.
        :param start: 'int' indicating start of slice.
        :param stop: 'int' indicating stop of slice.
        """  # noqa: F811
        df[column_name] = df[column_name].str[start:stop]
        return df  # noqa: F811
    return


@app.cell
def _(clean_df_5):
    clean_df_6 = clean_df_5.str_remove(column_name='aired', pat="\\{|\\}|'from':\\s*|'to':\\s*").str_word(column_name='aired', start=0, stop=2, pat=',').str_join(column_name='aired', sep=',').deconcatenate_column(column_name='aired', new_column_names=['start_date', 'end_date'], sep=',').remove_columns(column_names=['aired']).str_remove(column_name='start_date', pat="'").str_slice(column_name='start_date', start=0, stop=10).str_remove(column_name='end_date', pat="'").str_slice(column_name='end_date', start=0, stop=11).to_datetime('start_date', format='%Y-%m-%d', errors='coerce').to_datetime('end_date', format='%Y-%m-%d', errors='coerce')  # .add_columns({'start_date': clean_df['aired'][0]})
    return (clean_df_6,)


@app.cell
def _(clean_df_6):
    # Resulting 'start_date' and 'end_date' columns with 'aired' column removed
    clean_df_6[['start_date', 'end_date']].head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4: Filter out unranked and unpopular series
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, let's drop the unranked or unpopular series with pyjanitor's `filter_on`.
    """)
    return


@app.cell
def _(clean_df_6):
    # First fill any NA values with 0 and then filter != 0
    clean_df_7 = clean_df_6.fill_empty(column_names=['rank', 'popularity'], value=0).filter_on('rank != 0 & popularity != 0')
    return (clean_df_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### End Result
    """)
    return


@app.cell
def _(clean_df_7):
    clean_df_7.head()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
