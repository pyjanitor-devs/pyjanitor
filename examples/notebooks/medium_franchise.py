import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tidy Up Web-Scraped Media Franchise Data

    ## Background
    This example combines functionalities of [pyjanitor](https://anaconda.org/conda-forge/pyjanitor) and [pandas-flavor](https://anaconda.org/conda-forge/pandas-flavor) to showcase an explicit--and thus reproducible--workflow enabled by dataframe __method chaining__.

    The data cleaning workflow largely follows the [R example](https://github.com/rfordatascience/tidytuesday/blob/master/data/2019/2019-07-02/revenue.R) from [the tidytuesday project](https://github.com/rfordatascience/tidytuesday). The raw data is scraped from [Wikipedia page](https://en.wikipedia.org/wiki/List_of_highest-grossing_media_franchises) titled "*List of highest-grossing media franchises*". The workflow is presented both in multi-step (section1) and in one-shot (section 2) fashions.

    More specifically, you will find several data-cleaning techniques that one may encounter frequently in web-scraping tasks; This includes:

    * String operations with regular expressions (with `pandas-favor`)
    * Data type changes (with `pyjanitor`)
    * Split strings in cells into separate rows (with `pandas-flavor`)
    * Split strings in cells into separate columns (with `pyjanitor` + `pandas-flavor`)
    * Filter dataframe values based on substring pattern (with `pyjanitor`)
    * Column value remapping with fuzzy substring matching (with `pyjanitor` + `pandas-flavor`)

    Data visualization is not included in this example. But if you are looking for inspirations, [here](https://www.reddit.com/r/dataisbeautiful/comments/c53540/highest_grossing_media_franchises_oc/) is a good example.

    ---

    ## Structural convention
    ### 1. Annotation system in code comments
    This example includes both `pyjanitor` and `pandas-flavors` methods. As you step through this example, you will see the following annotation system in code comments that explains various methods' categories:

    * `[pyjanitor]` denotes `pyjanitor` methods
    * `[pandas-flavor]` denotes custom `pandas-flavor` methods
    * `[pyjanitor + pandas-flavor]` denotes `pandas-flavor` methods built on top of `pyjanitor` functions

    ### 2. R counterpart snippets and python code in tandem
    The multi-step workflow is presented by alternating the original R snippets (from tidytuesday) and the corresponding python implementations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Python implementation

    ### Preparation
    """)
    return


@app.cell
def _():
    # Import pyjanitor and pandas
    from typing import List

    import pandas as pd
    import pandas_flavor as pf
    return List, pd, pf


@app.cell
def _():
    # Suppress user warnings
    # when we try overwriting our custom pandas flavor functions
    import warnings

    warnings.filterwarnings("ignore")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Section 1 Multi-Step
    #### Load data

    ##### Note: The table from the url has been saved as a csv file for use in this example notebook.

    R snippet:
    ```R
    url <- "https://en.wikipedia.org/wiki/List_of_highest-grossing_media_franchises"
    df <- url %>%
      read_html() %>%
      html_table(fill = TRUE) %>%
      .[[2]]
    ```
    """)
    return


@app.cell
def _(pd):
    # originally from
    # https://en.wikipedia.org/wiki/List_of_highest-grossing_media_franchises
    fileurl = "../data/medium_franchise_raw_table.csv"
    df_raw = pd.read_csv(fileurl)
    df_raw.head(3)
    return df_raw, fileurl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Rename columns
    R snippet:
    ```R
    clean_money <- df %>%
      set_names(nm = c("franchise", "year_created", "total_revenue", "revenue_items",
                       "original_media", "creators", "owners"))
    ```
    """)
    return


@app.cell
def _(pf):
    # pandas-flavor helper functions
    @pf.register_dataframe_method
    def str_remove(df, column_name: str, pattern: str=''):
    # [pandas-flavor]
        """Remove string pattern from a column

        Wrapper around df.str.replace()

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified
        column_name: str
            Name of the column to be operated on
        pattern: str, default to ''
            String pattern to be removed

        Returns
        --------
        df: pd.Dataframe

        """
        df[_column_name] = df[_column_name].str.replace(pattern, '')
        return df

    @pf.register_dataframe_method
    def str_trim(df, column_name: str):
        """Remove leading and trailing white space from a column of strings

        Wrapper around df.str.strip()

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified
        column_name: str
            Name of the column to be operated on

        Returns
        --------
        df: pd.Dataframe

        """
        df[_column_name] = df[_column_name].str.strip()
        return df

    @pf.register_dataframe_method
    def str_slice(df, column_name: str, start: int=0, stop: int=-1):
        """Slice a column of strings by indexes

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified
        column_name: str
            Name of the column to be operated on
        start: int, optional, default to 0
            Staring index for string slicing
        stop: int, optional, default to -1
            Ending index for string slicing

        Returns
        --------
        df: pd.Dataframe

        """
        df[_column_name] = df[_column_name].str[start:stop]
        return df
    return


@app.cell
def _(df_raw):
    colnames = (
        "franchise",
        "year_created",
        "total_revenue",
        "revenue_items",
        "original_media",
        "creators",
        "owners",
    )
    df_dirty = df_raw.rename(
        columns={col_old: col_new for col_old, col_new in zip(df_raw.columns, colnames)}
    )
    df_dirty.head(3)
    return colnames, df_dirty


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Clean up `total_revenue` column
    R snippet:
    ```R
    clean_money <- df %>% ... %>%
    mutate(total_revenue = str_remove(total_revenue, "est."),
         total_revenue = str_trim(total_revenue),
         total_revenue = str_remove(total_revenue, "[$]"),
         total_revenue = word(total_revenue, 1, 1),
         total_revenue = as.double(total_revenue))
    ```
    """)
    return


@app.cell
def _(df_dirty):
    _column_name = 'total_revenue'
    df_clean_money = df_dirty.str_remove(_column_name, pattern='est.').str_trim(_column_name).str_remove(_column_name, pattern='\\$').str_slice(_column_name, start=0, stop=2).change_type(_column_name, float)
    df_clean_money.head(3)  # [pandas-flavor]  # [pyjanitor]
    return (df_clean_money,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Split column `revenue_items` into `revenue_category` and `revenue`
    R snippet:
    ```R
    clean_category <- clean_money %>%
        separate_rows(revenue_items, sep = "\\[") %>%
        filter(str_detect(revenue_items, "illion")) %>%
        separate(revenue_items, into = c("revenue_category", "revenue"), sep = "[$]") %>%
        mutate(revenue_category = str_remove(revenue_category, " – "),
             revenue_category = str_remove(revenue_category, regex(".*\\]")),
             revenue_category = str_remove(revenue_category, "\n"))
    ```
    """)
    return


@app.cell
def _(List, pd, pf):
    # pandas-flavor helper functions
    @pf.register_dataframe_method
    def separate_rows(df, column_name: str, sep: str=''):
    # [pandas-flavor]
        """Split each cell of a column that contains a list of items
        (separated by `sep`) into separate rows

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified
        column_name: str
            Name of the column to be operated on
        sep: str, default to ''
            Substring used as separators for cell splitting

        Returns
        --------
        df: pd.Dataframe

        """
        columns_original = list(df.columns)
        df['id'] = df.index
        wdf = pd.DataFrame(df[_column_name].str.split(sep).tolist()).stack().reset_index()  # Preserve an id field for later merge
        wdf.rename(columns={'level_0': 'id', 0: 'revenue_items'}, inplace=True)
        wdf.drop(columns=['level_1'], inplace=True)
        return pd.merge(df, wdf, on='id', suffixes=('_drop', '')).drop(columns=['id', _column_name + '_drop'])[columns_original]
      # Preserve the same id field for merge
    @pf.register_dataframe_method
    def separate(df, column_name: str, into: List[str]=None, sep: str=''):
        """Split a column into separate columns at separator specified by `sep`  # Merge and preserve original order

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified
    # [pyjanitor + pandas-flavor]
        column_name: str
            Name of the column to be operated on
        into: List[str], default to None
            New column names for the split columns
        sep: str, default to ''
            Separator at which to split the column

        Returns
        --------
        df: pd.Dataframe

        """
        index_original = list(df.columns).index(_column_name)
        cols = list(df.columns)
        cols.remove(_column_name)
        for i, col in enumerate(into):
            cols.insert(index_original + i, col)
        return df.deconcatenate_column(_column_name, new_column_names=into, sep=sep).drop(columns=_column_name)[cols]
    return


@app.cell
def _(df_clean_money):
    # Generate `df_clean_category` on top of `df_clean_money`
    _column_name = 'revenue_items'
    df_clean_category = df_clean_money.separate_rows(_column_name, sep='\\[').filter_string(_column_name, 'illion').separate(_column_name, into=['revenue_category', 'revenue'], sep='\\$').str_remove('revenue_category', pattern=' – ').str_remove('revenue_category', pattern='.*\\]').str_remove('revenue_category', pattern='\n')
    df_clean_category.head(3)  # [pandas-flavor]  # [pyjanitor]  # [pyjanitor + pandas-flavor]  # [pandas-flavor]
    return (df_clean_category,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Clean up `revenue_category` column
    R snippet:
    ```R
    clean_df <- clean_category %>%
      mutate(revenue_category = case_when(
        str_detect(str_to_lower(revenue_category), "box office") ~ "Box Office",
        str_detect(str_to_lower(revenue_category), "dvd|blu|vhs|home video|video rentals|video sales|streaming|home entertainment") ~ "Home Video/Entertainment",
        str_detect(str_to_lower(revenue_category), "video game|computer game|mobile game|console|game|pachinko|pet|card") ~ "Video Games/Games",
        str_detect(str_to_lower(revenue_category), "comic|manga") ~ "Comic or Manga",
        str_detect(str_to_lower(revenue_category), "music|soundtrack") ~ "Music",
        str_detect(str_to_lower(revenue_category), "tv") ~ "TV",
        str_detect(str_to_lower(revenue_category), "merchandise|licens|mall|stage|retail") ~ "Merchandise, Licensing & Retail",
        TRUE ~ revenue_category))
    ```
    """)
    return


@app.cell
def _(pf):
    # pandas-flavor helper functions
    @pf.register_dataframe_method
    def fuzzy_match_replace(df, column_name: str, mapper: dict=None):
    # [pyjanitor + pandas-flavor]
        """Value remapping for specific column with fuzzy matching and replacement
        of strings

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified
        column_name: str
            Name of the column to be operated on
        mapper: dict, default to None
            {oldstr_0: newstr_0, oldstr_1: newstr_1, ..., oldstr_n: newstr_n}

        Returns
        --------
        df: pd.Dataframe

        """
        for k, v in mapper.items():
            condition = df[_column_name].str.contains(k)
            df = df.update_where(condition, _column_name, v)
        return df  # [pyjanitor] update_where: update value when condition is True
    return


@app.cell
def _(df_clean_category):
    value_mapper = {'box office': 'Box Office', 'dvd|blu|vhs|home video|video rentals|video sales|streaming|home entertainment': 'Home Video/Entertainment', 'video game|computer game|mobile game|console|game|pachinko|pet|card': 'Video Games/Games', 'comic|manga': 'Comic or Manga', 'music|soundtrac': 'Music', 'tv': 'TV', 'merchandise|licens|mall|stage|retail': 'Merchandise, Licensing & Retail'}
    _column_name = 'revenue_category'
    df_clean_category_1 = df_clean_category.transform_column(_column_name, str.lower).transform_column(_column_name, str.strip).fuzzy_match_replace(_column_name, mapper=value_mapper)
    df_clean_category_1.head(3)
    return df_clean_category_1, value_mapper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Clean up `revenue` column
    R snippet:
    ```R
    # magic command not supported in marimo; please file an issue to add support
    # %>%
    mutate(revenue = str_remove(revenue, "illion"),
         revenue = str_trim(revenue),
         revenue = str_remove(revenue, " "),
         revenue = case_when(str_detect(revenue, "m") ~ paste0(str_extract(revenue, "[:digit:]+"), "e-3"),
                             str_detect(revenue, "b") ~ str_extract(revenue, "[:digit:]+"),
                             TRUE ~ NA_character_),
         revenue = format(revenue, scientific = FALSE),
         revenue = parse_number(revenue)) %>%
    mutate(original_media = str_remove(original_media, "\\[.+"))
    ```
    """)
    return


@app.cell
def _(pd, pf):
    # pandas-flavor helper functions
    @pf.register_dataframe_method
    def str_replace(df, column_name: str, old: str='', new: str=''):
    # [pandas-flavor]
        """Match and replace strings from a dataframe column.
        Wrapper around df.str.replace

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified
        column_name: str
            Name of the column to be operated on
        old: str, default to ''
            Old string to be matched and replaced
        new: str, default to ''
            New string to replace old

        Returns
        --------
        df: pd.Dataframe

        """
        df[_column_name] = df[_column_name].str.replace(old, new)
        return df

    @pf.register_dataframe_method
    def parse_number(df):
        """Check all columns of dataframe and properly parse numeric types

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified

        Returns
        --------
        df: pd.Dataframe

        """
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                continue
        return df

    @pf.register_dataframe_method
    def flatten_multiindex(df):
        """Flatten dataframe with multilevel index
        A wrapper around pd.DataFrame(df.to_records())

        Parameters
        -----------
        df: pd.Dataframe
            Input dataframe to be modified

        Returns
        --------
        df: pd.Dataframe

        """
        return pd.DataFrame(df.to_records())
    return


@app.cell
def _(df_clean_category_1):
    _column_name = 'revenue'
    df_clean = df_clean_category_1.str_remove(_column_name, 'illion').str_trim(_column_name).str_remove(_column_name, ' ').str_replace(_column_name, '\\s*b', '').str_replace(_column_name, '\\s*m', 'e-3').parse_number().str_remove('original_media', '\\[.+')
    df_clean.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Section 2 One-Shot
    """)
    return


@app.cell
def _(colnames, df_raw, fileurl, pd, value_mapper):
    df_clean_1 = pd.read_csv(fileurl).rename(columns={col_old: col_new for col_old, col_new in zip(df_raw.columns, colnames)}).str_remove('total_revenue', pattern='est.').str_trim('total_revenue').str_remove('total_revenue', pattern='\\$').str_slice('total_revenue', start=0, stop=2).change_type('total_revenue', float).separate_rows('revenue_items', sep='\\[').filter_string('revenue_items', 'illion').separate('revenue_items', into=['revenue_category', 'revenue'], sep='\\$').str_remove('revenue_category', pattern=' – ').str_remove('revenue_category', pattern='.*\\]').str_remove('revenue_category', pattern='\n').transform_column('revenue_category', str.lower).transform_column('revenue_category', str.strip).fuzzy_match_replace('revenue_category', mapper=value_mapper).str_remove('revenue', 'illion').str_trim('revenue').str_remove('revenue', ' ').str_replace('revenue', '\\s*b', '').str_replace('revenue', '\\s*m', 'e-3').parse_number().str_remove('original_media', '\\[.+')  # [pandas-flavor]  # [pyjanitor]  # [pandas-flavor]  # [pyjanitor]  # [pyjanitor + pandas-flavor]  # [pandas-flavor]  # [pyjanitor] convert to lower case  # [pyjanitor] strip leading/trailing white space  # [pyjanitor + pandas_flavor]  # [pandas-flavor]
    return (df_clean_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Final aggregation and join
    R snippet:
    ```R
     sum_df <- clean_df %>%
      group_by(franchise, revenue_category) %>%
      summarize(revenue = sum(revenue))

    metadata_df <- clean_df %>%
      select(franchise:revenue_category, original_media:owners, -total_revenue)

    final_df <- left_join(sum_df, metadata_df,
                          by = c("franchise", "revenue_category")) %>%
      distinct(.keep_all = TRUE)

    final_df
    ```
    """)
    return


@app.cell
def _(df_clean_1):
    df_sum = df_clean_1.groupby(['franchise', 'revenue_category']).sum().flatten_multiindex()
    df_sum.head(3)
    return (df_sum,)


@app.cell
def _(df_clean_1):
    df_metadata = df_clean_1[['franchise', 'revenue_category', 'original_media', 'creators']]
    df_metadata.head(3)
    return (df_metadata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Final Dataframe
    """)
    return


@app.cell
def _(df_metadata, df_sum, pd):
    # Generate final dataframe
    df_final = (
        pd.merge(df_sum, df_metadata, how="left", on=["franchise", "revenue_category"])
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    df_final.head(3)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
