import numpy as np
import pytest


@pytest.mark.functions
def test_no_descriptions(dataframe):
    df = dataframe.data_description.df
    assert df.index.name == "column_name"
    assert (df["description"].str.len() == 0).all()
    assert (df["count"] == len(dataframe)).all()
    assert (df["pct_missing"] == 0).all()


@pytest.mark.functions
def test_description_list(dataframe):
    desc = ["A", "B", "C", "D", "E"]

    dataframe.data_description.set_description(desc)
    df = dataframe.data_description.df

    assert (df["description"] == desc).all()

    with pytest.raises(ValueError):
        dataframe.data_description.set_description([])

    with pytest.raises(ValueError):
        dataframe.data_description.set_description(desc[0:3])


@pytest.mark.functions
def test_description_full_dict(dataframe):
    desc = {
        "a": "First",
        "Bell__Chart": "Second",
        "decorated-elephant": "Third",
        "animals@#$%^": "Fourth",
        "cities": "Fifth",
    }

    dataframe.data_description.set_description(desc)
    df = dataframe.data_description.df

    assert not (df["description"].str.len() == 0).any()

    for k, v in desc.items():
        assert df.loc[k]["description"] == v


@pytest.mark.functions
def test_description_partial_dict(dataframe):
    desc = {"a": "First", "decorated-elephant": "Third", "cities": "Fifth"}

    dataframe.data_description.set_description(desc)
    df = dataframe.data_description.df

    assert len(df[df["description"].apply(lambda x: len(x) == 0)]) == 2

    for k, v in desc.items():
        assert df.loc[k]["description"] == v


@pytest.mark.functions
def test_null_rows(dataframe):
    dataframe = dataframe.copy()
    dataframe.loc[1] = None
    dataframe.loc[4] = None
    dataframe.loc[6] = None

    df = dataframe.data_description.df
    assert (df["count"] == 6).all()
    assert np.isclose(df["pct_missing"], (3 / 9)).all()


@pytest.mark.functions
def test_null_values(dataframe):
    dataframe = dataframe.copy()
    dataframe.loc[1, "a"] = None
    dataframe.loc[4, "a"] = None
    dataframe.loc[3, "cities"] = None
    dataframe.loc[3, "Bell__Chart"] = None
    dataframe.loc[6, "decorated-elephant"] = None

    df = dataframe.data_description.df
    assert df.loc["a", "count"] == 7
    assert np.isclose(df.loc["a", "pct_missing"], 2 / 9)

    assert df.loc["cities", "count"] == 8
    assert np.isclose(df.loc["cities", "pct_missing"], 1 / 9)

    assert df.loc["Bell__Chart", "count"] == 8
    assert np.isclose(df.loc["Bell__Chart", "pct_missing"], 1 / 9)

    assert df.loc["decorated-elephant", "count"] == 8
    assert np.isclose(df.loc["decorated-elephant", "pct_missing"], 1 / 9)


@pytest.mark.functions
def test_display(dataframe):
    dataframe.data_description.display()
