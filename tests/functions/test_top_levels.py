import pandas as pd
import pytest

import janitor
from janitor.functions.top_levels import _get_level_groups


@pytest.fixture
def fac():
    levels = list("abcdef")[::-1]
    values = ["a", "b", "c", "d", "e", "f", "f"]
    return pd.Categorical(values, categories=levels, ordered=True)


@pytest.mark.functions
def test_top_levels_values(fac):
    result = janitor.top_levels(fac)
    assert result["n"].tolist() == [3, 2, 2]
    assert result["percent"].tolist() == pytest.approx([3 / 7, 2 / 7, 2 / 7])

    result_n3 = janitor.top_levels(fac, n=3)
    assert result_n3["n"].tolist() == [4, 3]
    assert result_n3["percent"].tolist() == pytest.approx([4 / 7, 3 / 7])


@pytest.mark.functions
def test_top_levels_odd_levels():
    levels = list("abcde")[::-1]
    values = ["a", "b", "c", "d", "e", "f", "f"]
    fac_odd = pd.Categorical(values, categories=levels, ordered=True)

    result = janitor.top_levels(fac_odd)
    assert result["n"].tolist() == [2, 1, 2]
    assert result["percent"].tolist() == pytest.approx([0.4, 0.2, 0.4])

    result_n1 = janitor.top_levels(fac_odd, n=1)
    assert result_n1["n"].tolist() == [1, 3, 1]
    assert result_n1["percent"].tolist() == pytest.approx([0.2, 0.6, 0.2])


@pytest.mark.functions
def test_top_levels_missing_levels():
    values = list("abc")
    levels = list("abcde")
    categorical = pd.Categorical(values, categories=levels, ordered=True)
    result = janitor.top_levels(categorical)
    assert result["n"].tolist() == [2, 1, 0]


@pytest.mark.functions
def test_top_levels_na_handling(fac):
    values = list(fac)
    values[-1] = None
    fac_na = pd.Categorical(values, categories=fac.categories, ordered=True)

    result = janitor.top_levels(fac_na, show_na=True)
    assert result["n"].tolist() == [2, 2, 2, 1]
    assert result["percent"].tolist() == pytest.approx(
        [2 / 7, 2 / 7, 2 / 7, 1 / 7]
    )
    assert result["valid_percent"].iloc[:3].tolist() == pytest.approx(
        [1 / 3] * 3
    )
    assert pd.isna(result["valid_percent"].iloc[3])


@pytest.mark.functions
def test_top_levels_column_name(fac):
    series = pd.Series(fac, name="fac")
    result = janitor.top_levels(series)
    assert result.columns[0] == "fac"


@pytest.mark.functions
def test_top_levels_type_errors():
    with pytest.raises(ValueError, match="factor_vec is not of type 'factor'"):
        janitor.top_levels([0, 1])
    with pytest.raises(ValueError, match="factor_vec is not of type 'factor'"):
        janitor.top_levels(pd.Series([0, 1]))


@pytest.mark.functions
def test_top_levels_invalid_n(fac):
    with pytest.raises(ValueError, match="double-counted"):
        janitor.top_levels(fac, n=4)
    with pytest.raises(ValueError, match="n must be a whole number at least 1"):
        janitor.top_levels(fac, n=0)
    with pytest.raises(ValueError, match="n must be a whole number at least 1"):
        janitor.top_levels(fac, n=1.5)

    small = pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True)
    with pytest.raises(
        ValueError, match="input factor variable must have at least 3 levels"
    ):
        janitor.top_levels(small)


@pytest.mark.functions
def test_get_level_groups_short_labels():
    levels = list("abcdef")[::-1]
    groups = _get_level_groups(levels, 1, len(levels))
    assert groups == {"top": "f", "mid": "e, d, c, b", "bot": "a"}

    groups_n2 = _get_level_groups(levels, 2, len(levels))
    assert groups_n2 == {"top": "f, e", "mid": "d, c", "bot": "b, a"}

    groups_n3 = _get_level_groups(levels, 3, len(levels))
    assert groups_n3 == {"top": "f, e, d", "mid": None, "bot": "c, b, a"}


@pytest.mark.functions
def test_get_level_groups_truncation():
    levels = [
        "dddddddddddddddd",
        "aaaaaaaaaaaaaaaa",
        "cccccccccccccccccccc",
        "bbbbbbbbbbbbbbbbb",
        "hhhhhhhhhhhhhhhh",
    ]
    groups_n1 = _get_level_groups(levels, 1, len(levels))
    assert groups_n1 == {
        "top": "dddddddddddddddd",
        "mid": "<<< Middle Group (3 categories) >>>",
        "bot": "hhhhhhhhhhhhhhhh",
    }

    groups_n2 = _get_level_groups(levels, 2, len(levels))
    assert groups_n2["top"] == "dddddddddddddddd, aaaaaaaaa..."
    assert groups_n2["mid"] == "cccccccccccccccccccc"
    assert groups_n2["bot"] == "bbbbbbbbbbbbbbbbb, hhhhhhhh..."
    assert len(groups_n2["top"]) == 30
    assert len(groups_n2["bot"]) == 30
