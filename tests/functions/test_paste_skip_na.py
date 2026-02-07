import pytest

import janitor


@pytest.mark.functions
def test_paste_skip_na_basic():
    assert janitor.paste_skip_na("A", None) == "A"
    assert janitor.paste_skip_na("A", None, ["B", None], sep=",") == ["A,B", "A"]


@pytest.mark.functions
def test_paste_skip_na_all_missing():
    assert janitor.paste_skip_na(None, None, None) is None


@pytest.mark.functions
def test_paste_skip_na_length_mismatch_error():
    with pytest.raises(
        ValueError,
        match="Arguments must be the same length or one argument must be a scalar.",
    ):
        janitor.paste_skip_na(["A", "B"], ["C", "D", "E"])


@pytest.mark.functions
def test_paste_skip_na_collapse():
    assert janitor.paste_skip_na(["A", None, "B"], collapse=",") == "A,B"


@pytest.mark.functions
def test_paste_skip_na_recycles_scalar():
    assert janitor.paste_skip_na("A", ["B", "C"], sep="-") == ["A-B", "A-C"]
