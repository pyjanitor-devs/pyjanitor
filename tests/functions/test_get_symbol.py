from janitor import get_symbol


def test_get_symbol():
    assert get_symbol("GME") == "GameStop Corp."
    assert get_symbol("GME") != "Globus Medical Inc."
    assert get_symbol("F") == "Ford Motor Company"
    assert get_symbol("ZZZZ") is None
