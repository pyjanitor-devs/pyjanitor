from janitor.functions import convert_stock

def test_convert_stock():
    assert(get_symbol("GME") == "GameStop Corp.")
    assert(get_symbol("AAPL") != "Aramark" )
    assert(get_symbol("ASNF") == "Not found.")
