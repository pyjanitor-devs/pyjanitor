from janitor import get_symbol

'''
Tests get_symbol function,
get_symbol should return appropriate string 
corresponding to abbreviation.
This string will be a company's full name,
and the abbreviation will be the NSYE 
symbol for the company.

Example:
    print(get_symbol("aapl"))
    console >> Apple Inc.

If the symbol does not have a corresponding 
company, Nonetype should be returned.

'''

def test_convert_stock():
    assert get_symbol("GME") == "GameStop Corp."
    assert get_symbol("AAPL") != "Aramark"
    assert get_symbol("ASNF") is None
