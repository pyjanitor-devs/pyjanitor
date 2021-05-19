from janitor import get_symbol

'''
Tests get_symbol function,
get_symbol should return appropriate string corresponding to abbreviation

If the abbreviation does not have a corresponding string,
an exception should be raised to alert the user.

Test 1: GME is Gamestop Corp. Test should run fine.
Test 2: GME is not Globius Medical Inc. 
Test 3: A little redundant, but it's another 
'happy path' to show get_symbol works for more
abbreviations than just the one tested so far.
Test 4: ZZZZ does not belong to any company,
it should therefore it should be None
'''

def test_get_symbol():
    assert get_symbol("GME") == "GameStop Corp."
    assert get_symbol("GME") != "Globus Medical Inc."
    assert get_symbol("F") == "Ford Motor Company"
    assert get_symbol("ZZZZ") is None
