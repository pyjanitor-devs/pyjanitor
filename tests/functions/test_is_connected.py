from janitor.functions import is_connected

def test_is_connected():
    print("Starting tests...")
    assert (is_connected("www.google.com"))
    assert (is_connected("www.google.comzzz") == False)
    assert (is_connected("www.facebook.com"))
    assert (is_connected("aadsfff.com") == False)
    print("Passed all tests")
