from janitor.functions import is_connected
import pytest

def test_is_connected():
    assert (is_connected("www.google.com"))
    assert (is_connected("www.google.comzzz") == False)
    assert (is_connected("www.facebook.com"))
    assert (is_connected("aadsfff.com") == False)
