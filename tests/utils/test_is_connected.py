import socket

import pytest

from janitor.utils import is_connected

"""
Tests the is_connected helper function,
    which is a function to check if the client
    is connected to the internet.

Example:
    print(is_connected("www.google.com"))
    console >> True

Test 1: happy path, ensures function work

Test 2: web addresses that are not recognized
    will return false (comzzz is not a tld).

Test 3: web addresses that are not recognized
    will return false (aadsfff.com does not exist
    at time of testing).

If test 3 fails, perhaps this is because
    the website now exists. If that is the case,
    alter or delete the test.
"""


def test_is_connected():
    assert is_connected("www.google.com")
    with pytest.raises(socket.gaierror):
        assert is_connected("www.google.comzzz") is False
    with pytest.raises(socket.gaierror):
        assert is_connected("aadsfff.com") is False
