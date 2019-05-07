import pytest
from janitor.utils import idempotent
from math import fabs, floor


@pytest.mark.functions
@pytest.mark.parametrize("func,data", [(fabs, -5), (floor, 10.45)])
def test__idempotence(func, data):
    idempotent(func, data)
