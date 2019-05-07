import pytest
from tests.utils import idempotent
from math import fabs, floor
import numpy as np


@pytest.mark.functions
@pytest.mark.parametrize("func,data", [(fabs, -5), (floor, 10.45)])
def test__idempotence(func, data):
    idempotent(func, data)
