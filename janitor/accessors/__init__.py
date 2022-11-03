"""Miscellaneous mathematical operators.

Lazy loading used here to speed up imports.
"""

import warnings
from typing import Tuple


import lazy_loader as lazy

scipy_special = lazy.load("scipy.special")
ss = lazy.load("scipy.stats")
pf = lazy.load("pandas_flavor")
pd = lazy.load("pandas")
np = lazy.load("numpy")
pdtypes = lazy.load("pandas.api.types")
