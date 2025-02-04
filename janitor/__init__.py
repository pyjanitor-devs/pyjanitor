"""Top-level janitor API lives here."""

import pandas_flavor as pf  # noqa: F401

from .accessors import *  # noqa: F403, F401
from .functions import *  # noqa: F403, F401
from .io import *  # noqa: F403, F401
from .math import *  # noqa: F403, F401
from .ml import get_features_targets as _get_features_targets
from .utils import refactored_function
from .xarray import *  # noqa: F403, F401


@refactored_function(
    "get_features_targets() has moved. Please use ml.get_features_targets()."
)
def get_features_targets(*args, **kwargs):
    """Wrapper for get_features_targets."""
    return _get_features_targets(*args, **kwargs)


__version__ = "0.30.0"
