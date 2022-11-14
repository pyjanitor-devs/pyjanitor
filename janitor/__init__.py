"""Top-level janitor API lives here.

Lazy loading used here to speed up imports.
"""
import lazy_loader as lazy


from .accessors import *  # noqa: F403, F401
from .functions import *  # noqa: F403, F401
from .io import *  # noqa: F403, F401
from .math import *  # noqa: F403, F401
from .ml import get_features_targets as _get_features_targets
from .utils import refactored_function


_ = lazy.load("pandas_flavor")

try:
    jxr = lazy.load("janitor.xarray")  # noqa: F401
except ImportError:
    pass


@refactored_function(
    "get_features_targets() has moved. Please use ml.get_features_targets()."
)
def get_features_targets(*args, **kwargs):
    """Wrapper for get_features_targets."""
    return _get_features_targets(*args, **kwargs)


__version__ = "0.24.0"
