try:
    import janitor.xarray
except ImportError:
    pass

from .functions import *  # noqa: F403, F401
from .math import *
from .ml import get_features_targets as _get_features_targets
from .utils import refactored_function

# from .dataframe import JanitorDataFrame as DataFrame  # noqa: F401
# from .dataframe import JanitorSeries as Series  # noqa: F401


@refactored_function(
    "get_features_targets() has moved. Please use ml.get_features_targets()."
)
def get_features_targets(*args, **kwargs):
    return _get_features_targets(*args, **kwargs)


__version__ = "0.19.0"
