""" Backend functions for pyspark."""

from functools import wraps

try:
    from pyspark.pandas.extensions import register_dataframe_accessor

except ImportError:
    from janitor.utils import import_message

    import_message(
        submodule="spark",
        package="pyspark",
        conda_channel="conda-forge",
        pip_install=True,
    )


def register_dataframe_method(method):
    """Register a function as a method attached to the Pyspark DataFrame.

    !!! note

        Modified based on pandas_flavor.register.

    <!--
    # noqa: DAR101 method
    # noqa: DAR201
    -->
    """

    def inner(*args, **kwargs):
        class AccessorMethod:
            def __init__(self, pyspark_obj):
                self._obj = pyspark_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        register_dataframe_accessor(method.__name__)(AccessorMethod)

        return method

    return inner()
