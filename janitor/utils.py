def import_message(submodule, package, installation):
    print(
        f"To use the janitor submodule {submodule}, you need to install \
        {package}."
    )
    print()
    print(f"To do so, use the following command:")
    print()
    print(f"    {installation}")


def idempotent(func, df, *args, **kwargs):
    """
        Checks if a function is idempotence,
        that is f(f(x))=f(x) is true for all x.

    :param func: a python method
    :param df: a pandas dataframe
    :param args: Arguments supplied to the method
    :param kwargs: Arguments supplied to the method
    :return:
    """
    assert func(df, *args, **kwargs) == func(
        func(df, *args, **kwargs), *args, **kwargs
    )
