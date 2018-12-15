def import_message(submodule, package, installation):
    print(
        f"To use the janitor submodule {submodule}, you need to install {package}."
    )
    print()
    print(f"To do so, use the following command:")
    print()
    print(f"    {installation}")
