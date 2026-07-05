def test_optional_dependency_imports_when_installed():
    import importlib

    try:
        importlib.import_module("py" + "recest")
    except ModuleNotFoundError:
        return
