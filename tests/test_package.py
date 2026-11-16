import importlib

import dataquorum

SUBPACKAGES = [
    "core",
    "sandbox",
    "agents",
    "ml",
    "ml.automl",
    "supervisor",
    "evals",
    "tracing",
    "app",
]


def test_version_matches_metadata() -> None:
    from importlib.metadata import version

    assert dataquorum.__version__ == version("dataquorum")


def test_subpackages_import() -> None:
    for name in SUBPACKAGES:
        module = importlib.import_module(f"dataquorum.{name}")
        assert module.__doc__, f"dataquorum.{name} is missing a docstring"
