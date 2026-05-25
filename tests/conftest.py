"""Shared fixtures and path setup for the test suite."""
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

# Make grossberg-rag source importable from the tests/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch tracing at module load time so chat.py's top-level tracing.init_tracing()
# call is a no-op when test modules import chat. Fixture-level patching fires too
# late (after test-module imports, which happen during pytest collection).
import tracing as _tracing_mod  # noqa: E402
_tracing_mod.init_tracing = lambda *a, **kw: None
_tracing_mod.shutdown_tracing = lambda: None
_tracing_mod.get_tracer = lambda: None


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that require live API keys.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="pass --integration to run live-API tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def patch_tracing(monkeypatch):
    """Replace tracing.span with a no-op context manager for all unit tests."""
    import tracing
    monkeypatch.setattr(tracing, "span", lambda *a, **kw: nullcontext(None))


@pytest.fixture(autouse=True)
def patch_tracing_init(monkeypatch):
    """Keep init/shutdown/tracer silenced per-test in case code re-calls them."""
    import tracing
    monkeypatch.setattr(tracing, "init_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(tracing, "shutdown_tracing", lambda: None)
    monkeypatch.setattr(tracing, "get_tracer", lambda: None)
