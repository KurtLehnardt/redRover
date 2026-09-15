"""Shared test fixtures.

Two guarantees this file enforces for every test:

1. **No test touches the developer's database.** ``config.database.path`` is
   redirected into a per-test temporary directory.
2. **No test reaches a real LLM.** The suite previously called
   ``run_patrol(skip_ai=False)``, which posted to ``localhost:11434``; on a
   machine with the configured model installed that turned a 20-second suite
   into a multi-minute one against a 26B model. ``OllamaClient`` is stubbed by
   default, and tests that genuinely want a live model must be marked ``llm``.
"""

from __future__ import annotations

import pytest

from src.ai.ollama import OllamaClient, OllamaUnavailable
from src.alerting import reset_alert_manager
from src.config import Settings, load_config


@pytest.fixture(autouse=True)
def _no_live_llm(request, monkeypatch):
    """Fail every Ollama call unless the test is marked ``llm``."""
    if request.node.get_closest_marker("llm"):
        return

    async def _refuse(self, *args, **kwargs):
        raise OllamaUnavailable("Ollama access is disabled in tests (mark with @pytest.mark.llm)")

    async def _unavailable(self, *args, **kwargs):
        return False, []

    monkeypatch.setattr(OllamaClient, "chat", _refuse)
    monkeypatch.setattr(OllamaClient, "chat_json", _refuse)
    monkeypatch.setattr(OllamaClient, "vision_json", _refuse)
    monkeypatch.setattr(OllamaClient, "available", _unavailable)


@pytest.fixture(autouse=True)
def _deterministic_simulator():
    """Pin the simulator's RNG so statistical assertions are reproducible.

    The generators previously drew from global ``np.random``, which made every
    signal-shape assertion in this suite a coin flip with no way to reproduce a
    failure.
    """
    from src.sensors import simulator

    simulator.set_seed(20260914)
    yield
    simulator.set_seed(None)


@pytest.fixture(autouse=True)
def _reset_globals():
    """Keep process-wide singletons from leaking between tests."""
    reset_alert_manager()
    yield
    reset_alert_manager()


@pytest.fixture
def test_config(tmp_path) -> Settings:
    """A Settings object whose database lives in a temp directory."""
    config = load_config()
    config.database.path = str(tmp_path / "redRover.db")
    config.telemetry.enabled = False
    config.dashboard.auth_token = "test-token"
    # Simulated delays exist to pace a demo, not to slow the suite down.
    config.simulation.time_scale = 0.0
    return config


@pytest.fixture
async def test_db(tmp_path):
    """An initialised Database in a temp directory, closed on teardown."""
    from src.database import Database

    db = Database(str(tmp_path / "test.db"))
    await db.init()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
def stub_llm(monkeypatch):
    """Let a test supply a canned Ollama JSON response.

    Usage::

        def test_x(stub_llm):
            stub_llm({"overall_health": "critical", ...})
    """
    def _install(payload: dict):
        async def _chat_json(self, *args, **kwargs):
            return payload
        monkeypatch.setattr(OllamaClient, "chat_json", _chat_json)
    return _install


def pytest_collection_modifyitems(config, items):
    """Give every test a default timeout so a hang cannot wedge CI.

    Only applied when pytest-timeout is installed; the marker is otherwise
    inert and pytest warns about it.
    """
    if not config.pluginmanager.hasplugin("timeout"):
        return
    for item in items:
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(120))
