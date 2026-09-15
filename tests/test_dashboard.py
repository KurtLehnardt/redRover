"""Dashboard security: the API can move the robot, so it must be guarded."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config import Settings


@pytest.fixture
def client(monkeypatch, tmp_path):
    from src.dashboard import app as dashboard

    monkeypatch.setattr(dashboard.config.dashboard, "auth_token", "s3cret")
    monkeypatch.setattr(dashboard.config.database, "path", str(tmp_path / "dash.db"))
    monkeypatch.setattr(dashboard.db, "path", tmp_path / "dash.db")
    with TestClient(dashboard.app) as c:
        yield c


def test_config_rejects_wildcard_cors():
    """'*' plus a robot-driving API is not a combination we allow."""
    with pytest.raises(ValueError, match="allowed_origins"):
        Settings(dashboard={"allowed_origins": ["*"]})


def test_default_bind_is_loopback():
    assert Settings().dashboard.host == "127.0.0.1"


def test_remap_requires_a_token(client):
    response = client.post("/api/remap")
    assert response.status_code == 401


def test_remap_rejects_a_wrong_token(client):
    response = client.post("/api/remap", headers={"X-RedRover-Token": "nope"})
    assert response.status_code == 401


def test_estop_requires_a_token(client):
    assert client.post("/api/estop").status_code == 401


def test_demo_patrol_requires_a_token(client):
    assert client.post("/api/demo-patrol").status_code == 401


def test_estop_accepts_a_valid_token(client):
    response = client.post("/api/estop", headers={"X-RedRover-Token": "s3cret"})
    assert response.status_code == 200
    assert response.json()["status"] == "estop_engaged"


def test_bearer_token_is_accepted(client):
    response = client.post("/api/estop", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 200


def test_read_only_endpoints_stay_open(client):
    assert client.get("/api/faults").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_control_disabled_without_a_configured_token(monkeypatch, tmp_path):
    """No token configured means the endpoints are off, not unauthenticated."""
    from src.dashboard import app as dashboard

    monkeypatch.setattr(dashboard.config.dashboard, "auth_token", "")
    monkeypatch.setattr(dashboard.db, "path", tmp_path / "dash2.db")
    with TestClient(dashboard.app) as c:
        response = c.post("/api/remap")
        assert response.status_code == 503
        assert "auth_token" in response.json()["detail"]


def test_alerts_endpoint_shares_the_process_alert_manager(client):
    """Regression: /api/alerts built a fresh manager and always returned []."""
    import asyncio

    from src.ai.fusion import FusedDiagnosis, ModalityResult, OverallHealth
    from src.alerting import get_alert_manager

    manager = get_alert_manager()
    asyncio.run(
        manager.evaluate(FusedDiagnosis(
            station_id="A-1",
            overall_health=OverallHealth.CRITICAL,
            overall_confidence=0.9,
            modality_results=[ModalityResult("vibration", True, "bearing_fault", 0.9, "severe")],
            correlated_faults=["bearing_fault"],
            recommendation="replace",
            priority=1,
            reasoning="test",
        ))
    )

    payload = client.get("/api/alerts").json()
    assert any(a["station_id"] == "A-1" for a in payload)
