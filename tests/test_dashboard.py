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


# === The UI must be able to drive its own API ===
#
# Regression: the page posted to /api/demo-patrol with no credentials at all,
# so adding token auth locked the dashboard out of its own button. The tests
# asserted the lock and never the key.


def test_index_offers_an_unlock_form_when_locked(client):
    body = client.get("/").text
    assert 'id="unlock-form"' in body
    assert 'id="demo-patrol"' not in body


def test_index_shows_controls_once_unlocked(client):
    assert client.post("/api/session", json={"token": "s3cret"}).status_code == 200
    body = client.get("/").text
    assert 'id="demo-patrol"' in body
    assert 'id="estop"' in body
    assert 'id="unlock-form"' not in body


def test_index_explains_why_controls_are_off(monkeypatch, tmp_path):
    from src.dashboard import app as dashboard

    monkeypatch.setattr(dashboard.config.dashboard, "auth_token", "")
    monkeypatch.setattr(dashboard.db, "path", tmp_path / "off.db")
    with TestClient(dashboard.app) as c:
        body = c.get("/").text
        assert "auth_token" in body
        assert 'id="unlock-form"' not in body
        assert 'id="demo-patrol"' not in body


def test_session_cookie_is_httponly_and_samesite(client):
    response = client.post("/api/session", json={"token": "s3cret"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    # HttpOnly keeps the token away from page scripts; SameSite=strict is what
    # makes a cookie safe on an API that can drive a robot.
    assert "httponly" in cookie
    assert "samesite=strict" in cookie


def test_session_rejects_a_wrong_token(client):
    assert client.post("/api/session", json={"token": "nope"}).status_code == 401
    assert client.post("/api/estop").status_code == 401


def test_session_unlocks_the_control_endpoints(client):
    assert client.post("/api/estop").status_code == 401
    client.post("/api/session", json={"token": "s3cret"})
    assert client.post("/api/estop").status_code == 200


def test_logout_relocks(client):
    client.post("/api/session", json={"token": "s3cret"})
    assert client.post("/api/estop").status_code == 200
    client.post("/api/session/logout")
    assert client.post("/api/estop").status_code == 401


def test_session_endpoint_is_off_without_a_configured_token(monkeypatch, tmp_path):
    from src.dashboard import app as dashboard

    monkeypatch.setattr(dashboard.config.dashboard, "auth_token", "")
    monkeypatch.setattr(dashboard.db, "path", tmp_path / "off2.db")
    with TestClient(dashboard.app) as c:
        assert c.post("/api/session", json={"token": "anything"}).status_code == 503


# === The page must not depend on the internet ===


def test_dashboard_loads_no_third_party_scripts():
    """Regression: the page pulled htmx from unpkg.com.

    This is an appliance that advertises no cloud dependency; a CDN script is
    both a supply-chain surface and a reason the UI goes blank when the factory
    network is down.
    """
    import re
    from pathlib import Path

    templates = Path(__file__).resolve().parent.parent / "src/dashboard/templates"
    for path in templates.glob("*.html"):
        body = path.read_text()
        external = re.findall(r'(?:src|href)\s*=\s*"(https?:)?//[^"]+"', body)
        assert external == [], f"{path.name} loads {external}"


def test_faults_endpoint_is_json_and_the_page_renders_it_client_side():
    """Regression: the page swapped this JSON response into innerHTML.

    The fault table was correct for exactly 30 seconds, then became a literal
    JSON dump.
    """
    from pathlib import Path

    index = (Path(__file__).resolve().parent.parent
             / "src/dashboard/templates/index.html").read_text()
    assert "hx-get" not in index and "hx-swap" not in index
    # It fetches the JSON and builds the rows itself.
    assert 'fetch("/api/faults"' in index
    assert "renderFaults" in index
