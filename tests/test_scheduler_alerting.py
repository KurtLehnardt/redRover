"""Tests for scheduler and alerting modules."""

from datetime import time

import pytest

from src.ai.fusion import FusedDiagnosis, ModalityResult, OverallHealth
from src.alerting import AlertManager
from src.scheduler import _in_quiet_hours

# === Quiet Hours Logic ===
#
# These replace two tests that asserted nothing: one was
# `assert _in_quiet_hours.__doc__ is None or True` -- a tautology -- and the
# other was a bare `pass`. The overnight wrap is the scheduler's only real
# branch, and it sat behind two green checkmarks untested.


@pytest.mark.parametrize(
    "at,expected",
    [
        ("22:00", True),  # exactly the start
        ("23:30", True),  # before midnight
        ("00:00", True),  # midnight itself
        ("03:00", True),  # after midnight
        ("06:00", True),  # exactly the end
        ("06:01", False),  # just past
        ("12:00", False),  # the middle of the working day
        ("21:59", False),  # just before
    ],
)
def test_quiet_hours_overnight_window(at, expected):
    """22:00-06:00 wraps past midnight."""
    assert _in_quiet_hours("22:00", "06:00", time.fromisoformat(at)) is expected


@pytest.mark.parametrize(
    "at,expected",
    [
        ("08:00", True),
        ("12:30", True),
        ("17:00", True),
        ("07:59", False),
        ("17:01", False),
        ("23:00", False),
        ("02:00", False),
    ],
)
def test_quiet_hours_daytime_window(at, expected):
    """08:00-17:00 does not wrap, so the simple comparison applies."""
    assert _in_quiet_hours("08:00", "17:00", time.fromisoformat(at)) is expected


def test_quiet_hours_defaults_to_now():
    """Called without a time it still answers, using the local clock."""
    assert _in_quiet_hours("00:00", "23:59") is True
    assert isinstance(_in_quiet_hours("22:00", "06:00"), bool)


def test_quiet_hours_zero_length_window():
    """A start equal to the end admits exactly that instant."""
    assert _in_quiet_hours("03:00", "03:00", time.fromisoformat("03:00")) is True
    assert _in_quiet_hours("03:00", "03:00", time.fromisoformat("03:01")) is False


# === AlertManager ===


@pytest.mark.asyncio
async def test_alert_critical_logged():
    """Critical diagnosis triggers an alert."""
    mgr = AlertManager()
    diagnosis = FusedDiagnosis(
        station_id="A-001",
        overall_health=OverallHealth.CRITICAL,
        overall_confidence=0.92,
        modality_results=[
            ModalityResult("vibration", True, "bearing_fault", 0.90, "severe"),
        ],
        correlated_faults=["bearing_fault"],
        recommendation="Replace bearing immediately",
        priority=1,
        reasoning="Test",
        inference_mode="rule_based",
    )
    await mgr.evaluate(diagnosis)
    assert len(mgr.recent_alerts) == 1
    assert mgr.recent_alerts[0]["level"] == "CRITICAL"
    assert mgr.recent_alerts[0]["station_id"] == "A-001"


@pytest.mark.asyncio
async def test_alert_warning_logged():
    """Warning diagnosis triggers an alert."""
    mgr = AlertManager()
    diagnosis = FusedDiagnosis(
        station_id="A-002",
        overall_health=OverallHealth.WARNING,
        overall_confidence=0.75,
        modality_results=[],
        correlated_faults=["air_leak"],
        recommendation="Seal leak",
        priority=2,
        reasoning="Test",
        inference_mode="rule_based",
    )
    await mgr.evaluate(diagnosis)
    assert len(mgr.recent_alerts) == 1
    assert mgr.recent_alerts[0]["level"] == "WARNING"


@pytest.mark.asyncio
async def test_alert_healthy_not_logged():
    """Healthy diagnosis does NOT trigger an alert."""
    mgr = AlertManager()
    diagnosis = FusedDiagnosis(
        station_id="A-003",
        overall_health=OverallHealth.HEALTHY,
        overall_confidence=0.95,
        modality_results=[],
        correlated_faults=[],
        recommendation="All clear",
        priority=4,
        reasoning="Test",
        inference_mode="rule_based",
    )
    await mgr.evaluate(diagnosis)
    assert len(mgr.recent_alerts) == 0


@pytest.mark.asyncio
async def test_alert_monitor_not_logged():
    """Monitor-level diagnosis does NOT trigger an alert."""
    mgr = AlertManager()
    diagnosis = FusedDiagnosis(
        station_id="A-004",
        overall_health=OverallHealth.MONITOR,
        overall_confidence=0.60,
        modality_results=[],
        correlated_faults=["imbalance"],
        recommendation="Watch closely",
        priority=3,
        reasoning="Test",
        inference_mode="rule_based",
    )
    await mgr.evaluate(diagnosis)
    assert len(mgr.recent_alerts) == 0


@pytest.mark.asyncio
async def test_alert_history_accumulates():
    """Multiple alerts accumulate in history."""
    mgr = AlertManager()
    for index in range(5):
        diagnosis = FusedDiagnosis(
            station_id=f"A-{index:03d}",
            overall_health=OverallHealth.CRITICAL,
            overall_confidence=0.90,
            modality_results=[],
            correlated_faults=["bearing_fault"],
            recommendation="Fix",
            priority=1,
            reasoning="Test",
            inference_mode="rule_based",
        )
        await mgr.evaluate(diagnosis)
    assert len(mgr.recent_alerts) == 5


@pytest.mark.asyncio
async def test_alert_webhook_failure_does_not_crash():
    """Webhook to bad URL should not raise."""
    mgr = AlertManager(webhook_url="http://localhost:99999/bad")
    diagnosis = FusedDiagnosis(
        station_id="A-005",
        overall_health=OverallHealth.CRITICAL,
        overall_confidence=0.90,
        modality_results=[],
        correlated_faults=["overheating"],
        recommendation="Cool down",
        priority=1,
        reasoning="Test",
        inference_mode="rule_based",
    )
    # Should not raise despite bad webhook URL
    await mgr.evaluate(diagnosis)
    assert len(mgr.recent_alerts) == 1
