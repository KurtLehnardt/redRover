"""End-to-end and regression tests.

These exercise the full patrol against the simulator. ``conftest`` redirects
the database into a temp directory and blocks Ollama, so a full run is
deterministic, offline, and finishes in seconds.
"""

from __future__ import annotations

import pytest

from src.ai.faults import FaultCode
from src.ai.fusion import FusionAnalyzer, OverallHealth
from src.database import Database
from src.main import run_patrol
from src.sensors.acoustic import AcousticFaultType
from src.sensors.simulator import (
    generate_acoustic_sample,
    generate_sample,
    generate_thermal_frame,
)
from src.sensors.thermal import ThermalFaultType
from src.sensors.vibration import FaultType

ANALYZER = FusionAnalyzer(model="gemma3", ollama_host="http://localhost:99999")


# === End-to-End Tests ===


@pytest.mark.asyncio
async def test_full_patrol_completes(test_config):
    results = await run_patrol(simulate=True, skip_ai=True, config=test_config)
    assert results == []  # skip_ai produces no diagnoses


@pytest.mark.asyncio
async def test_full_patrol_with_fusion(test_config):
    results = await run_patrol(simulate=True, skip_ai=False, config=test_config)
    assert len(results) == len(test_config.route.waypoints)


@pytest.mark.asyncio
async def test_full_patrol_detects_faults(test_config):
    results = await run_patrol(simulate=True, skip_ai=False, config=test_config)
    faults = [r for r in results if r.overall_health is not OverallHealth.HEALTHY]
    assert len(faults) >= 1


@pytest.mark.asyncio
async def test_patrol_creates_db_records(test_config):
    await run_patrol(simulate=True, skip_ai=True, config=test_config)
    db = Database(test_config.database.path)
    await db.init()
    try:
        patrols = await db.get_recent_patrols(1)
        assert len(patrols) >= 1
        assert patrols[0]["completed_at"] is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_patrol_persists_diagnoses(test_config):
    """Regression: the patrol used to log measurements but never diagnoses.

    With no diagnosis rows the dashboard's fault list and the fusion engine's
    trend context were both permanently empty.
    """
    await run_patrol(simulate=True, skip_ai=False, config=test_config)

    db = Database(test_config.database.path)
    await db.init()
    try:
        active = await db.get_active_faults()
        assert active, "patrol produced no diagnosis rows"
        # Every persisted fault must be a canonical code the rule engine can
        # match against on the next patrol.
        for row in active:
            assert FaultCode.parse(row["fault_type"]) is not FaultCode.UNKNOWN
            assert row["health"] in {h.value for h in OverallHealth}

        trend = await db.get_station_trend(active[0]["station_id"], limit=5)
        assert trend and trend[-1]["fault_type"] is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_patrol_marks_simulated_rows(test_config):
    """A simulated run must be distinguishable from a real one in the data."""
    await run_patrol(simulate=True, skip_ai=False, config=test_config)

    db = Database(test_config.database.path)
    await db.init()
    try:
        history = await db.get_station_history(
            test_config.route.waypoints[0].station_id, limit=5,
        )
        assert history
        assert history[0]["simulated"] == 1
        assert "simulated" in (history[0]["source_name"] or "")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_patrol_syncs_stations_from_config(test_config):
    await run_patrol(simulate=True, skip_ai=True, config=test_config)
    db = Database(test_config.database.path)
    await db.init()
    try:
        stations = await db.get_stations()
        assert {s["id"] for s in stations} == {
            w.station_id for w in test_config.route.waypoints
        }
    finally:
        await db.close()


# === Regression Tests (known fault signatures) ===


@pytest.mark.asyncio
async def test_regression_bearing_outer_detected():
    vib = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.9)
    result = ANALYZER._analyze_vibration(vib)
    assert result.fault_detected
    assert result.code is FaultCode.BEARING_FAULT


@pytest.mark.asyncio
async def test_regression_air_leak_detected():
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.8)
    result = ANALYZER._analyze_acoustic(aco)
    assert result.fault_detected
    assert result.code is FaultCode.AIR_LEAK


@pytest.mark.asyncio
async def test_regression_overheating_detected():
    thm = generate_thermal_frame(fault_type=ThermalFaultType.OVERHEATING, severity=1.0)
    result = ANALYZER._analyze_thermal(thm)
    assert result.fault_detected
    assert result.code in (FaultCode.OVERHEATING, FaultCode.HOTSPOT)


@pytest.mark.asyncio
async def test_regression_normal_stays_normal():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("R-001", vibration=vib, acoustic=aco, thermal=thm)
    assert result.overall_health is OverallHealth.HEALTHY
    assert result.correlated_faults == []
