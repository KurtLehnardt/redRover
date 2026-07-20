"""Tests for fusion analyzer, trend escalation, and database."""

import os
import uuid
import pytest

from src.sensors.vibration import FaultType
from src.sensors.acoustic import AcousticFaultType
from src.sensors.thermal import ThermalFaultType
from src.sensors.simulator import (
    generate_sample,
    generate_acoustic_sample,
    generate_thermal_frame,
)
from src.ai.fusion import FusionAnalyzer, OverallHealth, FusedDiagnosis
from src.database import Database


# Use a fake ollama host so LLM always fails → rule-based fallback
ANALYZER = FusionAnalyzer(model="gemma3", ollama_host="http://localhost:99999")


# === Fusion Rule-Based Logic ===

@pytest.mark.asyncio
async def test_fusion_all_normal():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("T-001", vibration=vib, acoustic=aco, thermal=thm)
    assert result.overall_health == OverallHealth.HEALTHY
    assert result.overall_confidence >= 0.8


@pytest.mark.asyncio
async def test_fusion_vibration_fault_only():
    vib = generate_sample(fault_type=FaultType.IMBALANCE, severity=0.6)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("T-002", vibration=vib, acoustic=aco, thermal=thm)
    assert result.overall_health in (OverallHealth.MONITOR, OverallHealth.WARNING)


@pytest.mark.asyncio
async def test_fusion_vibration_plus_thermal():
    vib = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.8)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.7)
    result = await ANALYZER.analyze("T-003", vibration=vib, acoustic=aco, thermal=thm)
    # Should correlate mechanical + thermal
    all_faults = " ".join(result.correlated_faults)
    has_correlation = ("mechanical" in all_faults or "heating" in all_faults
                       or len(result.correlated_faults) >= 2)
    assert has_correlation or result.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL)


@pytest.mark.asyncio
async def test_fusion_acoustic_only_leak():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.7)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("T-004", vibration=vib, acoustic=aco, thermal=thm)
    all_faults = " ".join(result.correlated_faults)
    assert "air_leak" in all_faults or "electrical" in all_faults


@pytest.mark.asyncio
async def test_fusion_multiple_faults_elevated_health():
    vib = generate_sample(fault_type=FaultType.MISALIGNMENT, severity=0.7)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.7)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.7)
    result = await ANALYZER.analyze("T-005", vibration=vib, acoustic=aco, thermal=thm)
    assert result.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL)
    assert result.priority <= 2


@pytest.mark.asyncio
async def test_fusion_confidence_boost():
    # Single fault
    vib = generate_sample(fault_type=FaultType.MISALIGNMENT, severity=0.7)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    single = await ANALYZER.analyze("T-006a", vibration=vib, acoustic=aco, thermal=thm)

    # Multiple faults
    aco2 = generate_acoustic_sample(fault_type=AcousticFaultType.FRICTION, severity=0.5)
    thm2 = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.5)
    multi = await ANALYZER.analyze("T-006b", vibration=vib, acoustic=aco2, thermal=thm2)

    assert multi.overall_confidence >= single.overall_confidence


@pytest.mark.asyncio
async def test_fusion_single_modality():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    result = await ANALYZER.analyze("T-007", vibration=vib)
    assert isinstance(result, FusedDiagnosis)
    assert result.inference_mode == "degraded"


@pytest.mark.asyncio
async def test_fusion_no_modalities():
    result = await ANALYZER.analyze("T-008")
    assert result.overall_health == OverallHealth.HEALTHY
    assert result.overall_confidence == 0.0
    assert result.inference_mode == "degraded"


def test_fusion_inference_mode_rule_based():
    """Direct call to _rule_based_fusion should set inference_mode."""
    vib_result = ANALYZER._analyze_vibration(
        generate_sample(fault_type=FaultType.NORMAL)
    )
    thm_result = ANALYZER._analyze_thermal(
        generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    )
    result = ANALYZER._rule_based_fusion("T-009", [vib_result, thm_result])
    assert result.inference_mode == "rule_based"


@pytest.mark.asyncio
async def test_fusion_recommendation_bearing():
    vib = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.9)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.FRICTION, severity=0.5)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.6)
    result = await ANALYZER.analyze("T-010", vibration=vib, acoustic=aco, thermal=thm)
    # Recommendation should mention something actionable
    assert len(result.recommendation) > 10


@pytest.mark.asyncio
async def test_fusion_recommendation_air_leak():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.7)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("T-011", vibration=vib, acoustic=aco, thermal=thm)
    rec = result.recommendation.lower()
    assert "leak" in rec or "$" in rec or "energy" in rec or "seal" in rec


# === Trend Escalation ===

@pytest.mark.asyncio
async def test_fusion_trend_escalation():
    """Persistent fault in history should escalate severity."""
    history = [
        {"measured_at": "2026-07-17", "rms": 0.6, "peak": 1.4, "kurtosis": -0.1,
         "crest_factor": 2.2, "fault_type": "hotspot", "severity": "moderate", "confidence": 0.75},
        {"measured_at": "2026-07-18", "rms": 0.65, "peak": 1.5, "kurtosis": 0.0,
         "crest_factor": 2.3, "fault_type": "hotspot", "severity": "moderate", "confidence": 0.78},
        {"measured_at": "2026-07-19", "rms": 0.7, "peak": 1.6, "kurtosis": 0.1,
         "crest_factor": 2.4, "fault_type": "hotspot", "severity": "moderate", "confidence": 0.80},
    ]
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.5)
    result = await ANALYZER.analyze("T-012", vibration=vib, acoustic=aco, thermal=thm,
                                     station_history=history)
    assert "TRENDING" in result.recommendation or result.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL)


@pytest.mark.asyncio
async def test_fusion_no_escalation_without_history():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.5)
    result = await ANALYZER.analyze("T-013", vibration=vib, acoustic=aco, thermal=thm,
                                     station_history=None)
    assert "TRENDING" not in result.recommendation


# === Database Tests ===

@pytest.fixture
async def test_db():
    path = f"/tmp/test_redrover_{uuid.uuid4().hex[:8]}.db"
    db = Database(path)
    await db.init()
    yield db
    os.unlink(path)


@pytest.mark.asyncio
async def test_db_init_creates_tables(test_db):
    import aiosqlite
    async with aiosqlite.connect(test_db.path) as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]
    assert "patrols" in tables
    assert "measurements" in tables
    assert "diagnoses" in tables
    assert "stations" in tables


@pytest.mark.asyncio
async def test_db_patrol_lifecycle(test_db):
    patrol_id = await test_db.start_patrol("Test Route", "2026-07-19T00:00:00")
    assert patrol_id > 0

    features = {
        "rms": 0.39, "peak": 0.78, "crest_factor": 2.0, "kurtosis": -1.2,
        "dominant_frequency_hz": 30.0, "energy_0_100hz": 0.1,
        "energy_100_500hz": 0.01, "energy_500_1000hz": 0.001, "energy_1000_2000hz": 0.0001,
    }
    m_id = await test_db.log_measurement(patrol_id, "M-001", "2026-07-19T00:01:00", features)
    assert m_id > 0

    await test_db.complete_patrol(patrol_id, "2026-07-19T00:05:00", 1, 0)
    patrols = await test_db.get_recent_patrols(5)
    assert len(patrols) == 1
    assert patrols[0]["stations_visited"] == 1


@pytest.mark.asyncio
async def test_db_station_history(test_db):
    patrol_id = await test_db.start_patrol("Test Route", "2026-07-19T00:00:00")
    features = {
        "rms": 0.5, "peak": 1.0, "crest_factor": 2.0, "kurtosis": 0.5,
        "dominant_frequency_hz": 30.0, "energy_0_100hz": 0.1,
        "energy_100_500hz": 0.01, "energy_500_1000hz": 0.001, "energy_1000_2000hz": 0.0001,
    }
    await test_db.log_measurement(patrol_id, "M-001", "2026-07-19T00:01:00", features)
    await test_db.log_measurement(patrol_id, "M-001", "2026-07-19T00:02:00", features)

    history = await test_db.get_station_history("M-001", 10)
    assert len(history) == 2


@pytest.mark.asyncio
async def test_db_station_trend(test_db):
    patrol_id = await test_db.start_patrol("Test Route", "2026-07-19T00:00:00")
    features = {
        "rms": 0.5, "peak": 1.0, "crest_factor": 2.0, "kurtosis": 0.5,
        "dominant_frequency_hz": 30.0, "energy_0_100hz": 0.1,
        "energy_100_500hz": 0.01, "energy_500_1000hz": 0.001, "energy_1000_2000hz": 0.0001,
    }
    await test_db.log_measurement(patrol_id, "M-001", "2026-07-19T00:01:00", features)
    await test_db.log_measurement(patrol_id, "M-001", "2026-07-19T00:02:00", features)

    trend = await test_db.get_station_trend("M-001", 5)
    assert len(trend) == 2
    # Should be oldest first
    assert trend[0]["measured_at"] <= trend[1]["measured_at"]
