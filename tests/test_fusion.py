"""Tests for fusion analyzer, trend escalation, and database."""

import pytest

from src.ai.faults import CorrelationTag, FaultCode
from src.ai.fusion import FusedDiagnosis, FusionAnalyzer, OverallHealth
from src.database import DiagnosisRecord
from src.sensors.acoustic import AcousticFaultType
from src.sensors.simulator import (
    generate_acoustic_sample,
    generate_sample,
    generate_thermal_frame,
)
from src.sensors.thermal import ThermalFaultType
from src.sensors.vibration import FaultType

# conftest stubs OllamaClient, so every analyze() here exercises the
# rule-based fusion path deterministically and without network access.
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
    # Fault codes are canonical, and the cross-modal relationship is a tag.
    assert FaultCode.BEARING_FAULT.value in result.correlated_faults
    assert CorrelationTag.MECHANICAL_WITH_HEATING.value in result.correlation_tags
    assert result.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL)


@pytest.mark.asyncio
async def test_fusion_acoustic_only_leak():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.7)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("T-004", vibration=vib, acoustic=aco, thermal=thm)
    assert FaultCode.AIR_LEAK.value in result.correlated_faults


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
    assert "TRENDING" in result.recommendation
    assert CorrelationTag.TRENDING_WORSE.value in result.correlation_tags


@pytest.mark.asyncio
async def test_fusion_no_escalation_without_history():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.5)
    result = await ANALYZER.analyze("T-013", vibration=vib, acoustic=aco, thermal=thm,
                                     station_history=None)
    assert "TRENDING" not in result.recommendation


# === Database Tests ===


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
        "sample_rate_hz": 4000.0, "bearing_analysis_available": True,
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
        "sample_rate_hz": 4000.0, "bearing_analysis_available": True,
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
        "sample_rate_hz": 4000.0, "bearing_analysis_available": True,
    }
    await test_db.log_measurement(patrol_id, "M-001", "2026-07-19T00:01:00", features)
    await test_db.log_measurement(patrol_id, "M-001", "2026-07-19T00:02:00", features)

    trend = await test_db.get_station_trend("M-001", 5)
    assert len(trend) == 2
    # Should be oldest first
    assert trend[0]["measured_at"] <= trend[1]["measured_at"]


@pytest.mark.asyncio
async def test_trend_matches_persisted_fault_codes(test_db):
    """The codes a diagnosis writes must be the codes a later trend reads.

    Regression test: diagnoses used to be stored under narrative labels like
    "mechanical_failure_with_heating" while the trend rule compared modality
    labels like "bearing_fault", so escalation could never fire.
    """
    patrol_id = await test_db.start_patrol("Trend Route", "2026-07-19T00:00:00")
    features = {
        "rms": 0.5, "peak": 1.0, "crest_factor": 2.0, "kurtosis": 0.5,
        "dominant_frequency_hz": 30.0, "energy_0_100hz": 0.1,
        "energy_100_500hz": 0.01, "energy_500_1000hz": 0.001,
        "energy_1000_2000hz": 0.0001,
        "sample_rate_hz": 4000.0, "bearing_analysis_available": True,
    }

    for i in range(3):
        vib = generate_sample(fault_type=FaultType.NORMAL)
        aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
        thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.5)
        history = await test_db.get_station_trend("TREND-1", limit=5)
        diagnosis = await ANALYZER.analyze(
            "TREND-1", vibration=vib, acoustic=aco, thermal=thm,
            station_history=history,
        )
        m_id = await test_db.log_measurement(
            patrol_id, "TREND-1", f"2026-07-19T00:0{i}:00", features,
        )
        await test_db.log_diagnosis(
            m_id, "TREND-1", DiagnosisRecord.from_fused(diagnosis),
            f"2026-07-19T00:0{i}:00",
        )

    # By the third patrol the persisted history must have escalated the verdict.
    assert CorrelationTag.TRENDING_WORSE.value in diagnosis.correlation_tags
    assert diagnosis.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL)


@pytest.mark.asyncio
async def test_db_rejects_ad_hoc_diagnosis_objects(test_db):
    """log_diagnosis only accepts a DiagnosisRecord."""
    patrol_id = await test_db.start_patrol("R", "2026-07-19T00:00:00")
    features = {"rms": 0.1, "peak": 0.2, "crest_factor": 2.0, "kurtosis": 0.0,
                "dominant_frequency_hz": 30.0, "sample_rate_hz": 4000.0,
                "bearing_analysis_available": True}
    m_id = await test_db.log_measurement(patrol_id, "M-9", "2026-07-19T00:01:00", features)

    class Fake:
        fault_type = "bearing_fault"

    with pytest.raises(TypeError):
        await test_db.log_diagnosis(m_id, "M-9", Fake(), "2026-07-19T00:01:00")


@pytest.mark.asyncio
async def test_active_faults_uses_canonical_codes(test_db):
    patrol_id = await test_db.start_patrol("R", "2026-07-19T00:00:00")
    features = {"rms": 0.1, "peak": 0.2, "crest_factor": 2.0, "kurtosis": 0.0,
                "dominant_frequency_hz": 30.0, "sample_rate_hz": 4000.0,
                "bearing_analysis_available": True}
    m_id = await test_db.log_measurement(patrol_id, "M-7", "2026-07-19T00:01:00", features)
    await test_db.log_diagnosis(
        m_id, "M-7",
        DiagnosisRecord(
            fault_type=FaultCode.BEARING_FAULT.value, confidence=0.9,
            severity="severe", recommendation="replace", reasoning="x",
            health="critical", priority=1,
        ),
        "2026-07-19T00:01:00",
    )

    faults = await test_db.get_active_faults()
    assert len(faults) == 1
    assert faults[0]["fault_type"] == FaultCode.BEARING_FAULT.value
    assert faults[0]["health"] == "critical"


@pytest.mark.asyncio
async def test_llm_fusion_uses_stubbed_response(stub_llm):
    """When the LLM answers, its verdict is used and its codes are normalised."""
    stub_llm({
        "overall_health": "critical",
        "correlated_faults": ["bearing_outer_race", "hotspot"],
        "recommendation": "Replace bearing now",
        "priority": 1,
        "reasoning": "vibration and thermal agree",
    })
    vib = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.8)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.7)
    result = await ANALYZER.analyze("T-LLM", vibration=vib, thermal=thm)

    assert result.inference_mode == "llm"
    assert result.overall_health == OverallHealth.CRITICAL
    # "bearing_outer_race" is normalised onto the canonical code.
    assert FaultCode.BEARING_FAULT.value in result.correlated_faults
