"""End-to-end, regression, and API tests."""

import pytest

from src.sensors.vibration import FaultType
from src.sensors.acoustic import AcousticFaultType
from src.sensors.thermal import ThermalFaultType
from src.sensors.simulator import (
    generate_sample,
    generate_acoustic_sample,
    generate_thermal_frame,
)
from src.ai.fusion import FusionAnalyzer, OverallHealth


ANALYZER = FusionAnalyzer(model="gemma3", ollama_host="http://localhost:99999")


# === End-to-End Tests ===

@pytest.mark.asyncio
async def test_full_patrol_completes():
    from src.main import run_patrol
    results = await run_patrol(simulate=True, skip_ai=True)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_full_patrol_with_fusion():
    from src.main import run_patrol
    results = await run_patrol(simulate=True, skip_ai=False)
    assert len(results) == 4


@pytest.mark.asyncio
async def test_full_patrol_detects_faults():
    from src.main import run_patrol
    results = await run_patrol(simulate=True, skip_ai=False)
    faults = [r for r in results if r.overall_health != OverallHealth.HEALTHY]
    assert len(faults) >= 1


@pytest.mark.asyncio
async def test_patrol_creates_db_records():
    from src.main import run_patrol
    from src.database import Database
    from src.config import load_config
    config = load_config()
    db = Database(config.database.path)
    await db.init()
    await run_patrol(simulate=True, skip_ai=True)
    patrols = await db.get_recent_patrols(1)
    assert len(patrols) >= 1


# === Regression Tests (known fault signatures) ===

@pytest.mark.asyncio
async def test_regression_bearing_outer_detected():
    vib = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.9)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("REG-001", vibration=vib, acoustic=aco, thermal=thm)
    fault_types = [mr.fault_type for mr in result.modality_results if mr.fault_detected]
    assert any("bearing" in ft or "looseness" in ft for ft in fault_types) or \
           result.overall_health != OverallHealth.HEALTHY


@pytest.mark.asyncio
async def test_regression_air_leak_detected():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.7)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("REG-002", vibration=vib, acoustic=aco, thermal=thm)
    fault_types = [mr.fault_type for mr in result.modality_results if mr.fault_detected]
    assert "air_leak" in fault_types


@pytest.mark.asyncio
async def test_regression_normal_not_flagged():
    vib = generate_sample(fault_type=FaultType.NORMAL)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL)
    result = await ANALYZER.analyze("REG-003", vibration=vib, acoustic=aco, thermal=thm)
    assert result.overall_health == OverallHealth.HEALTHY


@pytest.mark.asyncio
async def test_regression_misalignment_with_thermal():
    vib = generate_sample(fault_type=FaultType.MISALIGNMENT, severity=0.7)
    aco = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    thm = generate_thermal_frame(fault_type=ThermalFaultType.HOTSPOT, severity=0.6)
    result = await ANALYZER.analyze("REG-004", vibration=vib, acoustic=aco, thermal=thm)
    assert result.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL)
    assert len(result.correlated_faults) >= 1


# === Dashboard API Tests ===

@pytest.mark.asyncio
async def test_health_endpoint():
    from httpx import AsyncClient, ASGITransport
    from src.dashboard.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_system_status_endpoint():
    from httpx import AsyncClient, ASGITransport
    from src.dashboard.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/system-status")
    assert response.status_code == 200
    data = response.json()
    assert "ai_status" in data


@pytest.mark.asyncio
async def test_faults_endpoint():
    from httpx import AsyncClient, ASGITransport
    from src.dashboard.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/faults")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_index_page():
    from httpx import AsyncClient, ASGITransport
    from src.dashboard.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "redRover" in response.text


@pytest.mark.asyncio
async def test_station_detail_page():
    from httpx import AsyncClient, ASGITransport
    from src.dashboard.app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/station/M-001")
    assert response.status_code == 200
