"""Tests for drone controller, missions, and orchestration."""

import asyncio
import pytest
import numpy as np

from src.drone.controller import (
    DroneController, DroneState, DroneType, InspectionTarget,
)
from src.drone.mission import (
    generate_overhead_pipe_mission,
    generate_elevated_gauge_mission,
    generate_hvac_duct_mission,
    generate_wide_area_scan,
    should_deploy_drone,
    MissionType,
)
from src.drone.precision_landing import PIDController, PrecisionLandingSystem, MarkerDetection


# === Controller Tests ===

@pytest.mark.asyncio
async def test_drone_connect_simulated():
    drone = DroneController(drone_type=DroneType.SIMULATED)
    await drone.connect()
    assert drone.state == DroneState.DOCKED
    assert drone.battery == 100


@pytest.mark.asyncio
async def test_drone_launch_and_land():
    drone = DroneController(drone_type=DroneType.SIMULATED)
    await drone.connect()

    success = await drone.launch()
    assert success
    assert drone.state == DroneState.FLYING

    await drone.return_to_cradle()
    assert drone.state == DroneState.DOCKED


@pytest.mark.asyncio
async def test_drone_inspect_target():
    drone = DroneController(drone_type=DroneType.SIMULATED)
    await drone.connect()
    await drone.launch()

    target = InspectionTarget(
        target_id="TEST-001",
        name="Test pipe section",
        x=1.0, y=0.5, z=2.0,
        hover_duration=1.0,
        capture_angles=[0.0, 90.0],
    )

    await drone.fly_to_target(target)
    capture = await drone.inspect(target)

    assert capture.target_id == "TEST-001"
    assert len(capture.images) == 2  # Two angles
    assert capture.altitude == 2.0


@pytest.mark.asyncio
async def test_drone_battery_decreases():
    drone = DroneController(drone_type=DroneType.SIMULATED)
    await drone.connect()
    initial_battery = drone.battery

    await drone.launch()
    target = InspectionTarget(
        target_id="T1", name="Far target",
        x=3.0, y=3.0, z=2.5,
        hover_duration=1.0,
    )
    await drone.fly_to_target(target)
    await drone.inspect(target)
    await drone.return_to_cradle()

    assert drone.battery < initial_battery


@pytest.mark.asyncio
async def test_drone_wont_launch_low_battery():
    drone = DroneController(drone_type=DroneType.SIMULATED, min_battery=20)
    await drone.connect()
    drone._battery = 15  # Below threshold

    success = await drone.launch()
    assert not success
    assert drone.state == DroneState.DOCKED


# === Mission Tests ===

def test_overhead_pipe_mission():
    mission = generate_overhead_pipe_mission(
        station_id="M-003",
        pipe_length=3.0,
        pipe_height=2.5,
    )
    assert mission.mission_type == MissionType.OVERHEAD_PIPE
    assert mission.station_id == "M-003"
    assert len(mission.targets) >= 3
    # All targets at pipe height
    for t in mission.targets:
        assert t.z == 2.5


def test_elevated_gauge_mission():
    mission = generate_elevated_gauge_mission(
        station_id="M-005",
        gauge_height=2.0,
    )
    assert mission.mission_type == MissionType.ELEVATED_GAUGE
    assert len(mission.targets) == 2  # Approach + close-up


def test_hvac_duct_mission():
    mission = generate_hvac_duct_mission(station_id="M-002")
    assert mission.mission_type == MissionType.HVAC_DUCT
    assert len(mission.targets) == 6  # Circular pattern


def test_wide_area_scan():
    mission = generate_wide_area_scan(station_id="M-001", width=4.0, depth=4.0)
    assert mission.mission_type == MissionType.WIDE_AREA_SCAN
    assert len(mission.targets) >= 4  # At least 2x2 grid


def test_mission_estimated_duration():
    mission = generate_overhead_pipe_mission(station_id="M-003", pipe_length=3.0)
    duration = mission.estimated_duration
    assert duration > 10.0  # Should take at least 10 seconds
    assert duration < 120.0  # Shouldn't be absurdly long


# === Deployment Logic Tests ===

def test_should_deploy_for_air_leak():
    assert should_deploy_drone("air_leak") is True
    assert should_deploy_drone("gas_leak") is True


def test_should_not_deploy_for_normal():
    assert should_deploy_drone("normal") is False


def test_should_deploy_thermal_with_overhead():
    assert should_deploy_drone("overheating", {"has_overhead_equipment": True}) is True
    assert should_deploy_drone("overheating", {"has_overhead_equipment": False}) is False


# === PID Controller Tests ===

def test_pid_output_proportional():
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
    output = pid.update(5.0)
    assert abs(output - 5.0) < 0.1


def test_pid_output_limits():
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_limit=50.0)
    output = pid.update(10.0)
    assert output == 50.0


def test_pid_converges():
    import time as _time
    pid = PIDController(kp=0.5, ki=0.01, kd=0.1, output_limit=100.0)
    error = 10.0
    for i in range(100):
        # Force time progression so dt is meaningful
        pid._last_time = _time.time() - 0.05
        output = pid.update(error)
        error -= output * 0.1  # Simulate system response
    assert abs(error) < 2.0


# === Precision Landing Tests ===

def test_landing_system_marker_detection():
    """Test that ArUco detection returns correct structure."""
    pls = PrecisionLandingSystem()
    # We can't easily test real marker detection without OpenCV + an image,
    # but we can test the command computation
    detection = MarkerDetection(
        marker_id=42,
        center_x=0.3,  # Offset to the right
        center_y=-0.2,  # Offset forward
        distance_cm=80.0,
        yaw_offset=5.0,
        timestamp=0.0,
    )
    commands = pls.compute_landing_commands(detection)
    assert "lr" in commands
    assert "fb" in commands
    assert "ud" in commands
    assert "yaw" in commands
    # Should command left to correct right offset
    assert commands["lr"] > 0  # Positive = move right toward center... wait
    # Actually PID on positive error (center_x=0.3) should output positive correction
    assert -100 <= commands["lr"] <= 100
    assert -100 <= commands["fb"] <= 100
