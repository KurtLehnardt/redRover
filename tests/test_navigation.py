"""Heading convention, odometry honesty, and e-stop behaviour."""

from __future__ import annotations

import math

import pytest

from src.rover.controller import (
    RoverController,
    RoverState,
    Waypoint,
    heading_difference,
    heading_to_vector,
    vector_to_heading,
)

# === Heading convention ===
#
# Regression: drive_to() used math convention (0 deg = +X, CCW) while the
# mapper used RVR+ convention (0 deg = +Y, CW). Mixing them steered the robot
# 90 degrees off and mirrored.


@pytest.mark.parametrize(
    "heading,expected",
    [
        (0.0, (0.0, 1.0)),     # forward is +Y
        (90.0, (1.0, 0.0)),    # right is +X
        (180.0, (0.0, -1.0)),
        (270.0, (-1.0, 0.0)),
    ],
)
def test_heading_to_vector(heading, expected):
    dx, dy = heading_to_vector(heading)
    assert math.isclose(dx, expected[0], abs_tol=1e-9)
    assert math.isclose(dy, expected[1], abs_tol=1e-9)


@pytest.mark.parametrize(
    "dx,dy,expected",
    [(0, 1, 0.0), (1, 0, 90.0), (0, -1, 180.0), (-1, 0, 270.0)],
)
def test_vector_to_heading(dx, dy, expected):
    assert math.isclose(vector_to_heading(dx, dy), expected, abs_tol=1e-9)


def test_heading_roundtrip():
    for heading in range(0, 360, 7):
        dx, dy = heading_to_vector(float(heading))
        assert math.isclose(vector_to_heading(dx, dy), heading, abs_tol=1e-6)


def test_heading_difference_takes_short_way():
    assert heading_difference(10.0, 350.0) == pytest.approx(20.0)
    assert heading_difference(350.0, 10.0) == pytest.approx(-20.0)
    assert heading_difference(0.0, 0.0) == pytest.approx(0.0)


# === Travel time is derived from calibrated speed ===


def test_travel_time_scales_with_speed():
    slow = RoverController(simulate=True, speed=0.25, max_speed_mps=2.0)
    fast = RoverController(simulate=True, speed=1.0, max_speed_mps=2.0)
    # 2 m at 0.5 m/s = 4 s; at 2.0 m/s = 1 s.
    assert slow.travel_time_for(2.0) == pytest.approx(4.0)
    assert fast.travel_time_for(2.0) == pytest.approx(1.0)


def test_travel_time_is_capped():
    rover = RoverController(simulate=True, speed=0.1, max_speed_mps=0.1,
                            max_drive_seconds=5.0)
    assert rover.travel_time_for(1000.0) == 5.0


# === Odometry honesty ===


@pytest.mark.asyncio
async def test_simulated_position_is_flagged_estimated():
    rover = RoverController(simulate=True, speed=1.0, time_scale=0.0)
    await rover.connect()
    await rover.drive_to(Waypoint(station_id="A", x=1.0, y=1.0))
    assert rover.position == (1.0, 1.0)
    # Nothing measured this position, and the controller says so.
    assert rover.position_is_estimated is True


def test_locator_stream_clears_estimated_flag():
    rover = RoverController(simulate=True)
    assert rover.position_is_estimated is True
    rover.inject_sensor_data(locator=(0.4, -0.2))
    assert rover.position == (0.4, -0.2)


# === Emergency stop ===


@pytest.mark.asyncio
async def test_estop_blocks_drive():
    rover = RoverController(simulate=True, time_scale=0.0)
    await rover.connect()
    await rover.emergency_stop()
    assert rover.estopped
    assert rover.state is RoverState.ESTOP

    with pytest.raises(RuntimeError):
        await rover.drive_to(Waypoint(station_id="A", x=1.0, y=0.0))

    rover.clear_estop()
    assert not rover.estopped
    await rover.drive_to(Waypoint(station_id="A", x=1.0, y=0.0))


@pytest.mark.asyncio
async def test_sensor_callbacks_are_retained():
    """Callback tasks must be held strongly or asyncio may collect them."""
    rover = RoverController(simulate=True)
    seen = []

    async def cb(data):
        seen.append(data)

    rover.add_sensor_callback(cb)
    rover.inject_sensor_data(locator=(1.0, 2.0))
    # Let the scheduled callback task run.
    import asyncio
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert seen and seen[0]["locator"] == (1.0, 2.0)

    rover.remove_sensor_callback(cb)
    rover.inject_sensor_data(locator=(3.0, 4.0))
    await asyncio.sleep(0)
    assert len(seen) == 1
