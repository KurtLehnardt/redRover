"""Precision landing, exercised against real rendered ArUco markers.

This module used to be unreachable: nothing outside it called
``PrecisionLandingSystem`` or ``execute_landing``, and the tests covered only
the PID arithmetic — so 263 lines looked tested while being dead. It is now
wired into ``DroneController._precision_land`` behind ``landing_mode="aruco"``,
and these tests drive the whole loop with synthetic camera frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.drone.controller import DroneController, DroneType
from src.drone.precision_landing import (
    MARKER_ID,
    MarkerDetection,
    PIDController,
    PrecisionLandingSystem,
)

cv2 = pytest.importorskip("cv2", reason="opencv-python required for ArUco tests")
aruco = pytest.importorskip("cv2.aruco", reason="opencv aruco module required")


def render_marker_frame(
    marker_id: int = MARKER_ID,
    size_px: int = 160,
    centre: tuple[int, int] = (320, 240),
    shape: tuple[int, int] = (480, 640),
) -> np.ndarray:
    """A camera frame containing one ArUco marker at a known position."""
    marker = aruco.generateImageMarker(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50), marker_id, size_px
    )
    frame = np.full(shape, 255, dtype=np.uint8)
    cx, cy = centre
    half = size_px // 2
    y0, x0 = cy - half, cx - half
    frame[y0:y0 + size_px, x0:x0 + size_px] = marker
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


# === detection ===


def test_detects_a_centred_marker():
    detection = PrecisionLandingSystem().detect_marker(render_marker_frame())
    assert detection is not None
    assert detection.marker_id == MARKER_ID
    assert detection.center_x == pytest.approx(0.0, abs=0.02)
    assert detection.center_y == pytest.approx(0.0, abs=0.02)


def test_offset_marker_reports_the_direction_of_the_offset():
    right = PrecisionLandingSystem().detect_marker(
        render_marker_frame(centre=(480, 240))
    )
    assert right is not None and right.center_x > 0.3

    down = PrecisionLandingSystem().detect_marker(
        render_marker_frame(centre=(320, 380))
    )
    assert down is not None and down.center_y > 0.3


def test_a_bigger_marker_reads_as_closer():
    system = PrecisionLandingSystem()
    near = system.detect_marker(render_marker_frame(size_px=300))
    far = system.detect_marker(render_marker_frame(size_px=80))
    assert near is not None and far is not None
    assert near.distance_cm < far.distance_cm


def test_ignores_a_marker_that_is_not_the_target():
    system = PrecisionLandingSystem(target_marker_id=MARKER_ID)
    assert system.detect_marker(render_marker_frame(marker_id=7)) is None


def test_empty_scene_detects_nothing():
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert PrecisionLandingSystem().detect_marker(blank) is None


def test_detector_is_built_once():
    """Regression: the dictionary, parameters, and detector were rebuilt on
    every frame, inside a 20 Hz control loop."""
    system = PrecisionLandingSystem()
    first = system._detector()
    system.detect_marker(render_marker_frame())
    assert system._detector() is first


# === the control loop ===


class RecordingDrone:
    """Stands in for DroneController's public landing surface."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.rc_commands: list[tuple[int, int, int, int]] = []
        self.landed = False

    async def send_rc(self, lr, fb, ud, yaw):
        self.rc_commands.append((lr, fb, ud, yaw))

    async def land(self):
        self.landed = True

    def read_frame(self):
        return self._frames.pop(0) if self._frames else None


@pytest.mark.asyncio
async def test_landing_converges_on_a_close_centred_marker():
    # A large, centred marker is inside the completion window immediately.
    frames = [render_marker_frame(size_px=340) for _ in range(10)]
    drone = RecordingDrone(frames)
    system = PrecisionLandingSystem()

    assert await system.execute_landing(drone, drone.read_frame) is True
    assert drone.landed
    # It stops commanding motion before it lands.
    assert drone.rc_commands[-1] == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_landing_aborts_when_the_marker_is_never_seen():
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    drone = RecordingDrone([blank] * 60)
    system = PrecisionLandingSystem()

    assert await system.execute_landing(drone, drone.read_frame) is False
    assert not drone.landed
    # Aborting must neutralise the sticks rather than leave the last command
    # running on a drone that can no longer see where it is.
    assert drone.rc_commands[-1] == (0, 0, 0, 0)


@pytest.mark.asyncio
async def test_landing_aborts_when_the_camera_returns_nothing():
    drone = RecordingDrone([])  # read_frame() yields None forever
    assert await PrecisionLandingSystem().execute_landing(drone, drone.read_frame) is False
    assert not drone.landed


@pytest.mark.asyncio
async def test_landing_steers_toward_an_offset_marker():
    offset = [render_marker_frame(size_px=120, centre=(500, 240)) for _ in range(5)]
    drone = RecordingDrone(offset)
    system = PrecisionLandingSystem()
    await system.execute_landing(drone, drone.read_frame)

    steering = [cmd for cmd in drone.rc_commands if cmd != (0, 0, 0, 0)]
    assert steering, "should have issued corrections"
    assert steering[0][0] > 0, "marker is to the right; should command right"


# === wiring ===


@pytest.mark.asyncio
async def test_controller_exposes_the_surface_the_landing_loop_needs(tmp_path):
    drone = DroneController(drone_type=DroneType.SIMULATED, capture_dir=tmp_path,
                            time_scale=0.0)
    await drone.connect()
    # These are what execute_landing calls; they must exist and be safe to
    # call without a Tello attached.
    await drone.send_rc(0, 0, 0, 0)
    assert drone.read_frame() is None
    await drone.land()


def test_landing_mode_is_configurable():
    default = DroneController(drone_type=DroneType.SIMULATED)
    assert default.landing_mode == "mission_pad"

    marker = DroneController(drone_type=DroneType.SIMULATED,
                             landing_mode="aruco", marker_id=17)
    assert marker.landing_mode == "aruco"
    assert marker.marker_id == 17


# === PID ===


def test_pid_anti_windup_clamps_the_integral():
    pid = PIDController(kp=0.0, ki=10.0, kd=0.0, output_limit=25.0)
    for _ in range(200):
        pid._last_time -= 0.05
        output = pid.update(10.0)
        assert -25.0 <= output <= 25.0


def test_pid_reset_clears_history():
    pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
    pid.update(10.0)
    pid.reset()
    assert pid._integral == 0.0
    assert pid._last_error == 0.0


def test_landing_completion_needs_close_and_centred():
    system = PrecisionLandingSystem()
    # Close but badly off-centre is not a landing.
    system.compute_landing_commands(
        MarkerDetection(MARKER_ID, 0.5, 0.0, 20.0, 0.0, 0.0)
    )
    assert not system.landing_complete

    system.compute_landing_commands(
        MarkerDetection(MARKER_ID, 0.01, 0.01, 20.0, 0.0, 0.0)
    )
    assert system.landing_complete
