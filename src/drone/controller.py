"""Drone controller — manages the autonomous drone deployed from the RVR+.

Supports:
- DJI Tello EDU (primary, via djitellopy)
- Bitcraze Crazyflie 2.1 (secondary, via cflib)
- Simulated (for development)

The drone piggybacks on the RVR+ in a magnetic cradle. When the ground robot
identifies an anomaly requiring aerial inspection (overhead pipes, elevated
equipment, ceiling), it deploys the drone for close-up investigation.

``djitellopy`` and ``cflib`` are synchronous and talk over UDP/radio with
multi-second round trips.  Every such call is dispatched with
``asyncio.to_thread`` so a flight does not freeze the event loop — and with it
the dashboard, telemetry export, and the rover's own BLE notifications.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Where aerial captures are written.  Created on connect so cv2.imwrite cannot
# silently fail against a missing directory.
CAPTURE_DIR = Path("data/captures")

# Battery is read over the air; cache briefly so a per-target safety check does
# not issue a fresh round trip on every access.
BATTERY_CACHE_SECONDS = 2.0


class DroneState(str, Enum):
    DOCKED = "docked"          # On RVR+ cradle
    LAUNCHING = "launching"    # Taking off from cradle
    FLYING = "flying"          # In autonomous flight
    INSPECTING = "inspecting"  # Hovering at target, capturing data
    RETURNING = "returning"    # Flying back to RVR+
    LANDING = "landing"        # Precision landing on cradle
    ERROR = "error"
    LOW_BATTERY = "low_battery"


class DroneType(str, Enum):
    TELLO = "tello"
    CRAZYFLIE = "crazyflie"
    SIMULATED = "simulated"


@dataclass
class InspectionTarget:
    """A point in 3D space the drone should inspect."""
    target_id: str
    name: str
    x: float  # metres relative to launch point
    y: float
    z: float  # altitude in metres
    hover_duration: float = 5.0  # seconds to hover and capture
    capture_angles: list[float] = field(default_factory=lambda: [0.0])  # yaw angles


@dataclass
class AerialCapture:
    """Data captured during aerial inspection.

    ``images`` holds paths to files that actually exist on disk.  Simulated
    flights leave it empty and record only ``angles``: inventing paths for
    files that were never written made a simulated run indistinguishable from
    a real one in the logs and the database.
    """

    target_id: str
    timestamp: float
    altitude: float
    images: list[str]  # file paths to frames written to disk
    telemetry: dict  # battery, attitude, position
    angles: list[float] = field(default_factory=list)  # yaw angles inspected
    simulated: bool = False


class DroneController:
    """Manages drone lifecycle: dock, launch, inspect, land."""

    def __init__(
        self,
        drone_type: DroneType = DroneType.SIMULATED,
        tello_ip: str = "192.168.10.1",
        min_battery: int = 20,
        capture_dir: Path | str = CAPTURE_DIR,
        time_scale: float = 1.0,
        landing_mode: str = "mission_pad",
        marker_id: int = 42,
    ):
        self.drone_type = drone_type
        self.tello_ip = tello_ip
        self.min_battery = min_battery
        # "mission_pad" uses the Tello's own pad detection; "aruco" closes a
        # PID loop on the marker printed on the RVR+ top plate; "plain" just
        # descends where it is.
        self.landing_mode = landing_mode
        self.marker_id = marker_id
        self._landing_system = None
        # Multiplies artificial delays in simulated flight (0.0 = no waiting).
        self.time_scale = max(0.0, time_scale)
        self.capture_dir = Path(capture_dir)
        self.state = DroneState.DOCKED
        self._battery = 100
        self._battery_read_at = 0.0
        self._altitude = 0.0
        self._position = (0.0, 0.0, 0.0)
        self._tello = None
        self._crazyflie = None
        self._on_state_change: Callable | None = None

    async def _sim_sleep(self, seconds: float) -> None:
        """Sleep for a simulated flight delay, scaled by ``time_scale``."""
        await asyncio.sleep(seconds * self.time_scale)

    # -- battery ------------------------------------------------------------

    @property
    def last_known_battery(self) -> int:
        """Most recent battery reading without issuing a new query."""
        return self._battery

    async def get_battery(self, max_age: float = BATTERY_CACHE_SECONDS) -> int:
        """Battery percentage, refreshed at most every ``max_age`` seconds."""
        if self.drone_type is not DroneType.TELLO or self._tello is None:
            return self._battery
        if time.monotonic() - self._battery_read_at < max_age:
            return self._battery
        try:
            self._battery = int(await asyncio.to_thread(self._tello.get_battery))
        except Exception as exc:
            logger.warning("[DRONE] battery query failed: %s", exc)
        self._battery_read_at = time.monotonic()
        return self._battery

    async def is_flight_ready(self) -> bool:
        return self.state is DroneState.DOCKED and await self.get_battery() >= self.min_battery

    # -- connection ---------------------------------------------------------

    async def connect(self):
        """Initialize connection to the drone."""
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        if self.drone_type is DroneType.SIMULATED:
            logger.info("[DRONE] Simulator connected (battery: %d%%)", self._battery)
            return
        if self.drone_type is DroneType.TELLO:
            await self._connect_tello()
        elif self.drone_type is DroneType.CRAZYFLIE:
            await self._connect_crazyflie()

    async def _connect_tello(self):
        """Connect to DJI Tello EDU via WiFi."""
        try:
            from djitellopy import Tello
        except ImportError as exc:
            raise RuntimeError(
                "djitellopy not installed. Run: pip install djitellopy"
            ) from exc

        def _setup():
            tello = Tello()
            tello.connect()
            battery = tello.get_battery()
            temperature = tello.get_temperature()
            # Mission pads give the precision-landing controller a target.
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(2)  # downward
            return tello, battery, temperature

        try:
            self._tello, self._battery, temperature = await asyncio.to_thread(_setup)
            self._battery_read_at = time.monotonic()
            logger.info(
                "[DRONE] Tello connected (battery: %d%%, temp: %s C)",
                self._battery, temperature,
            )
        except Exception as e:
            logger.error("[DRONE] Tello connection failed: %s", e)
            self.state = DroneState.ERROR
            raise

    async def _connect_crazyflie(self):
        """Connect to Bitcraze Crazyflie 2.1."""
        try:
            import cflib.crtp
            from cflib.crazyflie import Crazyflie
            from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        except ImportError as exc:
            raise RuntimeError("cflib not installed. Run: pip install cflib") from exc

        uri = "radio://0/80/2M/E7E7E7E7E7"

        def _setup():
            cflib.crtp.init_drivers()
            link = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))
            link.open_link()
            return link

        self._crazyflie = await asyncio.to_thread(_setup)
        logger.info("[DRONE] Crazyflie connected at %s", uri)

    # -- flight -------------------------------------------------------------

    async def launch(self) -> bool:
        """Take off from the RVR+ cradle. Returns True if launch succeeded."""
        if not await self.is_flight_ready():
            logger.warning(
                "[DRONE] Cannot launch: state=%s, battery=%d%%",
                self.state.value, self._battery,
            )
            return False

        self._set_state(DroneState.LAUNCHING)
        logger.info("[DRONE] Launching from cradle...")

        if self.drone_type is DroneType.SIMULATED:
            await self._sim_sleep(2.0)
            self._altitude = 1.0
            self._position = (0.0, 0.0, 1.0)
            self._battery -= 2
            logger.info("[DRONE] Airborne at %.1fm", self._altitude)
            self._set_state(DroneState.FLYING)
            return True

        if self.drone_type is DroneType.TELLO:
            try:
                await asyncio.to_thread(self._tello.takeoff)
                await asyncio.sleep(3.0)
                await asyncio.to_thread(self._tello.move_up, 80)  # 80cm above takeoff
            except Exception as exc:
                logger.error("[DRONE] launch failed: %s", exc)
                self._set_state(DroneState.ERROR)
                return False
            self._altitude = 1.8
            self._set_state(DroneState.FLYING)
            return True

        return False

    async def fly_to_target(self, target: InspectionTarget):
        """Navigate to an inspection target."""
        self._set_state(DroneState.FLYING)
        logger.info(
            "[DRONE] Flying to target: %s at (%.1f, %.1f, %.1f)m",
            target.name, target.x, target.y, target.z,
        )

        if self.drone_type is DroneType.SIMULATED:
            dx = target.x - self._position[0]
            dy = target.y - self._position[1]
            dz = target.z - self._position[2]
            distance = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            await self._sim_sleep(min(distance / 1.0, 3.0))  # ~1 m/s cruise
            self._position = (target.x, target.y, target.z)
            self._altitude = target.z
            self._battery -= max(2, int(distance))
            logger.info(
                "[DRONE] Arrived at target: %s (alt: %.1fm)", target.name, self._altitude
            )
            return

        if self.drone_type is DroneType.TELLO:
            dx_cm = int((target.x - self._position[0]) * 100)
            dy_cm = int((target.y - self._position[1]) * 100)
            dz_cm = int((target.z - self._altitude) * 100)
            await asyncio.to_thread(
                self._tello.go_xyz_speed, dx_cm, dy_cm, dz_cm, 30,
            )
            self._altitude = target.z
            self._position = (target.x, target.y, target.z)

    async def inspect(self, target: InspectionTarget) -> AerialCapture:
        """Hover at target and capture inspection data."""
        self._set_state(DroneState.INSPECTING)
        logger.info(
            "[DRONE] Inspecting: %s (hovering %.1fs)...", target.name, target.hover_duration
        )

        images: list[str] = []
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        if self.drone_type is DroneType.SIMULATED:
            for _ in target.capture_angles:
                await self._sim_sleep(0.1)
            await self._sim_sleep(target.hover_duration)
            self._battery -= 3

        elif self.drone_type is DroneType.TELLO:
            await asyncio.to_thread(self._tello.streamon)
            await asyncio.sleep(1.0)
            try:
                for angle in target.capture_angles:
                    if angle:
                        await asyncio.to_thread(
                            self._tello.rotate_clockwise, int(angle)
                        )
                        await asyncio.sleep(1.0)
                    path = await asyncio.to_thread(self._capture_frame, target, angle)
                    if path:
                        images.append(path)
            finally:
                await asyncio.to_thread(self._tello.streamoff)

        return AerialCapture(
            target_id=target.target_id,
            timestamp=time.time(),
            altitude=self._altitude,
            images=images,
            telemetry={
                "battery": self._battery,
                "altitude": self._altitude,
                "position": self._position,
            },
            angles=list(target.capture_angles),
            simulated=self.drone_type is DroneType.SIMULATED,
        )

    def _capture_frame(self, target: InspectionTarget, angle: float) -> str | None:
        """Grab one frame and write it to disk. Returns the path, or None."""
        try:
            import cv2
        except ImportError:
            logger.warning("[DRONE] opencv-python not installed; no frames captured")
            return None

        frame = self._tello.get_frame_read().frame
        if frame is None:
            logger.warning("[DRONE] no frame available for %s", target.target_id)
            return None

        path = self.capture_dir / f"{target.target_id}_{int(angle)}deg_{int(time.time())}.jpg"
        if not cv2.imwrite(str(path), frame):
            logger.error("[DRONE] failed to write capture to %s", path)
            return None
        logger.info("[DRONE]   Captured at yaw=%.0f deg -> %s", angle, path)
        return str(path)

    async def return_to_cradle(self):
        """Fly back to the RVR+ position and land on the magnetic cradle."""
        self._set_state(DroneState.RETURNING)
        logger.info("[DRONE] Returning to cradle...")

        if self.drone_type is DroneType.SIMULATED:
            distance = (self._position[0] ** 2 + self._position[1] ** 2) ** 0.5
            await self._sim_sleep(min(distance / 1.0, 3.0))
            self._position = (0.0, 0.0, self._altitude)
            self._battery -= max(2, int(distance))
            logger.info("[DRONE] Above cradle, beginning landing sequence")

        elif self.drone_type is DroneType.TELLO:
            x_cm = int(-self._position[0] * 100)
            y_cm = int(-self._position[1] * 100)
            if abs(x_cm) > 20 or abs(y_cm) > 20:
                await asyncio.to_thread(self._tello.go_xyz_speed, x_cm, y_cm, 0, 30)
            self._position = (0.0, 0.0, self._altitude)

        await self._precision_land()

    async def _precision_land(self):
        """Precision landing onto the RVR+ cradle using a mission pad."""
        self._set_state(DroneState.LANDING)
        logger.info("[DRONE] Precision landing...")

        if self.drone_type is DroneType.SIMULATED:
            while self._altitude > 0.1:
                self._altitude -= 0.3
                await self._sim_sleep(0.3)
            self._altitude = 0.0
            self._position = (0.0, 0.0, 0.0)
            self._battery -= 2
            logger.info("[DRONE] Landed on cradle. Battery: %d%%", self._battery)
            self._set_state(DroneState.DOCKED)
            return

        if self.drone_type is not DroneType.TELLO:
            return

        if self.landing_mode == "aruco":
            if await self._aruco_land():
                return
            logger.warning("[DRONE] marker landing failed; falling back to mission pad")

        if self.landing_mode in ("aruco", "mission_pad"):
            pad_id = await asyncio.to_thread(self._tello.get_mission_pad_id)
            if pad_id != -1:
                logger.info(
                    "[DRONE] Mission pad detected (ID: %d), precision landing...", pad_id
                )
                await asyncio.to_thread(
                    self._tello.go_xyz_speed_mid, 0, 0, 40, 20, pad_id,
                )
                await asyncio.sleep(2.0)

        await self.land()

    async def _aruco_land(self) -> bool:
        """Close a PID loop on the marker printed on the RVR+ top plate."""
        from .precision_landing import PrecisionLandingSystem

        if self._landing_system is None:
            self._landing_system = PrecisionLandingSystem(target_marker_id=self.marker_id)

        streaming = False
        try:
            await asyncio.to_thread(self._tello.streamon)
            streaming = True
            await asyncio.sleep(1.0)
            return await self._landing_system.execute_landing(self, self.read_frame)
        except Exception as exc:
            logger.error("[DRONE] marker landing error: %s", exc)
            return False
        finally:
            if streaming:
                try:
                    await asyncio.to_thread(self._tello.streamoff)
                except Exception as exc:
                    logger.debug("[DRONE] streamoff: %s", exc)

    # -- surface used by the precision-landing controller -------------------

    async def send_rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Send a raw RC command. Each axis is -100..100."""
        if self.drone_type is DroneType.TELLO and self._tello is not None:
            await asyncio.to_thread(self._tello.send_rc_control, lr, fb, ud, yaw)
        else:
            logger.debug("[DRONE] rc lr=%d fb=%d ud=%d yaw=%d", lr, fb, ud, yaw)

    async def land(self) -> None:
        """Land where the drone currently is."""
        if self.drone_type is DroneType.TELLO and self._tello is not None:
            await asyncio.to_thread(self._tello.land)
        self._altitude = 0.0
        self._set_state(DroneState.DOCKED)

    def read_frame(self):
        """Grab the current camera frame, or None. Blocking; call off-loop."""
        if self.drone_type is DroneType.TELLO and self._tello is not None:
            return self._tello.get_frame_read().frame
        return None

    async def emergency_land(self):
        """Emergency landing — land immediately at the current position."""
        logger.warning("[DRONE] EMERGENCY LANDING")
        self._set_state(DroneState.LANDING)

        if self.drone_type is DroneType.TELLO and self._tello:
            try:
                await asyncio.to_thread(self._tello.land)
            except Exception as exc:
                logger.error("[DRONE] emergency land failed: %s", exc)
        elif self.drone_type is DroneType.SIMULATED:
            self._altitude = 0.0

        self._set_state(DroneState.ERROR)

    async def disconnect(self):
        """Clean disconnect from the drone."""
        if self._tello:
            try:
                await asyncio.to_thread(self._tello.end)
            except Exception as exc:
                logger.debug("[DRONE] tello end: %s", exc)
            self._tello = None
        if self._crazyflie:
            try:
                await asyncio.to_thread(self._crazyflie.close_link)
            except Exception as exc:
                logger.debug("[DRONE] crazyflie close: %s", exc)
            self._crazyflie = None
        logger.info("[DRONE] Disconnected")

    def _set_state(self, new_state: DroneState):
        old_state = self.state
        self.state = new_state
        if self._on_state_change:
            self._on_state_change(old_state, new_state)
        logger.debug("[DRONE] State: %s -> %s", old_state.value, new_state.value)
