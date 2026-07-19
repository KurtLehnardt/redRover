"""Drone controller — manages autonomous drone deployed from RVR+.

Supports:
- DJI Tello EDU (primary, via djitellopy)
- Bitcraze Crazyflie 2.1 (secondary, via cflib)
- Simulated (for development)

The drone piggybacks on the RVR+ in a magnetic cradle. When the ground
robot identifies an anomaly requiring aerial inspection (overhead pipes,
elevated equipment, ceiling), it deploys the drone for close-up investigation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


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
    x: float  # meters relative to launch point
    y: float
    z: float  # altitude in meters
    hover_duration: float = 5.0  # seconds to hover and capture
    capture_angles: list[float] = field(default_factory=lambda: [0.0])  # yaw angles to photograph


@dataclass
class AerialCapture:
    """Data captured during aerial inspection."""
    target_id: str
    timestamp: float
    altitude: float
    images: list[str]  # file paths to captured frames
    telemetry: dict  # battery, attitude, position


class DroneController:
    """Manages drone lifecycle: dock, launch, inspect, land."""

    def __init__(
        self,
        drone_type: DroneType = DroneType.SIMULATED,
        tello_ip: str = "192.168.10.1",
        min_battery: int = 20,
    ):
        self.drone_type = drone_type
        self.tello_ip = tello_ip
        self.min_battery = min_battery
        self.state = DroneState.DOCKED
        self._battery = 100
        self._altitude = 0.0
        self._position = (0.0, 0.0, 0.0)
        self._tello = None
        self._crazyflie = None
        self._on_state_change: Callable | None = None

    @property
    def battery(self) -> int:
        if self._tello:
            return self._tello.get_battery()
        return self._battery

    @property
    def is_flight_ready(self) -> bool:
        return self.state == DroneState.DOCKED and self.battery >= self.min_battery

    async def connect(self):
        """Initialize connection to the drone."""
        if self.drone_type == DroneType.SIMULATED:
            logger.info("[DRONE] Simulator connected (battery: %d%%)", self._battery)
            return

        if self.drone_type == DroneType.TELLO:
            await self._connect_tello()
        elif self.drone_type == DroneType.CRAZYFLIE:
            await self._connect_crazyflie()

    async def _connect_tello(self):
        """Connect to DJI Tello EDU via WiFi."""
        try:
            from djitellopy import Tello
            self._tello = Tello()
            self._tello.connect()
            self._battery = self._tello.get_battery()
            logger.info("[DRONE] Tello connected (battery: %d%%, temp: %d°C)",
                        self._battery, self._tello.get_temperature())

            # Enable mission pad detection for precision landing
            self._tello.enable_mission_pads()
            self._tello.set_mission_pad_detection_direction(2)  # downward

        except ImportError:
            logger.error("djitellopy not installed. Run: pip install djitellopy")
            raise
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

            cflib.crtp.init_drivers()
            # Default URI for Crazyflie over Crazyradio
            uri = "radio://0/80/2M/E7E7E7E7E7"
            self._crazyflie = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))
            self._crazyflie.open_link()
            logger.info("[DRONE] Crazyflie connected at %s", uri)

        except ImportError:
            logger.error("cflib not installed. Run: pip install cflib")
            raise

    async def launch(self) -> bool:
        """Take off from RVR+ cradle.

        Returns True if launch successful, False if aborted.
        """
        if not self.is_flight_ready:
            logger.warning("[DRONE] Cannot launch: state=%s, battery=%d%%",
                           self.state, self.battery)
            return False

        self._set_state(DroneState.LAUNCHING)
        logger.info("[DRONE] Launching from cradle...")

        if self.drone_type == DroneType.SIMULATED:
            await asyncio.sleep(2.0)
            self._altitude = 1.0
            self._position = (0.0, 0.0, 1.0)
            self._battery -= 2
            logger.info("[DRONE] Airborne at %.1fm", self._altitude)
            self._set_state(DroneState.FLYING)
            return True

        if self.drone_type == DroneType.TELLO:
            self._tello.takeoff()
            await asyncio.sleep(3.0)
            # Rise to safe inspection altitude
            self._tello.move_up(80)  # 80cm above takeoff
            self._altitude = 1.8
            self._set_state(DroneState.FLYING)
            return True

        return False

    async def fly_to_target(self, target: InspectionTarget):
        """Navigate to an inspection target."""
        self._set_state(DroneState.FLYING)
        logger.info("[DRONE] Flying to target: %s at (%.1f, %.1f, %.1f)m",
                    target.name, target.x, target.y, target.z)

        if self.drone_type == DroneType.SIMULATED:
            # Simulate flight time based on distance
            dx = target.x - self._position[0]
            dy = target.y - self._position[1]
            dz = target.z - self._position[2]
            distance = (dx**2 + dy**2 + dz**2) ** 0.5
            flight_time = distance / 1.0  # ~1 m/s cruise speed
            await asyncio.sleep(min(flight_time, 3.0))
            self._position = (target.x, target.y, target.z)
            self._altitude = target.z
            self._battery -= max(2, int(distance))
            logger.info("[DRONE] Arrived at target: %s (alt: %.1fm)", target.name, self._altitude)
            return

        if self.drone_type == DroneType.TELLO:
            # Convert to Tello coordinate system (cm, relative to current position)
            dx_cm = int(target.x * 100)
            dy_cm = int(target.y * 100)
            dz_cm = int((target.z - self._altitude) * 100)

            # Use go command for 3D movement
            self._tello.go_xyz_speed(dx_cm, dy_cm, dz_cm, speed=30)
            self._altitude = target.z
            self._position = (target.x, target.y, target.z)

    async def inspect(self, target: InspectionTarget) -> AerialCapture:
        """Hover at target and capture inspection data."""
        self._set_state(DroneState.INSPECTING)
        logger.info("[DRONE] Inspecting: %s (hovering %.1fs)...",
                    target.name, target.hover_duration)

        images = []

        if self.drone_type == DroneType.SIMULATED:
            # Simulate capture at each angle
            for angle in target.capture_angles:
                await asyncio.sleep(0.5)
                img_path = f"data/captures/{target.target_id}_{int(angle)}deg_{int(time.time())}.jpg"
                images.append(img_path)
                logger.info("[DRONE]   Captured at yaw=%.0f° → %s", angle, img_path)

            await asyncio.sleep(target.hover_duration)
            self._battery -= 3

        elif self.drone_type == DroneType.TELLO:
            # Start video stream
            self._tello.streamon()
            await asyncio.sleep(1.0)

            for angle in target.capture_angles:
                # Rotate to capture angle
                if angle != 0:
                    self._tello.rotate_clockwise(int(angle))
                    await asyncio.sleep(1.0)

                # Capture frame
                frame = self._tello.get_frame_read().frame
                if frame is not None:
                    import cv2
                    img_path = f"data/captures/{target.target_id}_{int(angle)}deg_{int(time.time())}.jpg"
                    cv2.imwrite(img_path, frame)
                    images.append(img_path)

            self._tello.streamoff()

        return AerialCapture(
            target_id=target.target_id,
            timestamp=time.time(),
            altitude=self._altitude,
            images=images,
            telemetry={
                "battery": self.battery,
                "altitude": self._altitude,
                "position": self._position,
            },
        )

    async def return_to_cradle(self):
        """Fly back to RVR+ position and land on magnetic cradle.

        Uses ArUco marker on RVR+ top plate for precision alignment.
        """
        self._set_state(DroneState.RETURNING)
        logger.info("[DRONE] Returning to cradle...")

        if self.drone_type == DroneType.SIMULATED:
            distance = (self._position[0]**2 + self._position[1]**2) ** 0.5
            await asyncio.sleep(min(distance / 1.0, 3.0))
            self._position = (0.0, 0.0, self._altitude)
            self._battery -= max(2, int(distance))
            logger.info("[DRONE] Above cradle, beginning landing sequence")

        elif self.drone_type == DroneType.TELLO:
            # Navigate back to origin
            x_cm = int(-self._position[0] * 100)
            y_cm = int(-self._position[1] * 100)
            if abs(x_cm) > 20 or abs(y_cm) > 20:
                self._tello.go_xyz_speed(x_cm, y_cm, 0, speed=30)
            self._position = (0.0, 0.0, self._altitude)

        # Precision landing
        await self._precision_land()

    async def _precision_land(self):
        """Execute precision landing onto RVR+ cradle using ArUco/mission pad."""
        self._set_state(DroneState.LANDING)
        logger.info("[DRONE] Precision landing...")

        if self.drone_type == DroneType.SIMULATED:
            # Simulate descent
            while self._altitude > 0.1:
                self._altitude -= 0.3
                await asyncio.sleep(0.3)
            self._altitude = 0.0
            self._position = (0.0, 0.0, 0.0)
            self._battery -= 2
            logger.info("[DRONE] Landed on cradle. Battery: %d%%", self._battery)
            self._set_state(DroneState.DOCKED)
            return

        if self.drone_type == DroneType.TELLO:
            # Try mission pad detection for precision landing
            pad_id = self._tello.get_mission_pad_id()
            if pad_id != -1:
                logger.info("[DRONE] Mission pad detected (ID: %d), precision landing...", pad_id)
                # Use mission pad-relative positioning for final approach
                self._tello.go_xyz_speed_mid(0, 0, 40, 20, pad_id)  # Center over pad at 40cm
                await asyncio.sleep(2.0)
            self._tello.land()
            self._altitude = 0.0
            self._set_state(DroneState.DOCKED)

    async def emergency_land(self):
        """Emergency landing — land immediately at current position."""
        logger.warning("[DRONE] EMERGENCY LANDING")
        self._set_state(DroneState.LANDING)

        if self.drone_type == DroneType.TELLO and self._tello:
            self._tello.land()
        elif self.drone_type == DroneType.SIMULATED:
            self._altitude = 0.0

        self._set_state(DroneState.ERROR)

    async def disconnect(self):
        """Clean disconnect from drone."""
        if self._tello:
            self._tello.end()
        if self._crazyflie:
            self._crazyflie.close_link()
        logger.info("[DRONE] Disconnected")

    def _set_state(self, new_state: DroneState):
        old_state = self.state
        self.state = new_state
        if self._on_state_change:
            self._on_state_change(old_state, new_state)
        logger.debug("[DRONE] State: %s → %s", old_state, new_state)
