"""RVR+ motor control and navigation.

Supports two modes:
- Real: connects to Sphero RVR+ via BLE or UART
- Simulated: logs movements for development without hardware
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RoverState(str, Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    DWELLING = "dwelling"  # Parked at station, measuring
    RETURNING = "returning"
    ERROR = "error"


@dataclass
class Waypoint:
    """A machine station location."""
    station_id: str
    x: float  # meters from origin
    y: float
    heading: float = 0.0  # degrees
    name: str = ""


@dataclass
class PatrolRoute:
    """Ordered list of waypoints to visit."""
    name: str
    waypoints: list[Waypoint] = field(default_factory=list)


class RoverController:
    """Controls RVR+ movement and navigation."""

    def __init__(self, connection: str = "ble", speed: float = 0.3, simulate: bool = False):
        self.connection = connection
        self.speed = speed
        self.simulate = simulate
        self.state = RoverState.IDLE
        self._position = (0.0, 0.0)
        self._heading = 0.0
        self._rvr = None

    async def connect(self):
        """Connect to the RVR+."""
        if self.simulate:
            logger.info("RVR+ simulator connected")
            return

        # Real hardware connection
        try:
            from sphero_sdk import SpheroRvrAsync, SerialAsyncDal
            if self.connection == "uart":
                self._rvr = SpheroRvrAsync(dal=SerialAsyncDal(port="/dev/ttyTHS1"))
            else:
                # BLE connection for laptop development
                self._rvr = SpheroRvrAsync(dal=SerialAsyncDal(port="/dev/tty.sphero"))
            await self._rvr.wake()
            await asyncio.sleep(2)
            logger.info("RVR+ connected via %s", self.connection)
        except ImportError:
            logger.warning("sphero_sdk not available, falling back to simulation")
            self.simulate = True
        except Exception as e:
            logger.error("Failed to connect to RVR+: %s", e)
            self.state = RoverState.ERROR
            raise

    async def disconnect(self):
        """Disconnect from the RVR+."""
        if self._rvr:
            await self._rvr.close()
        logger.info("RVR+ disconnected")

    async def drive_to(self, waypoint: Waypoint):
        """Navigate to a waypoint."""
        self.state = RoverState.NAVIGATING
        logger.info("Navigating to station %s (%s) at (%.1f, %.1f)",
                    waypoint.station_id, waypoint.name, waypoint.x, waypoint.y)

        if self.simulate:
            # Simulate travel time based on distance
            dx = waypoint.x - self._position[0]
            dy = waypoint.y - self._position[1]
            distance = (dx**2 + dy**2) ** 0.5
            travel_time = distance / (self.speed * 2.0)  # rough estimate
            await asyncio.sleep(min(travel_time, 2.0))  # cap sim time
            self._position = (waypoint.x, waypoint.y)
            self._heading = waypoint.heading
            logger.info("Arrived at %s (simulated)", waypoint.station_id)
        else:
            # Real navigation using RVR+ drive commands
            # TODO: Implement dead reckoning or SLAM-based navigation
            dx = waypoint.x - self._position[0]
            dy = waypoint.y - self._position[1]
            import math
            target_heading = math.degrees(math.atan2(dy, dx))
            distance_cm = ((dx**2 + dy**2) ** 0.5) * 100

            await self._rvr.drive_with_heading(
                speed=int(self.speed * 255),
                heading=int(target_heading) % 360,
                flags=0,
            )
            # Wait proportional to distance
            await asyncio.sleep(distance_cm / 50.0)
            await self._rvr.drive_with_heading(speed=0, heading=int(target_heading) % 360, flags=0)
            self._position = (waypoint.x, waypoint.y)

        self.state = RoverState.DWELLING

    async def return_home(self):
        """Return to origin position."""
        self.state = RoverState.RETURNING
        home = Waypoint(station_id="HOME", x=0.0, y=0.0, heading=0.0, name="Home Base")
        await self.drive_to(home)
        self.state = RoverState.IDLE

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def heading(self) -> float:
        return self._heading
