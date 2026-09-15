"""The rover backend contract.

Everything above this line — patrols, mapping, the dashboard — talks to a
rover through this surface. Below it sits whichever robot you actually own:
a Sphero RVR+ over BLE, an Arduino-class controller running the firmware in
``firmware/``, or the simulator.

Adding a platform means implementing this protocol, not editing the patrol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RoverCapabilities:
    """What a particular robot can actually do.

    Callers check these rather than assuming: a differential base cannot
    strafe, and a rover with no locator cannot report a measured position.
    """

    name: str = "unknown"
    holonomic: bool = False
    has_odometry: bool = False
    has_leds: bool = False
    has_battery: bool = False
    # Highest vibration sampling rate the platform can stream, in Hz. The
    # difference between 50 and 4000 decides whether bearing analysis is
    # possible at all.
    max_sensor_rate_hz: float = 0.0
    max_speed_mps: float = 0.0


@runtime_checkable
class RoverBackend(Protocol):
    """Structural interface shared by every rover implementation."""

    simulate: bool
    speed: float

    # -- lifecycle --
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    # -- motion --
    async def drive_to(self, waypoint) -> None: ...
    async def drive_with_heading(self, speed: int, heading: int) -> None: ...
    async def stop(self) -> None: ...
    async def return_home(self) -> None: ...

    # -- safety --
    async def emergency_stop(self) -> None: ...
    def clear_estop(self) -> None: ...

    # -- state --
    @property
    def position(self) -> tuple[float, float]: ...
    @property
    def heading(self) -> float: ...
    @property
    def position_is_estimated(self) -> bool: ...
    @property
    def streaming(self) -> bool: ...

    # -- sensors --
    async def start_sensor_streaming(self, period_ms: int = 100) -> None: ...
    async def stop_sensor_streaming(self) -> None: ...
    def add_sensor_callback(self, callback) -> None: ...
    def remove_sensor_callback(self, callback) -> None: ...
    @property
    def sensor_data(self) -> dict: ...

    # Test and simulation seam. RoomExplorer's simulated loop drives the
    # backend through this, so it is part of the contract rather than an
    # implementation detail of one backend.
    def inject_sensor_data(self, **values) -> None: ...

    # -- optional hardware --
    async def set_leds(self, r: int, g: int, b: int) -> None: ...
    async def get_battery(self) -> int | None: ...

    @property
    def capabilities(self) -> RoverCapabilities: ...
