"""Pick a rover backend from configuration."""

from __future__ import annotations

import logging

from ..controller import RoverController
from .base import RoverBackend

logger = logging.getLogger(__name__)


def create_rover(config, simulate: bool = False) -> RoverBackend:
    """Build the rover described by ``[rover].connection``.

    ``ble`` / ``uart`` -> Sphero RVR+ (:class:`~src.rover.controller.RoverController`)
    ``serial``         -> any board running the firmware in ``firmware/``
    """
    connection = config.rover.connection.lower()

    if connection == "serial":
        from .firmware import FirmwareRover

        return FirmwareRover(
            port=config.rover.serial_port,
            baud=config.rover.serial_baud,
            speed=config.rover.speed,
            simulate=simulate,
            max_speed_mps=config.rover.max_speed_mps,
            max_drive_seconds=config.rover.max_drive_seconds,
            time_scale=config.simulation.time_scale,
        )

    if connection in ("ble", "uart"):
        return RoverController(
            connection=connection,
            speed=config.rover.speed,
            simulate=simulate,
            max_speed_mps=config.rover.max_speed_mps,
            max_drive_seconds=config.rover.max_drive_seconds,
            time_scale=config.simulation.time_scale,
        )

    raise ValueError(
        f"unknown [rover].connection={connection!r}; expected 'ble', 'uart', or 'serial'"
    )
