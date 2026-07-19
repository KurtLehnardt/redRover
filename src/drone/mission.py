"""Drone mission planning — defines aerial inspection patterns.

When the ground robot identifies an anomaly or reaches a station requiring
aerial inspection, it generates a drone mission with specific inspection targets.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from .controller import InspectionTarget

logger = logging.getLogger(__name__)


class MissionType(str, Enum):
    """Types of aerial inspection missions."""
    OVERHEAD_PIPE = "overhead_pipe"       # Inspect overhead pipes/conduits
    ELEVATED_GAUGE = "elevated_gauge"     # Read gauges mounted high
    HVAC_DUCT = "hvac_duct"             # Check HVAC ductwork
    CEILING_CHECK = "ceiling_check"      # Roof leaks, structural
    CABLE_TRAY = "cable_tray"           # Overhead cable tray inspection
    WIDE_AREA_SCAN = "wide_area_scan"   # General overhead survey
    STACK_TOP = "stack_top"             # Top of storage racks/shelves


@dataclass
class DroneMission:
    """A complete drone mission with ordered inspection targets."""
    mission_id: str
    mission_type: MissionType
    station_id: str  # Which ground station triggered this
    targets: list[InspectionTarget]
    max_altitude: float = 3.0  # meters (safety limit)
    timeout: float = 120.0  # seconds max mission duration
    reason: str = ""  # Why was this mission triggered?

    @property
    def total_targets(self) -> int:
        return len(self.targets)

    @property
    def estimated_duration(self) -> float:
        """Estimate mission duration in seconds."""
        flight_time = 0.0
        prev_pos = (0.0, 0.0, 0.0)
        for t in self.targets:
            dx = t.x - prev_pos[0]
            dy = t.y - prev_pos[1]
            dz = t.z - prev_pos[2]
            distance = (dx**2 + dy**2 + dz**2) ** 0.5
            flight_time += distance / 1.0  # ~1 m/s cruise
            flight_time += t.hover_duration
            prev_pos = (t.x, t.y, t.z)
        # Add return trip
        flight_time += (prev_pos[0]**2 + prev_pos[1]**2 + prev_pos[2]**2) ** 0.5
        return flight_time + 10.0  # +10s for takeoff/landing


def generate_overhead_pipe_mission(
    station_id: str,
    pipe_direction: str = "x",  # "x" or "y" — pipe runs along this axis
    pipe_length: float = 3.0,   # meters
    pipe_height: float = 2.5,   # meters above ground
    reason: str = "Acoustic leak detected below overhead pipe",
) -> DroneMission:
    """Generate a mission to inspect overhead pipes/conduits.

    The drone flies along the pipe at its height, capturing images at intervals.
    """
    targets = []
    n_points = max(3, int(pipe_length / 0.75))  # inspect every 75cm

    for i in range(n_points):
        t = i / (n_points - 1)  # 0 to 1
        offset = -pipe_length / 2 + t * pipe_length

        if pipe_direction == "x":
            x, y = offset, 0.0
        else:
            x, y = 0.0, offset

        targets.append(InspectionTarget(
            target_id=f"{station_id}_pipe_{i:02d}",
            name=f"Pipe section {i+1}/{n_points}",
            x=x, y=y, z=pipe_height,
            hover_duration=3.0,
            capture_angles=[0.0, 90.0],  # Front and side views
        ))

    return DroneMission(
        mission_id=f"pipe_{station_id}_{int(pipe_length)}m",
        mission_type=MissionType.OVERHEAD_PIPE,
        station_id=station_id,
        targets=targets,
        max_altitude=pipe_height + 0.5,
        reason=reason,
    )


def generate_elevated_gauge_mission(
    station_id: str,
    gauge_x: float = 0.0,
    gauge_y: float = 0.5,
    gauge_height: float = 2.0,
    reason: str = "Elevated gauge requires aerial reading",
) -> DroneMission:
    """Generate a mission to read an elevated gauge.

    Drone flies to gauge height, faces it, and captures close-up.
    """
    targets = [
        # Position in front of gauge
        InspectionTarget(
            target_id=f"{station_id}_gauge_approach",
            name="Gauge approach",
            x=gauge_x, y=gauge_y - 0.8, z=gauge_height,
            hover_duration=2.0,
            capture_angles=[0.0],
        ),
        # Close-up
        InspectionTarget(
            target_id=f"{station_id}_gauge_closeup",
            name="Gauge close-up reading",
            x=gauge_x, y=gauge_y - 0.4, z=gauge_height,
            hover_duration=5.0,
            capture_angles=[0.0],  # Head-on for best OCR
        ),
    ]

    return DroneMission(
        mission_id=f"gauge_{station_id}",
        mission_type=MissionType.ELEVATED_GAUGE,
        station_id=station_id,
        targets=targets,
        max_altitude=gauge_height + 0.5,
        reason=reason,
    )


def generate_hvac_duct_mission(
    station_id: str,
    duct_height: float = 2.8,
    scan_radius: float = 2.0,
    reason: str = "Thermal anomaly detected — checking overhead HVAC",
) -> DroneMission:
    """Generate a mission to inspect HVAC ductwork above a station."""
    import math

    targets = []
    # Fly a circular pattern below the duct
    n_points = 6
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        x = scan_radius * math.cos(angle)
        y = scan_radius * math.sin(angle)

        targets.append(InspectionTarget(
            target_id=f"{station_id}_hvac_{i:02d}",
            name=f"HVAC section {i+1}/{n_points}",
            x=x, y=y, z=duct_height - 0.3,
            hover_duration=3.0,
            capture_angles=[0.0],  # Looking up at duct
        ))

    return DroneMission(
        mission_id=f"hvac_{station_id}",
        mission_type=MissionType.HVAC_DUCT,
        station_id=station_id,
        targets=targets,
        max_altitude=duct_height,
        reason=reason,
    )


def generate_wide_area_scan(
    station_id: str,
    width: float = 4.0,
    depth: float = 4.0,
    altitude: float = 2.5,
    reason: str = "General overhead survey of area",
) -> DroneMission:
    """Generate a lawnmower scan pattern for wide-area overhead inspection."""
    targets = []
    rows = max(2, int(depth / 1.5))
    cols = max(2, int(width / 1.5))

    for row in range(rows):
        y = -depth / 2 + row * (depth / (rows - 1))
        # Alternate direction for efficient lawnmower pattern
        col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)

        for col in col_range:
            x = -width / 2 + col * (width / (cols - 1))
            targets.append(InspectionTarget(
                target_id=f"{station_id}_scan_{row:02d}_{col:02d}",
                name=f"Grid ({row},{col})",
                x=x, y=y, z=altitude,
                hover_duration=2.0,
                capture_angles=[0.0],
            ))

    return DroneMission(
        mission_id=f"scan_{station_id}",
        mission_type=MissionType.WIDE_AREA_SCAN,
        station_id=station_id,
        targets=targets,
        max_altitude=altitude + 0.5,
        reason=reason,
    )


# Mapping from detected faults to appropriate drone missions
FAULT_TO_MISSION = {
    "air_leak": (MissionType.OVERHEAD_PIPE, generate_overhead_pipe_mission),
    "gas_leak": (MissionType.OVERHEAD_PIPE, generate_overhead_pipe_mission),
    "overheating": (MissionType.HVAC_DUCT, generate_hvac_duct_mission),
    "hotspot": (MissionType.HVAC_DUCT, generate_hvac_duct_mission),
    "gauge_out_of_range": (MissionType.ELEVATED_GAUGE, generate_elevated_gauge_mission),
}


def should_deploy_drone(fault_type: str, station_config: dict | None = None) -> bool:
    """Determine if a detected fault warrants drone deployment.

    Args:
        fault_type: The type of fault detected by ground sensors
        station_config: Optional station-specific config (e.g., has_overhead_pipes)
    """
    # Always deploy for these fault types
    if fault_type in ("air_leak", "gas_leak"):
        return True

    # Deploy for thermal if station has overhead equipment
    if fault_type in ("overheating", "hotspot"):
        if station_config and station_config.get("has_overhead_equipment"):
            return True

    # Deploy for elevated gauges
    if fault_type == "gauge_out_of_range":
        return True

    return False
