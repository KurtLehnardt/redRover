"""redRover — Multi-Modal Facility Health Robot with Aerial Drone.

Main entry point. Runs a patrol cycle:
1. Connect to RVR+ and drone
2. Drive to each machine station
3. Measure: vibration + acoustic + thermal
4. Fuse sensor data with local AI
5. If aerial inspection needed → deploy drone from cradle
6. Log results + alert if fault detected
7. Return home
"""

import asyncio
import argparse
import logging
from datetime import datetime

from .config import load_config
from .rover.controller import RoverController, Waypoint, PatrolRoute
from .sensors.vibration import FaultType, extract_features
from .sensors.acoustic import AcousticFaultType
from .sensors.thermal import ThermalFaultType
from .sensors.simulator import (
    generate_sample,
    generate_acoustic_sample,
    generate_thermal_frame,
)
from .ai.fusion import FusionAnalyzer, OverallHealth
from .drone.controller import DroneController, DroneType
from .drone.orchestrator import DroneRoverOrchestrator
from .database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("redRover")


# Demo patrol route
DEMO_ROUTE = PatrolRoute(
    name="Demo Factory Floor",
    waypoints=[
        Waypoint(station_id="M-001", x=2.0, y=0.0, name="Air Handler Unit #1"),
        Waypoint(station_id="M-002", x=4.0, y=1.0, name="Conveyor Motor #3"),
        Waypoint(station_id="M-003", x=4.0, y=3.0, name="Compressor A"),
        Waypoint(station_id="M-004", x=2.0, y=3.0, name="Pump Station #2"),
    ],
)

# Multi-modal fault scenarios for demo
DEMO_SCENARIOS = {
    "M-001": {
        "vibration": (FaultType.NORMAL, 0.0),
        "acoustic": (AcousticFaultType.NORMAL, 0.0),
        "thermal": (ThermalFaultType.NORMAL, 0.0),
        "has_overhead_equipment": False,
    },
    "M-002": {
        # Bearing failure: confirmed by vibration + thermal
        "vibration": (FaultType.BEARING_OUTER, 0.7),
        "acoustic": (AcousticFaultType.FRICTION, 0.4),
        "thermal": (ThermalFaultType.HOTSPOT, 0.6),
        "has_overhead_equipment": True,
    },
    "M-003": {
        # Air leak near compressor — drone should deploy to inspect overhead pipes
        "vibration": (FaultType.NORMAL, 0.0),
        "acoustic": (AcousticFaultType.AIR_LEAK, 0.7),
        "thermal": (ThermalFaultType.NORMAL, 0.0),
        "has_overhead_equipment": True,
    },
    "M-004": {
        # Misalignment causing overheating
        "vibration": (FaultType.MISALIGNMENT, 0.5),
        "acoustic": (AcousticFaultType.NORMAL, 0.0),
        "thermal": (ThermalFaultType.HOTSPOT, 0.4),
        "has_overhead_equipment": True,
    },
}


async def run_patrol(simulate: bool = True, skip_ai: bool = False, enable_drone: bool = True):
    """Execute a single patrol cycle with multi-modal sensor fusion + drone."""
    config = load_config()

    # Initialize components
    rover = RoverController(
        connection=config.rover.connection,
        speed=config.rover.speed,
        simulate=simulate,
    )
    fusion = FusionAnalyzer(
        model=config.ai.model,
        ollama_host=config.ai.ollama_host,
    )
    drone = DroneController(
        drone_type=DroneType.SIMULATED if simulate else DroneType.TELLO,
    )
    orchestrator = DroneRoverOrchestrator(rover=rover, drone=drone)
    db = Database(config.database.path)
    await db.init()

    logger.info("=" * 70)
    logger.info("redRover Multi-Modal Patrol — %s", datetime.now().isoformat())
    logger.info("Route: %s (%d stations)", DEMO_ROUTE.name, len(DEMO_ROUTE.waypoints))
    logger.info("Modalities: Vibration + Acoustic + Thermal + Aerial Drone")
    logger.info("Drone: %s (battery: %d%%)",
                drone.drone_type.value, drone.battery)
    logger.info("=" * 70)

    await rover.connect()
    await drone.connect()

    patrol_id = await db.start_patrol(DEMO_ROUTE.name, datetime.now().isoformat())
    results = []

    for waypoint in DEMO_ROUTE.waypoints:
        logger.info("")
        logger.info("━" * 70)
        logger.info("STATION: %s — %s", waypoint.station_id, waypoint.name)
        logger.info("━" * 70)

        # Navigate to station
        await rover.drive_to(waypoint)

        # Get scenario for this station
        scenario = DEMO_SCENARIOS.get(waypoint.station_id, {})

        # === VIBRATION ===
        vib_fault, vib_severity = scenario.get("vibration", (FaultType.NORMAL, 0.0))
        logger.info("  [VIB] Measuring vibration (%ds dwell)...", config.sensors.measurement_duration)
        vib_sample = generate_sample(
            station_id=waypoint.station_id,
            fault_type=vib_fault,
            severity=vib_severity,
            sample_rate=config.sensors.sample_rate,
            duration=config.sensors.measurement_duration,
        ) if simulate else generate_sample(station_id=waypoint.station_id)

        vib_features = extract_features(vib_sample)
        logger.info("  [VIB] RMS=%.3f Peak=%.3f Kurtosis=%.1f Crest=%.1f",
                    vib_features["rms"], vib_features["peak"],
                    vib_features["kurtosis"], vib_features["crest_factor"])

        # === ACOUSTIC ===
        aco_fault, aco_severity = scenario.get("acoustic", (AcousticFaultType.NORMAL, 0.0))
        logger.info("  [ACO] Listening for acoustic emissions (3s)...")
        aco_sample = generate_acoustic_sample(
            station_id=waypoint.station_id,
            fault_type=aco_fault,
            severity=aco_severity,
        ) if simulate else generate_acoustic_sample(station_id=waypoint.station_id)

        logger.info("  [ACO] RMS=%.4f Ultrasonic=%.4f",
                    aco_sample.rms, aco_sample.ultrasonic_energy)

        # === THERMAL ===
        therm_fault, therm_severity = scenario.get("thermal", (ThermalFaultType.NORMAL, 0.0))
        logger.info("  [THM] Scanning thermal profile...")
        thermal_frame = generate_thermal_frame(
            station_id=waypoint.station_id,
            fault_type=therm_fault,
            severity=therm_severity,
        ) if simulate else generate_thermal_frame(station_id=waypoint.station_id)

        logger.info("  [THM] Max=%.1f°C Mean=%.1f°C Delta=%.1f°C",
                    thermal_frame.max_temp, thermal_frame.mean_temp,
                    thermal_frame.delta_above_ambient)

        # === FUSION ANALYSIS ===
        diagnosis = None
        if not skip_ai:
            logger.info("  [AI]  Running multi-modal fusion analysis...")
            try:
                diagnosis = await fusion.analyze(
                    station_id=waypoint.station_id,
                    vibration=vib_sample,
                    acoustic=aco_sample,
                    thermal=thermal_frame,
                )
                results.append(diagnosis)

                health_icon = {
                    OverallHealth.HEALTHY: "✓",
                    OverallHealth.MONITOR: "◎",
                    OverallHealth.WARNING: "⚠",
                    OverallHealth.CRITICAL: "✖",
                }
                icon = health_icon.get(diagnosis.overall_health, "?")
                log_level = logging.WARNING if diagnosis.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL) else logging.INFO

                logger.log(log_level,
                    "  [AI]  %s HEALTH: %s (confidence: %.0f%%, priority: %d)",
                    icon, diagnosis.overall_health.value.upper(),
                    diagnosis.overall_confidence * 100, diagnosis.priority)

                if diagnosis.correlated_faults:
                    logger.warning("  [AI]  Correlated faults: %s", ", ".join(diagnosis.correlated_faults))
                if diagnosis.recommendation and diagnosis.overall_health != OverallHealth.HEALTHY:
                    logger.warning("  [AI]  → %s", diagnosis.recommendation)

                for mr in diagnosis.modality_results:
                    status = f"{mr.fault_type} ({mr.severity})" if mr.fault_detected else "normal"
                    logger.info("         %s: %s [%.0f%%]", mr.modality, status, mr.confidence * 100)

            except Exception as e:
                logger.error("  [AI]  Fusion analysis failed: %s", e)
        else:
            logger.info("  [AI]  Analysis skipped (--skip-ai)")

        # === DRONE DEPLOYMENT (if fault warrants aerial inspection) ===
        if enable_drone and diagnosis and diagnosis.overall_health != OverallHealth.HEALTHY:
            # Check each detected fault to see if drone should deploy
            for mr in diagnosis.modality_results:
                if mr.fault_detected:
                    station_config = {
                        "has_overhead_equipment": scenario.get("has_overhead_equipment", False),
                    }
                    deploy_result = await orchestrator.evaluate_and_deploy(
                        station_id=waypoint.station_id,
                        fault_type=mr.fault_type,
                        station_config=station_config,
                    )
                    if deploy_result:
                        # Only deploy once per station
                        break

        # Log measurement to database
        await db.log_measurement(
            patrol_id, waypoint.station_id, datetime.now().isoformat(), vib_features
        )

    # Return home
    logger.info("")
    logger.info("━" * 70)
    await rover.return_home()

    # === PATROL SUMMARY ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("PATROL COMPLETE — SUMMARY")
    logger.info("=" * 70)

    if results:
        critical = [r for r in results if r.overall_health == OverallHealth.CRITICAL]
        warnings = [r for r in results if r.overall_health == OverallHealth.WARNING]
        monitors = [r for r in results if r.overall_health == OverallHealth.MONITOR]
        healthy = [r for r in results if r.overall_health == OverallHealth.HEALTHY]

        logger.info("  Stations: %d | Critical: %d | Warning: %d | Monitor: %d | Healthy: %d",
                    len(results), len(critical), len(warnings), len(monitors), len(healthy))
        logger.info("  Drone deployments: %d (%d successful)",
                    orchestrator.total_deployments, orchestrator.successful_deployments)

        for r in sorted(results, key=lambda x: x.priority):
            if r.overall_health != OverallHealth.HEALTHY:
                logger.warning("  P%d %s: %s — %s",
                               r.priority, r.station_id,
                               r.overall_health.value.upper(),
                               r.recommendation)
    else:
        logger.info("  Stations visited: %d (AI analysis was skipped)", len(DEMO_ROUTE.waypoints))

    logger.info("  Drone battery remaining: %d%%", drone.battery)
    logger.info("=" * 70)

    await db.complete_patrol(
        patrol_id, datetime.now().isoformat(),
        len(DEMO_ROUTE.waypoints),
        len([r for r in results if r.overall_health != OverallHealth.HEALTHY]),
    )
    await drone.disconnect()
    await rover.disconnect()
    return results


def main():
    parser = argparse.ArgumentParser(description="redRover — Multi-Modal Facility Health Robot")
    parser.add_argument("--simulate", action="store_true", default=True,
                        help="Run with simulated hardware (default: true)")
    parser.add_argument("--real", action="store_true",
                        help="Connect to real RVR+ hardware")
    parser.add_argument("--skip-ai", action="store_true",
                        help="Skip AI analysis (just collect sensor data)")
    parser.add_argument("--no-drone", action="store_true",
                        help="Disable drone deployment")
    args = parser.parse_args()

    simulate = not args.real
    asyncio.run(run_patrol(
        simulate=simulate,
        skip_ai=args.skip_ai,
        enable_drone=not args.no_drone,
    ))


if __name__ == "__main__":
    main()
