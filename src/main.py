"""redRover — Multi-Modal Facility Health Robot with Aerial Drone.

Main entry point. Runs a patrol cycle:

1. Connect to RVR+ and drone
2. Drive to each machine station (route comes from ``config/default.toml``)
3. Measure: vibration + acoustic + thermal
4. Fuse sensor data with local AI
5. If aerial inspection needed -> deploy drone from cradle
6. Log measurement AND diagnosis, alert if a fault is detected
7. Return home

Simulated and real runs are never mixed: ``--real`` binds every modality to
actual hardware, and a modality with no hardware behind it is recorded as a
sensor failure rather than filled in with synthetic data.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import UTC, datetime

from .ai.fusion import FusedDiagnosis, FusionAnalyzer, OverallHealth
from .alerting import get_alert_manager
from .config import Settings, load_config
from .database import Database, DiagnosisRecord
from .drone.controller import DroneController, DroneType
from .drone.orchestrator import DroneRoverOrchestrator
from .rover.controller import PatrolRoute, RoverController, Waypoint
from .sensors.acoustic import AcousticFaultType
from .sensors.sources import SensorSuite, SourceUnavailable, build_sensor_suite
from .sensors.thermal import ThermalFaultType
from .sensors.vibration import FaultType, extract_features
from .telemetry import get_meter, get_tracer, init_telemetry, shutdown_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("redRover")


# Fault scenarios used only when running simulated.  Real runs read hardware.
DEMO_SCENARIOS: dict[str, dict] = {
    "M-001": {
        "vibration": (FaultType.NORMAL, 0.0),
        "acoustic": (AcousticFaultType.NORMAL, 0.0),
        "thermal": (ThermalFaultType.NORMAL, 0.0),
    },
    "M-002": {
        # Bearing failure: confirmed by vibration + thermal
        "vibration": (FaultType.BEARING_OUTER, 0.7),
        "acoustic": (AcousticFaultType.FRICTION, 0.4),
        "thermal": (ThermalFaultType.HOTSPOT, 0.6),
    },
    "M-003": {
        # Air leak near compressor — drone should inspect overhead pipes
        "vibration": (FaultType.NORMAL, 0.0),
        "acoustic": (AcousticFaultType.AIR_LEAK, 0.7),
        "thermal": (ThermalFaultType.NORMAL, 0.0),
    },
    "M-004": {
        # Misalignment causing overheating
        "vibration": (FaultType.MISALIGNMENT, 0.5),
        "acoustic": (AcousticFaultType.NORMAL, 0.0),
        "thermal": (ThermalFaultType.HOTSPOT, 0.4),
    },
}

HEALTH_ICON = {
    OverallHealth.HEALTHY: "OK",
    OverallHealth.MONITOR: "..",
    OverallHealth.WARNING: "!!",
    OverallHealth.CRITICAL: "XX",
}


def route_from_config(config: Settings) -> PatrolRoute:
    """Build the patrol route from configuration."""
    waypoints = [
        Waypoint(
            station_id=wp.station_id,
            x=wp.x,
            y=wp.y,
            heading=wp.heading,
            name=wp.name or wp.station_id,
        )
        for wp in config.route.waypoints
    ]
    return PatrolRoute(name=config.route.name, waypoints=waypoints)


def station_configs(config: Settings) -> dict[str, dict]:
    """Per-station metadata used by the drone deployment rules."""
    return {
        wp.station_id: {
            "has_overhead_equipment": wp.has_overhead_equipment,
            "machine_type": wp.machine_type,
            "rpm": wp.rpm,
        }
        for wp in config.route.waypoints
    }


async def _sync_stations(db: Database, config: Settings) -> None:
    for wp in config.route.waypoints:
        await db.upsert_station(
            station_id=wp.station_id,
            name=wp.name or wp.station_id,
            x=wp.x,
            y=wp.y,
            heading=wp.heading,
            machine_type=wp.machine_type or None,
            rpm=wp.rpm,
        )


async def _read_modality(
    suite_source,
    station_id: str,
    label: str,
    failures: dict[str, int],
    counter,
):
    """Read one modality, converting unavailability into a recorded failure."""
    try:
        return await suite_source.read(station_id)
    except SourceUnavailable as exc:
        logger.warning("  [%s] unavailable: %s", label, exc)
    except Exception as exc:
        logger.warning("  [%s] sensor failed: %s", label, exc)
    failures[label] += 1
    counter.add(1, {"sensor.type": label, "station.id": station_id})
    return None


async def run_patrol(
    simulate: bool = True,
    skip_ai: bool = False,
    enable_drone: bool = True,
    config: Settings | None = None,
    scenarios: dict[str, dict] | None = None,
) -> list[FusedDiagnosis]:
    """Execute a single patrol cycle with multi-modal sensor fusion + drone."""
    config = config or load_config()
    route = route_from_config(config)
    stations_meta = station_configs(config)
    scenarios = DEMO_SCENARIOS if scenarios is None else scenarios

    init_telemetry(
        service_name=config.telemetry.service_name,
        endpoint=config.telemetry.endpoint or None,
        enabled=config.telemetry.enabled,
        export_interval_ms=config.telemetry.export_interval_ms,
        console_export=config.telemetry.console_export,
        environment=config.telemetry.environment,
    )
    tracer = get_tracer()
    meter = get_meter()

    patrol_duration_hist = meter.create_histogram(
        "redrover.patrol.duration_seconds",
        description="Duration of a full patrol cycle", unit="s",
    )
    station_measurement_hist = meter.create_histogram(
        "redrover.station.measurement_seconds",
        description="Time to measure a single station", unit="s",
    )
    faults_detected_counter = meter.create_counter(
        "redrover.patrol.faults_detected",
        description="Total faults detected across patrols",
    )
    sensor_failure_counter = meter.create_counter(
        "redrover.sensor.failures", description="Sensor read failures by type",
    )

    rover = RoverController(
        connection=config.rover.connection,
        speed=config.rover.speed,
        simulate=simulate,
        max_speed_mps=config.rover.max_speed_mps,
        max_drive_seconds=config.rover.max_drive_seconds,
        time_scale=config.simulation.time_scale,
    )
    fusion = FusionAnalyzer(model=config.ai.model, ollama_host=config.ai.ollama_host)
    drone = DroneController(
        drone_type=DroneType.SIMULATED if simulate else DroneType.TELLO,
        time_scale=config.simulation.time_scale,
    )
    orchestrator = DroneRoverOrchestrator(rover=rover, drone=drone)
    db = Database(config.database.path)
    alert_mgr = get_alert_manager(config)

    suite: SensorSuite = build_sensor_suite(
        config, simulate=simulate, rover=rover, scenarios=scenarios,
    )

    results: list[FusedDiagnosis] = []
    failures = {"vibration": 0, "acoustic": 0, "thermal": 0}
    total_stations = len(route.waypoints)
    patrol_start = time.time()
    patrol_span = tracer.start_span("patrol", attributes={
        "patrol.route": route.name,
        "patrol.station_count": total_stations,
        "patrol.simulate": simulate,
        "patrol.skip_ai": skip_ai,
    })

    logger.info("=" * 70)
    logger.info("redRover Multi-Modal Patrol — %s", datetime.now(UTC).isoformat())
    logger.info("Route: %s (%d stations)", route.name, total_stations)
    logger.info("Mode: %s", "SIMULATED" if simulate else "LIVE HARDWARE")
    logger.info("Sensors: %s", suite.describe())
    logger.info("=" * 70)

    patrol_id: int | None = None
    try:
        # Inside the try, so a failure here still runs the finally that closes
        # the database and the LLM client. The scheduler calls run_patrol in a
        # loop, and a leak per failed iteration adds up.
        await db.init()
        await _sync_stations(db, config)

        await rover.connect()
        if enable_drone:
            await drone.connect()
            logger.info("Drone: %s (battery: %s%%)", drone.drone_type.value, await drone.get_battery())

        patrol_id = await db.start_patrol(route.name, datetime.now(UTC).isoformat())

        for waypoint in route.waypoints:
            station_start = time.time()
            logger.info("")
            logger.info("-" * 70)
            logger.info("STATION: %s — %s", waypoint.station_id, waypoint.name)
            logger.info("-" * 70)

            await rover.drive_to(waypoint)

            # === VIBRATION ===
            logger.info("  [vibration] measuring (%ds) via %s ...",
                        config.sensors.measurement_duration, suite.vibration.name)
            vib_sample = await _read_modality(
                suite.vibration, waypoint.station_id, "vibration",
                failures, sensor_failure_counter,
            )
            vib_features = extract_features(vib_sample) if vib_sample else None
            if vib_features:
                logger.info(
                    "  [vibration] RMS=%.3f Peak=%.3f Kurtosis=%.1f Crest=%.1f @%.0fHz",
                    vib_features["rms"], vib_features["peak"], vib_features["kurtosis"],
                    vib_features["crest_factor"], vib_features["sample_rate_hz"],
                )
                if not vib_features["bearing_analysis_available"]:
                    logger.warning(
                        "  [vibration] sample rate too low for bearing analysis; "
                        "bearing verdicts suppressed at this station"
                    )

            # === ACOUSTIC ===
            logger.info("  [acoustic] listening (%.1fs) via %s ...",
                        config.sensors.acoustic_duration, suite.acoustic.name)
            aco_sample = await _read_modality(
                suite.acoustic, waypoint.station_id, "acoustic",
                failures, sensor_failure_counter,
            )
            if aco_sample is not None:
                ultrasonic = aco_sample.ultrasonic_energy
                logger.info(
                    "  [acoustic] RMS=%.4f Ultrasonic=%s",
                    aco_sample.rms,
                    "not measured" if ultrasonic is None else f"{ultrasonic:.4f}",
                )

            # === THERMAL ===
            logger.info("  [thermal] scanning via %s ...", suite.thermal.name)
            thermal_frame = await _read_modality(
                suite.thermal, waypoint.station_id, "thermal",
                failures, sensor_failure_counter,
            )
            if thermal_frame is not None:
                logger.info(
                    "  [thermal] Max=%.1fC Mean=%.1fC Delta=%.1fC",
                    thermal_frame.max_temp, thermal_frame.mean_temp,
                    thermal_frame.delta_above_ambient,
                )

            # === PERSIST MEASUREMENT ===
            measurement_id: int | None = None
            if vib_features is not None and patrol_id is not None:
                measurement_id = await db.log_measurement(
                    patrol_id, waypoint.station_id,
                    datetime.now(UTC).isoformat(), vib_features,
                    source_name=suite.vibration.name,
                    simulated=getattr(suite.vibration, "simulated", False),
                )

            # === FUSION ANALYSIS ===
            diagnosis: FusedDiagnosis | None = None
            if skip_ai:
                logger.info("  [ai] analysis skipped (--skip-ai)")
            else:
                history = await db.get_station_trend(waypoint.station_id, limit=5)
                logger.info("  [ai] fusing %d modality reading(s)%s ...",
                            sum(x is not None for x in (vib_sample, aco_sample, thermal_frame)),
                            f" with {len(history)} historical" if history else "")
                try:
                    diagnosis = await fusion.analyze(
                        station_id=waypoint.station_id,
                        vibration=vib_sample,
                        acoustic=aco_sample,
                        thermal=thermal_frame,
                        station_history=history,
                    )
                except Exception as exc:
                    logger.error("  [ai] fusion analysis failed: %s", exc)

            if diagnosis is not None:
                results.append(diagnosis)
                _log_diagnosis(diagnosis)
                await alert_mgr.evaluate(diagnosis)

                # Persist the diagnosis so trend context exists next patrol.
                if measurement_id is not None:
                    await db.log_diagnosis(
                        measurement_id,
                        waypoint.station_id,
                        DiagnosisRecord.from_fused(diagnosis, simulated=simulate),
                        datetime.now(UTC).isoformat(),
                    )

                if enable_drone and diagnosis.overall_health is not OverallHealth.HEALTHY:
                    await _maybe_deploy_drone(
                        orchestrator, waypoint.station_id, diagnosis,
                        stations_meta.get(waypoint.station_id, {}),
                    )

            station_measurement_hist.record(
                time.time() - station_start, {"station.id": waypoint.station_id},
            )

        logger.info("")
        logger.info("-" * 70)
        await rover.return_home()

    finally:
        n_faults = len([r for r in results if r.overall_health is not OverallHealth.HEALTHY])
        if patrol_id is not None:
            await db.complete_patrol(
                patrol_id, datetime.now(UTC).isoformat(), len(results), n_faults,
            )

        _print_summary(route, results, failures, total_stations, orchestrator,
                       enable_drone, drone)

        patrol_duration = time.time() - patrol_start
        patrol_duration_hist.record(patrol_duration, {"patrol.route": route.name})
        faults_detected_counter.add(n_faults, {"patrol.route": route.name})
        patrol_span.set_attribute("patrol.faults_detected", n_faults)
        patrol_span.set_attribute("patrol.duration_seconds", patrol_duration)
        patrol_span.end()

        if enable_drone:
            await drone.disconnect()
        await rover.disconnect()
        await fusion.aclose()
        await db.close()

    return results


def _log_diagnosis(diagnosis: FusedDiagnosis) -> None:
    icon = HEALTH_ICON.get(diagnosis.overall_health, "??")
    level = (
        logging.WARNING
        if diagnosis.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL)
        else logging.INFO
    )
    logger.log(
        level, "  [ai] %s HEALTH: %s (confidence %.0f%%, priority P%d, %s)",
        icon, diagnosis.overall_health.value.upper(),
        diagnosis.overall_confidence * 100, diagnosis.priority,
        diagnosis.inference_mode,
    )
    if diagnosis.correlated_faults:
        logger.warning("  [ai] faults: %s", ", ".join(diagnosis.correlated_faults))
    if diagnosis.correlation_tags:
        logger.info("  [ai] correlations: %s", ", ".join(diagnosis.correlation_tags))
    if diagnosis.unobservable:
        logger.info("  [ai] not measured: %s", ", ".join(diagnosis.unobservable))
    if diagnosis.recommendation and diagnosis.overall_health is not OverallHealth.HEALTHY:
        logger.warning("  [ai] -> %s", diagnosis.recommendation)
    for mr in diagnosis.modality_results:
        status = f"{mr.fault_type} ({mr.severity})" if mr.fault_detected else "normal"
        logger.info("        %-10s %s [%.0f%%]", mr.modality, status, mr.confidence * 100)


async def _maybe_deploy_drone(
    orchestrator: DroneRoverOrchestrator,
    station_id: str,
    diagnosis: FusedDiagnosis,
    station_config: dict,
) -> None:
    """Deploy the drone at most once per station, for the first eligible fault."""
    for mr in diagnosis.modality_results:
        if not mr.fault_detected:
            continue
        result = await orchestrator.evaluate_and_deploy(
            station_id=station_id,
            fault_type=mr.code.value,
            station_config=station_config,
        )
        if result is not None:
            return


def _print_summary(route, results, failures, total_stations, orchestrator,
                   enable_drone, drone) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("PATROL COMPLETE — SUMMARY")
    logger.info("=" * 70)

    if results:
        buckets = {h: [r for r in results if r.overall_health is h] for h in OverallHealth}
        logger.info(
            "  Stations: %d | Critical: %d | Warning: %d | Monitor: %d | Healthy: %d",
            len(results),
            len(buckets[OverallHealth.CRITICAL]), len(buckets[OverallHealth.WARNING]),
            len(buckets[OverallHealth.MONITOR]), len(buckets[OverallHealth.HEALTHY]),
        )
        if enable_drone:
            logger.info(
                "  Drone deployments: %d (%d successful)",
                orchestrator.total_deployments, orchestrator.successful_deployments,
            )
        for r in sorted(results, key=lambda x: x.priority):
            if r.overall_health is not OverallHealth.HEALTHY:
                logger.warning(
                    "  P%d %s: %s — %s",
                    r.priority, r.station_id, r.overall_health.value.upper(),
                    r.recommendation,
                )
    else:
        logger.info("  Stations visited: %d (no diagnoses produced)", total_stations)

    logger.info(
        "  Sensor reliability: VIB %d/%d ACO %d/%d THM %d/%d",
        total_stations - failures["vibration"], total_stations,
        total_stations - failures["acoustic"], total_stations,
        total_stations - failures["thermal"], total_stations,
    )
    if enable_drone:
        logger.info("  Drone battery remaining: %s%%", drone.last_known_battery)
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="redRover — Multi-Modal Facility Health Robot"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", dest="simulate", action="store_true",
                      help="Run with simulated sensors and rover (default)")
    mode.add_argument("--real", dest="simulate", action="store_false",
                      help="Connect to real hardware; modalities without a driver "
                           "are reported as sensor failures, never simulated")
    parser.set_defaults(simulate=True)
    parser.add_argument("--skip-ai", action="store_true",
                        help="Skip AI analysis (just collect sensor data)")
    parser.add_argument("--no-drone", action="store_true",
                        help="Disable drone deployment")
    args = parser.parse_args()

    try:
        asyncio.run(run_patrol(
            simulate=args.simulate,
            skip_ai=args.skip_ai,
            enable_drone=not args.no_drone,
        ))
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
