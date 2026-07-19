"""redRover — Mobile Vibration Analyst for Predictive Maintenance.

Main entry point. Runs a patrol cycle:
1. Connect to RVR+
2. Drive to each machine station
3. Measure vibration
4. Analyze with local AI
5. Log results + alert if fault detected
6. Return home
"""

import asyncio
import argparse
import logging
from datetime import datetime

from .config import load_config
from .rover.controller import RoverController, Waypoint, PatrolRoute
from .sensors.vibration import FaultType, extract_features
from .sensors.simulator import generate_sample
from .ai.analyzer import VibrationAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("redRover")


# Example patrol route for demo
DEMO_ROUTE = PatrolRoute(
    name="Demo Factory Floor",
    waypoints=[
        Waypoint(station_id="M-001", x=2.0, y=0.0, name="Air Handler Unit #1"),
        Waypoint(station_id="M-002", x=4.0, y=1.0, name="Conveyor Motor #3"),
        Waypoint(station_id="M-003", x=4.0, y=3.0, name="Compressor A"),
        Waypoint(station_id="M-004", x=2.0, y=3.0, name="Pump Station #2"),
    ],
)

# Simulated fault scenario for demo
DEMO_FAULTS = {
    "M-001": (FaultType.NORMAL, 0.0),
    "M-002": (FaultType.BEARING_OUTER, 0.6),
    "M-003": (FaultType.NORMAL, 0.0),
    "M-004": (FaultType.MISALIGNMENT, 0.4),
}


async def run_patrol(simulate: bool = True, skip_ai: bool = False):
    """Execute a single patrol cycle."""
    config = load_config()

    # Initialize components
    rover = RoverController(
        connection=config.rover.connection,
        speed=config.rover.speed,
        simulate=simulate,
    )
    analyzer = VibrationAnalyzer(
        model=config.ai.model,
        ollama_host=config.ai.ollama_host,
    )

    logger.info("=" * 60)
    logger.info("redRover Patrol Starting — %s", datetime.now().isoformat())
    logger.info("Route: %s (%d stations)", DEMO_ROUTE.name, len(DEMO_ROUTE.waypoints))
    logger.info("=" * 60)

    await rover.connect()

    results = []

    for waypoint in DEMO_ROUTE.waypoints:
        logger.info("-" * 40)

        # Navigate to station
        await rover.drive_to(waypoint)

        # Measure vibration
        logger.info("Measuring vibration at %s for %ds...",
                    waypoint.name, config.sensors.measurement_duration)

        if simulate:
            fault_type, severity = DEMO_FAULTS.get(
                waypoint.station_id, (FaultType.NORMAL, 0.0)
            )
            sample = generate_sample(
                station_id=waypoint.station_id,
                fault_type=fault_type,
                severity=severity,
                sample_rate=config.sensors.sample_rate,
                duration=config.sensors.measurement_duration,
            )
        else:
            # TODO: Read from real sensor
            sample = generate_sample(station_id=waypoint.station_id)

        # Extract features
        features = extract_features(sample)
        logger.info("  RMS: %.4f | Peak: %.4f | Kurtosis: %.2f | Crest: %.2f",
                    features["rms"], features["peak"],
                    features["kurtosis"], features["crest_factor"])

        # AI analysis
        if not skip_ai:
            try:
                diagnosis = await analyzer.analyze(sample)
                results.append(diagnosis)

                level = "INFO" if diagnosis.fault_type == FaultType.NORMAL else "WARNING"
                logger.log(
                    logging.WARNING if level == "WARNING" else logging.INFO,
                    "  DIAGNOSIS: %s (confidence: %.0f%%, severity: %s)",
                    diagnosis.fault_type.value,
                    diagnosis.confidence * 100,
                    diagnosis.severity,
                )
                if diagnosis.fault_type != FaultType.NORMAL:
                    logger.warning("  ACTION: %s", diagnosis.recommendation)
            except Exception as e:
                logger.error("  AI analysis failed: %s", e)
                logger.info("  Features logged for manual review")
        else:
            logger.info("  AI analysis skipped (--skip-ai flag)")

    # Return home
    logger.info("-" * 40)
    await rover.return_home()

    # Summary
    logger.info("=" * 60)
    logger.info("PATROL COMPLETE")
    faults = [r for r in results if r.fault_type != FaultType.NORMAL]
    logger.info("Stations visited: %d | Faults detected: %d", len(DEMO_ROUTE.waypoints), len(faults))
    for fault in faults:
        logger.warning("  ⚠ %s: %s (%s) — %s",
                       fault.station_id, fault.fault_type.value,
                       fault.severity, fault.recommendation)
    logger.info("=" * 60)

    await rover.disconnect()
    return results


def main():
    parser = argparse.ArgumentParser(description="redRover — Mobile Vibration Analyst")
    parser.add_argument("--simulate", action="store_true", default=True,
                        help="Run with simulated hardware (default: true)")
    parser.add_argument("--real", action="store_true",
                        help="Connect to real RVR+ hardware")
    parser.add_argument("--skip-ai", action="store_true",
                        help="Skip AI analysis (just collect data)")
    args = parser.parse_args()

    simulate = not args.real
    asyncio.run(run_patrol(simulate=simulate, skip_ai=args.skip_ai))


if __name__ == "__main__":
    main()
