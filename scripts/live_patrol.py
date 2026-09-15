#!/usr/bin/env python3
"""
redRover — Live AI-Driven Patrol Script
========================================

Runs a full patrol using the physical Sphero RVR+ over BLE, the Mac's
microphone for acoustic analysis, and local Ollama/Gemma 4 for AI fusion.

Usage:
    python scripts/live_patrol.py
    python scripts/live_patrol.py --stations 6 --speed 100
    python scripts/live_patrol.py --simulate --skip-ai
    python scripts/live_patrol.py --duration 3.0 --speed 60

Requirements:
    - Sphero RVR+ paired via BLE (or use --simulate)
    - Ollama running locally with Gemma 4 (or use --skip-ai for rule-based)
    - sounddevice + numpy for microphone capture
    - Mac with `say` command for text-to-speech
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

# ---------------------------------------------------------------------------
# Path setup — allow `from src.…` imports when running as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from src.ai.fusion import FusionAnalyzer, OverallHealth
from src.alerting import get_alert_manager
from src.config import load_config
from src.database import Database, DiagnosisRecord
from src.rover.backends import create_rover
from src.rover.controller import Waypoint
from src.sensors.simulator import (
    generate_acoustic_sample,
)
from src.sensors.simulator import (
    generate_sample as sim_vibration_sample,
)
from src.sensors.sources import (
    MicrophoneAcousticSource,
    RoverIMUVibrationSource,
    SourceUnavailable,
)
from src.sensors.vibration import extract_features
from src.telemetry import get_meter, init_telemetry, shutdown_telemetry

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_WHITE = "\033[97m"
_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"
_BG_BLUE = "\033[44m"
_BG_CYAN = "\033[46m"

logger = logging.getLogger("redRover.live_patrol")


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

def _banner(text: str, color: str = _CYAN) -> None:
    width = 60
    border = "=" * width
    pad = (width - len(text) - 2) // 2
    print(f"\n{color}{_BOLD}{border}")
    print(f"{' ' * pad} {text} {' ' * pad}")
    print(f"{border}{_RESET}\n")


def _section(text: str, color: str = _BLUE) -> None:
    print(f"\n{color}{_BOLD}--- {text} ---{_RESET}")


def _info(text: str) -> None:
    print(f"  {_DIM}{text}{_RESET}")


def _status(label: str, value: str, color: str = _WHITE) -> None:
    print(f"  {_BOLD}{label:.<30}{_RESET} {color}{value}{_RESET}")


def _health_color(health: OverallHealth) -> str:
    return {
        OverallHealth.HEALTHY: _GREEN,
        OverallHealth.MONITOR: _YELLOW,
        OverallHealth.WARNING: _YELLOW,
        OverallHealth.CRITICAL: _RED,
    }.get(health, _WHITE)


def _health_badge(health: OverallHealth) -> str:
    bg = {
        OverallHealth.HEALTHY: _BG_GREEN,
        OverallHealth.MONITOR: _BG_YELLOW,
        OverallHealth.WARNING: _BG_YELLOW,
        OverallHealth.CRITICAL: _BG_RED,
    }.get(health, "")
    return f"{bg}{_BOLD} {health.value.upper()} {_RESET}"


# ---------------------------------------------------------------------------
# macOS text-to-speech (non-blocking)
# ---------------------------------------------------------------------------

def _say(message: str) -> None:
    """Announce a message via macOS `say` command (non-blocking)."""
    try:
        subprocess.Popen(
            ["say", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.debug("`say` command not found — TTS disabled")


# ---------------------------------------------------------------------------
# Microphone capture
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

async def _check_ollama(host: str, model: str) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    from src.ai.ollama import OllamaClient

    client = OllamaClient(host=host, model=model)
    try:
        present, available = await client.available()
    finally:
        await client.aclose()
    if not present:
        logger.warning(
            "Model '%s' not found in Ollama. Available: %s",
            model, ", ".join(available) or "(none)",
        )
    return present


# ---------------------------------------------------------------------------
# Route generation
# ---------------------------------------------------------------------------

def _generate_patrol_route(
    n_stations: int,
    drive_duration: float,
    speed: int,
) -> list[Waypoint]:
    """Generate a rectangular/polygonal patrol route.

    Headings cycle through 0, 90, 180, 270 (like the EDU script).
    Waypoint positions are approximate — the RVR+ drives on heading for
    `drive_duration` seconds at `speed`, so exact coordinates are estimates.
    """
    station_names = [
        "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
        "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima",
    ]
    headings = [0, 90, 180, 270]

    # Approximate distance per leg (speed is 0-255, ~speed/100 m/s is rough)
    leg_m = (speed / 100.0) * drive_duration

    waypoints = []
    x, y = 0.0, 0.0
    import math
    for i in range(n_stations):
        heading = headings[i % len(headings)]
        rad = math.radians(heading)
        x += leg_m * math.cos(rad)
        y += leg_m * math.sin(rad)
        name = station_names[i] if i < len(station_names) else f"Station-{i+1}"
        waypoints.append(Waypoint(
            station_id=f"STN-{i+1:03d}",
            x=round(x, 2),
            y=round(y, 2),
            heading=float(heading),
            name=name,
        ))
    return waypoints


# ---------------------------------------------------------------------------
# Main patrol
# ---------------------------------------------------------------------------

async def run_patrol(args: argparse.Namespace) -> None:
    """Execute the full patrol sequence."""

    # -- Load config --------------------------------------------------------
    config = load_config()

    # -- Telemetry ----------------------------------------------------------
    init_telemetry(
        service_name=config.telemetry.service_name,
        endpoint=config.telemetry.endpoint or None,
        enabled=config.telemetry.enabled,
        export_interval_ms=config.telemetry.export_interval_ms,
        console_export=config.telemetry.console_export,
        environment=config.telemetry.environment,
    )
    meter = get_meter("redrover.live_patrol")
    station_counter = meter.create_counter("redrover.patrol.stations_visited")
    fault_counter = meter.create_counter("redrover.patrol.faults_detected")
    patrol_duration_hist = meter.create_histogram(
        "redrover.patrol.duration_seconds", unit="s",
    )

    # -- Database -----------------------------------------------------------
    db = Database(path=config.database.path)
    await db.init()

    # -- Rover connection ---------------------------------------------------
    simulate = args.simulate
    speed_normalized = args.speed / 255.0  # RoverController expects 0.0-1.0

    # The factory picks the backend from [rover].connection, so this script
    # works with a firmware rover as well as an RVR+.
    config.rover.speed = speed_normalized
    rover = create_rover(config, simulate=simulate)

    # -- Fusion analyzer ----------------------------------------------------
    ai_model = config.ai.model
    ollama_host = config.ai.ollama_host
    fusion = FusionAnalyzer(model=ai_model, ollama_host=ollama_host)

    # -- Alert manager ------------------------------------------------------
    alert_mgr = get_alert_manager(config)

    # -- Graceful shutdown state --------------------------------------------
    shutdown_event = asyncio.Event()

    def _handle_sigint():
        print(f"\n\n{_RED}{_BOLD}  CTRL+C received — emergency stop ...{_RESET}\n")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _handle_sigint)

    # ======================================================================
    # STARTUP
    # ======================================================================
    _banner("redRover — Live AI Patrol", _CYAN)

    _section("Configuration")
    _status("Stations", str(args.stations))
    _status("Speed", f"{args.speed}/255")
    _status("Drive duration", f"{args.duration}s per leg")
    _status("Mode", "SIMULATED" if simulate else "LIVE BLE", _YELLOW if simulate else _GREEN)
    _status("AI model", f"{ai_model} @ {ollama_host}")
    _status("AI fusion", "SKIP (rule-based)" if args.skip_ai else "ENABLED", _YELLOW if args.skip_ai else _GREEN)

    # Connect rover
    _section("Connecting to RVR+")
    try:
        await rover.connect()
        _status("Rover", "CONNECTED", _GREEN)
    except Exception as e:
        print(f"  {_RED}Failed to connect: {e}{_RESET}")
        if not simulate:
            print(f"  {_YELLOW}Hint: use --simulate to run without hardware{_RESET}")
            return
        rover.simulate = True
        simulate = True
        _status("Rover", "FALLBACK TO SIMULATION", _YELLOW)

    # LEDs green + reset yaw
    await rover.set_leds(0, 255, 0)
    await rover.reset_yaw()

    # Battery check
    battery = await rover.get_battery()
    if battery is not None:
        batt_color = _GREEN if battery > 30 else (_YELLOW if battery > 15 else _RED)
        _status("Battery", f"{battery}%", batt_color)
    else:
        _status("Battery", "N/A", _DIM)

    # Ollama check
    _section("AI Backend")
    if args.skip_ai:
        _status("Ollama", "SKIPPED (rule-based mode)", _YELLOW)
        ollama_ok = False
    else:
        ollama_ok = await _check_ollama(ollama_host, ai_model)
        if ollama_ok:
            _status("Ollama", f"{ai_model} READY", _GREEN)
        else:
            _status("Ollama", f"{ai_model} UNAVAILABLE — will use rule-based fallback", _YELLOW)

    # Generate route
    waypoints = _generate_patrol_route(args.stations, args.duration, args.speed)

    _section("Patrol Route")
    for wp in waypoints:
        _info(f"{wp.station_id}  {wp.name:12s}  heading={wp.heading:>5.0f}deg  ({wp.x:.1f}, {wp.y:.1f})")

    # Start patrol in DB
    patrol_started = datetime.now(UTC).isoformat()
    patrol_id = await db.start_patrol(route_name="live_patrol", started_at=patrol_started)
    _status("Patrol ID", str(patrol_id))

    # TTS announcement
    _say("Red Rover live patrol starting. Initiating AI-driven facility health scan.")

    print(f"\n{_GREEN}{_BOLD}  PATROL STARTING{_RESET}\n")

    # ======================================================================
    # PATROL LOOP
    # ======================================================================
    patrol_start_time = time.time()
    results: list[dict] = []
    total_faults = 0

    for idx, waypoint in enumerate(waypoints):
        if shutdown_event.is_set():
            break

        station_start = time.time()
        station_num = idx + 1

        _banner(f"Station {station_num}/{len(waypoints)}: {waypoint.name}", _BLUE)

        # -- a) Navigate ---------------------------------------------------
        print(f"  {_BLUE}[NAV]{_RESET} Driving to {waypoint.name} "
              f"(heading={waypoint.heading:.0f}deg, {args.duration}s @ speed {args.speed}) ...")
        await rover.set_leds(0, 0, 255)  # Blue = navigating
        _say(f"Navigating to station {waypoint.name}")

        # drive_to derives the leg duration from rover.max_speed_mps and
        # stops the motors itself; sleeping again here used to double the leg.
        await rover.drive_to(waypoint)
        print(f"  {_GREEN}[NAV]{_RESET} Arrived at {waypoint.name}")

        if shutdown_event.is_set():
            await rover.stop()
            break

        # -- b) Set LEDs cyan (measuring) ----------------------------------
        await rover.set_leds(0, 255, 255)

        # -- c) IMU / vibration capture ------------------------------------
        _section("Vibration Capture")
        if not simulate:
            # Real capture from the rover's streaming accelerometer.  It samples
            # far below the rate bearing analysis needs, so the sample carries
            # its true rate and bearing verdicts are suppressed downstream
            # rather than being invented.
            imu_source = RoverIMUVibrationSource(
                rover,
                duration=float(config.sensors.measurement_duration),
                period_ms=config.sensors.imu_period_ms,
            )
            try:
                vib_sample = await imu_source.read(waypoint.station_id)
                print(f"  {_CYAN}[IMU]{_RESET} Captured {len(vib_sample.raw_signal)} "
                      f"samples @ {vib_sample.sample_rate} Hz")
            except SourceUnavailable as exc:
                print(f"  {_RED}[IMU]{_RESET} unavailable: {exc}")
                vib_sample = None
        else:
            # Pick a random fault type occasionally for demo variety
            from src.sensors.vibration import FaultType
            fault_choices = list(FaultType)
            weights = [0.5] + [0.5 / (len(fault_choices) - 1)] * (len(fault_choices) - 1)
            fault_idx = np.random.choice(len(fault_choices), p=weights)
            fault = fault_choices[fault_idx]
            severity = float(np.random.uniform(0.3, 0.8))
            vib_sample = sim_vibration_sample(
                station_id=waypoint.station_id,
                fault_type=fault,
                severity=severity,
                sample_rate=config.sensors.sample_rate,
                duration=float(config.sensors.measurement_duration),
            )
            print(f"  {_CYAN}[SIM]{_RESET} Vibration: fault={fault.value}, severity={severity:.2f}")

        vib_features = extract_features(vib_sample) if vib_sample is not None else None
        if vib_features is not None:
            _status("RMS", f"{vib_features['rms']:.4f}")
            _status("Peak", f"{vib_features['peak']:.4f}")
            _status("Crest factor", f"{vib_features['crest_factor']:.2f}")
            _status("Kurtosis", f"{vib_features['kurtosis']:.2f}")
            _status("Dominant freq", f"{vib_features['dominant_frequency_hz']:.1f} Hz")
            _status("Sample rate", f"{vib_features['sample_rate_hz']:.0f} Hz")
            if not vib_features["bearing_analysis_available"]:
                _status("Bearing analysis", "UNAVAILABLE at this sample rate", _YELLOW)
        else:
            _status("Vibration", "NO DATA", _RED)

        if shutdown_event.is_set():
            break

        # -- d) Microphone capture -----------------------------------------
        _section("Acoustic Capture (Microphone)")
        mic_duration = 3.0
        mic_sample_rate = 44100

        if simulate:
            aco_sample = generate_acoustic_sample(
                station_id=waypoint.station_id,
                sample_rate=mic_sample_rate,
                duration=mic_duration,
            )
            print(f"  {_CYAN}[SIM]{_RESET} Acoustic: simulated ambient noise")
        else:
            mic_source = MicrophoneAcousticSource(
                sample_rate=mic_sample_rate,
                duration=mic_duration,
                device=config.sensors.microphone_device or None,
            )
            print(f"  {_CYAN}[MIC]{_RESET} Recording {mic_duration}s @ {mic_sample_rate} Hz ...")
            try:
                aco_sample = await mic_source.read(waypoint.station_id)
                print(f"  {_CYAN}[MIC]{_RESET} Captured: "
                      f"RMS={aco_sample.rms:.6f}  Peak={aco_sample.peak:.6f}")
            except SourceUnavailable as exc:
                print(f"  {_RED}[MIC]{_RESET} unavailable: {exc}")
                aco_sample = None

        if shutdown_event.is_set():
            break

        # -- e) Vibration analysis (already done via extract_features) ------
        #    Features are in vib_features. VibrationSample is vib_sample.

        # -- f) AI Fusion ---------------------------------------------------
        _section("AI Fusion Analysis")
        fusion_start = time.time()

        # Get station history for trend analysis
        station_history = await db.get_station_trend(waypoint.station_id, limit=5)

        diagnosis = await fusion.analyze(
            station_id=waypoint.station_id,
            vibration=vib_sample,
            acoustic=aco_sample,
            thermal=None,  # No thermal camera attached
            station_history=station_history or None,
        )
        fusion_elapsed = time.time() - fusion_start

        _status("Inference mode", diagnosis.inference_mode,
                _GREEN if diagnosis.inference_mode == "llm" else _YELLOW)
        _status("Inference time", f"{fusion_elapsed:.2f}s")

        # -- g) LED feedback ------------------------------------------------
        led_map = {
            OverallHealth.HEALTHY: (0, 255, 0),
            OverallHealth.MONITOR: (255, 255, 0),
            OverallHealth.WARNING: (255, 255, 0),
            OverallHealth.CRITICAL: (255, 0, 0),
        }
        r, g, b = led_map.get(diagnosis.overall_health, (255, 255, 255))
        await rover.set_leds(r, g, b)

        # -- h) Logging -----------------------------------------------------
        measured_at = datetime.now(UTC).isoformat()

        if vib_features is not None:
            measurement_id = await db.log_measurement(
                patrol_id=patrol_id,
                station_id=waypoint.station_id,
                measured_at=measured_at,
                features=vib_features,
                source_name="simulated" if simulate else "rover-imu",
                simulated=simulate,
            )
            await db.log_diagnosis(
                measurement_id,
                waypoint.station_id,
                DiagnosisRecord.from_fused(diagnosis, simulated=simulate),
                measured_at,
            )

        # OTel metrics
        station_counter.add(1, {"station.id": waypoint.station_id})
        if diagnosis.overall_health in (OverallHealth.WARNING, OverallHealth.CRITICAL):
            fault_counter.add(1, {"station.id": waypoint.station_id, "health": diagnosis.overall_health.value})
            total_faults += 1

        # Alert evaluation
        await alert_mgr.evaluate(diagnosis)

        # -- i) Print results -----------------------------------------------
        _section("Diagnosis")
        hc = _health_color(diagnosis.overall_health)
        print(f"\n  {_BOLD}Station:{_RESET}    {waypoint.name} ({waypoint.station_id})")
        print(f"  {_BOLD}Health:{_RESET}     {_health_badge(diagnosis.overall_health)}")
        print(f"  {_BOLD}Confidence:{_RESET} {hc}{diagnosis.overall_confidence:.0%}{_RESET}")
        print(f"  {_BOLD}Priority:{_RESET}   P{diagnosis.priority}")

        if diagnosis.correlated_faults:
            faults_str = ", ".join(diagnosis.correlated_faults)
            print(f"  {_BOLD}Faults:{_RESET}     {hc}{faults_str}{_RESET}")

        if diagnosis.recommendation:
            print(f"  {_BOLD}Action:{_RESET}     {diagnosis.recommendation}")

        if diagnosis.reasoning:
            print(f"  {_BOLD}Reasoning:{_RESET}  {_DIM}{diagnosis.reasoning}{_RESET}")

        # Per-modality breakdown
        if diagnosis.modality_results:
            print(f"\n  {_DIM}Modality breakdown:{_RESET}")
            for mr in diagnosis.modality_results:
                icon = f"{_RED}!" if mr.fault_detected else f"{_GREEN}ok"
                print(f"    {icon}{_RESET} {mr.modality:12s} "
                      f"{'FAULT: ' + mr.fault_type if mr.fault_detected else 'normal':24s} "
                      f"conf={mr.confidence:.0%}  sev={mr.severity}")

        station_elapsed = time.time() - station_start
        print(f"\n  {_DIM}Station completed in {station_elapsed:.1f}s{_RESET}")

        # -- j) Speak diagnosis ---------------------------------------------
        if diagnosis.overall_health == OverallHealth.CRITICAL:
            tts_msg = f"Station {waypoint.name}: Critical! {diagnosis.recommendation}"
        elif diagnosis.overall_health == OverallHealth.WARNING:
            tts_msg = f"Station {waypoint.name}: Warning. {diagnosis.recommendation}"
        elif diagnosis.overall_health == OverallHealth.MONITOR:
            tts_msg = f"Station {waypoint.name}: Monitor. Minor anomaly detected."
        else:
            tts_msg = f"Station {waypoint.name}: Healthy. All sensors nominal."
        _say(tts_msg)

        # Store result for summary
        results.append({
            "station": waypoint.name,
            "station_id": waypoint.station_id,
            "health": diagnosis.overall_health,
            "confidence": diagnosis.overall_confidence,
            "faults": diagnosis.correlated_faults,
            "recommendation": diagnosis.recommendation,
            "elapsed": station_elapsed,
        })

        # Brief pause between stations
        await asyncio.sleep(1.0)

    # ======================================================================
    # RETURN HOME
    # ======================================================================
    if not shutdown_event.is_set():
        _section("Returning Home")
        await rover.set_leds(255, 0, 255)  # Magenta = returning
        _say("All stations scanned. Returning to base.")
        await rover.return_home()
        print(f"  {_GREEN}[NAV]{_RESET} Arrived at home base")

    # ======================================================================
    # PATROL SUMMARY
    # ======================================================================
    patrol_elapsed = time.time() - patrol_start_time
    patrol_duration_hist.record(patrol_elapsed)

    completed_at = datetime.now(UTC).isoformat()
    await db.complete_patrol(
        patrol_id=patrol_id,
        completed_at=completed_at,
        stations=len(results),
        faults=total_faults,
    )

    _banner("PATROL SUMMARY", _MAGENTA)

    _status("Patrol ID", str(patrol_id))
    _status("Duration", f"{patrol_elapsed:.1f}s")
    _status("Stations visited", f"{len(results)}/{len(waypoints)}")
    _status("Faults detected", str(total_faults),
            _GREEN if total_faults == 0 else _RED)

    print(f"\n  {_BOLD}{'Station':15s} {'Health':12s} {'Confidence':12s} {'Faults':30s} {'Time':>6s}{_RESET}")
    print(f"  {'-'*80}")
    for r in results:
        hc = _health_color(r["health"])
        faults_str = ", ".join(r["faults"]) if r["faults"] else "-"
        print(
            f"  {r['station']:15s} "
            f"{hc}{r['health'].value:12s}{_RESET} "
            f"{r['confidence']:>10.0%}   "
            f"{faults_str:30s} "
            f"{r['elapsed']:>5.1f}s"
        )

    # Overall facility health
    print()
    has_critical = any(r["health"] == OverallHealth.CRITICAL for r in results)
    has_warning = any(r["health"] in (OverallHealth.WARNING, OverallHealth.MONITOR) for r in results)

    if has_critical:
        print(f"  {_BG_RED}{_BOLD} FACILITY STATUS: CRITICAL — IMMEDIATE MAINTENANCE REQUIRED {_RESET}")
        _say("Alert: Critical faults detected. Maintenance required.")
    elif has_warning:
        print(f"  {_BG_YELLOW}{_BOLD} FACILITY STATUS: WARNING — SCHEDULE MAINTENANCE {_RESET}")
        _say("Caution: Elevated readings at some stations. Monitor closely.")
    else:
        print(f"  {_BG_GREEN}{_BOLD} FACILITY STATUS: ALL CLEAR {_RESET}")
        _say("All clear. Facility health nominal. Red Rover standing by.")

    print()

    # ======================================================================
    # CLEANUP
    # ======================================================================
    await rover.set_leds(0, 0, 0)
    await rover.disconnect()
    await fusion.aclose()
    await db.close()
    _info("Rover disconnected")
    _info(f"Data saved to {config.database.path}")

    print(f"\n{_GREEN}{_BOLD}  Patrol complete.{_RESET}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="redRover — Live AI-Driven Patrol (Sphero RVR+ over BLE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/live_patrol.py                        # 4-station BLE patrol
  python scripts/live_patrol.py --simulate             # No hardware needed
  python scripts/live_patrol.py --stations 6 --speed 100
  python scripts/live_patrol.py --skip-ai              # Rule-based only
  python scripts/live_patrol.py --simulate --skip-ai   # Full dry-run
        """,
    )
    parser.add_argument(
        "--stations", type=int, default=4,
        help="Number of patrol stations (default: 4)",
    )
    parser.add_argument(
        "--speed", type=int, default=80,
        help="Drive speed 0-255 (default: 80)",
    )
    parser.add_argument(
        "--duration", type=float, default=2.5,
        help="Drive time between stations in seconds (default: 2.5)",
    )
    parser.add_argument(
        "--skip-ai", action="store_true",
        help="Skip Ollama/Gemma — use rule-based analysis only",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Simulate rover (no BLE connection needed)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Validate
    args.speed = max(0, min(255, args.speed))
    args.stations = max(1, min(12, args.stations))
    args.duration = max(0.5, min(30.0, args.duration))

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Run
    try:
        asyncio.run(run_patrol(args))
    except KeyboardInterrupt:
        print(f"\n{_RED}Interrupted.{_RESET}")
        sys.exit(1)
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
