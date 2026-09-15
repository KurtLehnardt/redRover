"""Sensor acquisition sources — the seam between the patrol loop and hardware.

Every source declares whether it is ``simulated``.  Nothing in this project may
fabricate a reading while claiming to be real: a source that has no hardware
behind it raises :class:`SourceUnavailable` so the caller records a sensor
failure instead of silently logging synthetic data.

Sources also advertise ``max_frequency_hz`` where it matters.  The RVR+ IMU
streams at tens of Hz, which is orders of magnitude below what bearing
envelope analysis needs; callers use this to warn rather than to quietly
produce a meaningless diagnosis.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from . import simulator
from .acoustic import AcousticFaultType, AcousticSample
from .thermal import ThermalFaultType, ThermalFrame
from .vibration import FaultType, VibrationSample

logger = logging.getLogger(__name__)

# Below this sample rate, rolling-element bearing defect frequencies and their
# resonance carriers (typically 1-5 kHz) are not observable at all.
BEARING_ANALYSIS_MIN_RATE_HZ = 2000


class SourceUnavailable(RuntimeError):
    """Raised when a source has no hardware behind it and refuses to guess."""


@runtime_checkable
class SensorSource(Protocol):
    """Common surface for every acquisition source."""

    name: str
    simulated: bool

    async def read(self, station_id: str):
        """Acquire one measurement, or raise :class:`SourceUnavailable`."""
        ...


# ---------------------------------------------------------------------------
# Vibration
# ---------------------------------------------------------------------------


@dataclass
class SimulatedVibrationSource:
    """Synthesises a vibration sample from a per-station fault scenario."""

    sample_rate: int = 4000
    duration: float = 5.0
    rpm: float = 1800.0
    scenarios: dict[str, tuple[FaultType, float]] | None = None
    name: str = "simulated-vibration"
    simulated: bool = True
    max_frequency_hz: float | None = None

    def __post_init__(self):
        self.max_frequency_hz = self.sample_rate / 2

    async def read(self, station_id: str) -> VibrationSample:
        fault, severity = (self.scenarios or {}).get(station_id, (FaultType.NORMAL, 0.0))
        return simulator.generate_sample(
            station_id=station_id,
            fault_type=fault,
            severity=severity,
            sample_rate=self.sample_rate,
            duration=self.duration,
            rpm=self.rpm,
        )


class RoverIMUVibrationSource:
    """Real vibration capture from the rover's streaming accelerometer.

    This is genuine sensor data, but the RVR+ streams at tens of Hz.  Bearing
    diagnosis needs kHz-class sampling — see ``BEARING_ANALYSIS_MIN_RATE_HZ``.
    The sample is tagged with its true rate so downstream feature extraction
    drops the frequency bands it cannot see rather than reporting zeros as if
    they were measurements.
    """

    simulated = False

    def __init__(self, rover, duration: float = 5.0, period_ms: int = 20):
        self.rover = rover
        self.duration = duration
        self.period_ms = max(10, int(period_ms))
        self.name = "rover-imu"
        self.max_frequency_hz = (1000.0 / self.period_ms) / 2

    async def read(self, station_id: str) -> VibrationSample:
        if getattr(self.rover, "simulate", False):
            raise SourceUnavailable("rover is in simulate mode; no real IMU behind it")

        samples: list[float] = []
        started_streaming = False
        if not self.rover.streaming:
            await self.rover.start_sensor_streaming(period_ms=self.period_ms)
            started_streaming = True

        try:
            deadline = time.monotonic() + self.duration
            interval = self.period_ms / 1000.0
            while time.monotonic() < deadline:
                ax, ay, az = self.rover.sensor_data.get("accelerometer", (0.0, 0.0, 0.0))
                # Gravity is a DC offset on whichever axis is down; magnitude
                # minus 1g keeps the AC component that carries the vibration.
                magnitude = float(np.sqrt(ax * ax + ay * ay + az * az))
                samples.append(magnitude - 1.0)
                await asyncio.sleep(interval)
        finally:
            if started_streaming:
                try:
                    await self.rover.stop_sensor_streaming()
                except Exception as exc:  # pragma: no cover - cleanup best effort
                    logger.debug("stop_sensor_streaming during IMU capture: %s", exc)

        if len(samples) < 8:
            raise SourceUnavailable(
                f"IMU produced only {len(samples)} samples in {self.duration}s"
            )

        effective_rate = int(round(len(samples) / self.duration))
        if effective_rate < BEARING_ANALYSIS_MIN_RATE_HZ:
            logger.warning(
                "[VIB] %s sampled at %d Hz — below the %d Hz needed for bearing "
                "analysis. Low-frequency faults (imbalance, misalignment, "
                "looseness) remain valid; bearing verdicts will be suppressed.",
                self.name, effective_rate, BEARING_ANALYSIS_MIN_RATE_HZ,
            )

        return VibrationSample(
            station_id=station_id,
            timestamp=time.time(),
            raw_signal=np.asarray(samples, dtype=np.float32),
            sample_rate=effective_rate,
            duration=self.duration,
        )


# ---------------------------------------------------------------------------
# Acoustic
# ---------------------------------------------------------------------------


@dataclass
class SimulatedAcousticSource:
    sample_rate: int = 96000
    duration: float = 3.0
    scenarios: dict[str, tuple[AcousticFaultType, float]] | None = None
    name: str = "simulated-acoustic"
    simulated: bool = True

    async def read(self, station_id: str) -> AcousticSample:
        fault, severity = (self.scenarios or {}).get(
            station_id, (AcousticFaultType.NORMAL, 0.0)
        )
        return simulator.generate_acoustic_sample(
            station_id=station_id,
            fault_type=fault,
            severity=severity,
            sample_rate=self.sample_rate,
            duration=self.duration,
        )


class MicrophoneAcousticSource:
    """Real capture from a system microphone via ``sounddevice``.

    Ultrasonic leak detection needs >=96 kHz; consumer microphones cap at
    44.1/48 kHz.  The source warns once so an ``air_leak`` that can never fire
    is not mistaken for an absence of leaks.
    """

    simulated = False

    def __init__(self, sample_rate: int = 96000, duration: float = 3.0, device=None):
        self.sample_rate = sample_rate
        self.duration = duration
        self.device = device
        self.name = "microphone"
        self._warned_bandwidth = False

    async def read(self, station_id: str) -> AcousticSample:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SourceUnavailable("sounddevice is not installed") from exc

        if self.sample_rate < 96000 and not self._warned_bandwidth:
            logger.warning(
                "[ACO] %s runs at %d Hz; ultrasonic (20-48 kHz) leak detection "
                "needs >=96 kHz and will be reported as unavailable.",
                self.name, self.sample_rate,
            )
            self._warned_bandwidth = True

        frames = int(self.duration * self.sample_rate)

        def _capture() -> np.ndarray:
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            sd.wait()
            return audio.reshape(-1)

        try:
            signal_1d = await asyncio.to_thread(_capture)
        except Exception as exc:
            raise SourceUnavailable(f"microphone capture failed: {exc}") from exc

        return AcousticSample(
            station_id=station_id,
            timestamp=time.time(),
            raw_signal=np.asarray(signal_1d, dtype=np.float32),
            sample_rate=self.sample_rate,
            duration=self.duration,
        )


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------


@dataclass
class SimulatedThermalSource:
    ambient: float = 22.0
    scenarios: dict[str, tuple[ThermalFaultType, float]] | None = None
    name: str = "simulated-thermal"
    simulated: bool = True

    async def read(self, station_id: str) -> ThermalFrame:
        fault, severity = (self.scenarios or {}).get(
            station_id, (ThermalFaultType.NORMAL, 0.0)
        )
        return simulator.generate_thermal_frame(
            station_id=station_id,
            fault_type=fault,
            severity=severity,
            ambient=self.ambient,
        )


class MLX90640ThermalSource:
    """Real 32x24 thermal frame from an MLX90640 over I2C."""

    simulated = False

    def __init__(self, ambient: float = 22.0, i2c_frequency: int = 800000):
        self.ambient = ambient
        self.i2c_frequency = i2c_frequency
        self.name = "mlx90640"
        self._sensor = None

    def _ensure_sensor(self):
        if self._sensor is not None:
            return self._sensor
        try:
            import adafruit_mlx90640
            import board
            import busio
        except ImportError as exc:
            raise SourceUnavailable(
                "adafruit-circuitpython-mlx90640 is not installed"
            ) from exc
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=self.i2c_frequency)
            self._sensor = adafruit_mlx90640.MLX90640(i2c)
            self._sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        except Exception as exc:
            raise SourceUnavailable(f"MLX90640 not reachable on I2C: {exc}") from exc
        return self._sensor

    async def read(self, station_id: str) -> ThermalFrame:
        sensor = await asyncio.to_thread(self._ensure_sensor)
        buf = [0.0] * 768

        def _grab():
            sensor.getFrame(buf)
            return np.asarray(buf, dtype=np.float32).reshape(24, 32)

        try:
            pixels = await asyncio.to_thread(_grab)
        except Exception as exc:
            raise SourceUnavailable(f"MLX90640 frame read failed: {exc}") from exc

        return ThermalFrame(
            station_id=station_id,
            timestamp=time.time(),
            pixels=pixels,
            ambient_temp=self.ambient,
        )


class UnavailableSource:
    """Placeholder for a modality with no hardware configured."""

    simulated = False

    def __init__(self, modality: str, reason: str):
        self.name = f"unavailable-{modality}"
        self.modality = modality
        self.reason = reason

    async def read(self, station_id: str):
        raise SourceUnavailable(self.reason)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass
class SensorSuite:
    """The three acquisition sources used by a patrol."""

    vibration: SensorSource
    acoustic: SensorSource
    thermal: SensorSource

    @property
    def any_real(self) -> bool:
        return not all(
            getattr(s, "simulated", True)
            for s in (self.vibration, self.acoustic, self.thermal)
        )

    def describe(self) -> str:
        return " | ".join(
            f"{label}={src.name}{'' if getattr(src, 'simulated', True) is False else ' (sim)'}"
            for label, src in (
                ("vib", self.vibration),
                ("aco", self.acoustic),
                ("thm", self.thermal),
            )
        )


def build_sensor_suite(
    config,
    simulate: bool,
    rover=None,
    scenarios: dict[str, dict] | None = None,
) -> SensorSuite:
    """Assemble the sensor suite for a patrol.

    In simulate mode every source is synthetic and says so.  In real mode each
    modality binds to actual hardware, or to :class:`UnavailableSource` when
    that hardware is not configured — never to the simulator.
    """
    scenarios = scenarios or {}

    if simulate:
        return SensorSuite(
            vibration=SimulatedVibrationSource(
                sample_rate=config.sensors.sample_rate,
                duration=float(config.sensors.measurement_duration),
                scenarios={
                    sid: s.get("vibration", (FaultType.NORMAL, 0.0))
                    for sid, s in scenarios.items()
                },
            ),
            acoustic=SimulatedAcousticSource(
                sample_rate=config.sensors.acoustic_sample_rate,
                duration=config.sensors.acoustic_duration,
                scenarios={
                    sid: s.get("acoustic", (AcousticFaultType.NORMAL, 0.0))
                    for sid, s in scenarios.items()
                },
            ),
            thermal=SimulatedThermalSource(
                ambient=config.sensors.ambient_temp_c,
                scenarios={
                    sid: s.get("thermal", (ThermalFaultType.NORMAL, 0.0))
                    for sid, s in scenarios.items()
                },
            ),
        )

    sensor_type = config.sensors.sensor_type
    if sensor_type == "imu" and rover is not None:
        vibration: SensorSource = RoverIMUVibrationSource(
            rover,
            duration=float(config.sensors.measurement_duration),
            period_ms=config.sensors.imu_period_ms,
        )
    else:
        vibration = UnavailableSource(
            "vibration",
            f"no driver for sensor_type={sensor_type!r}; set [sensors].sensor_type "
            "to 'imu' with a connected rover, or attach a firmware rover "
            "(see firmware/README.md) for kHz-rate sampling",
        )

    acoustic: SensorSource
    if config.sensors.microphone_enabled:
        acoustic = MicrophoneAcousticSource(
            sample_rate=config.sensors.acoustic_sample_rate,
            duration=config.sensors.acoustic_duration,
            device=config.sensors.microphone_device or None,
        )
    else:
        acoustic = UnavailableSource(
            "acoustic", "[sensors].microphone_enabled is false"
        )

    thermal: SensorSource
    if config.sensors.thermal_camera == "mlx90640":
        thermal = MLX90640ThermalSource(ambient=config.sensors.ambient_temp_c)
    else:
        thermal = UnavailableSource(
            "thermal",
            f"no driver for thermal_camera={config.sensors.thermal_camera!r}",
        )

    return SensorSuite(vibration=vibration, acoustic=acoustic, thermal=thermal)
