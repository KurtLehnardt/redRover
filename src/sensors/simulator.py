"""Simulated sensor data for development without hardware.

Generates realistic signals for various fault conditions across all modalities:
- Vibration (bearing, misalignment, imbalance, looseness)
- Acoustic emission (air leaks, gas leaks, friction, arcing)
- Thermal (hotspots, overheating)
"""

import time
import numpy as np
from numpy.typing import NDArray

from .vibration import VibrationSample, FaultType
from .acoustic import AcousticSample, AcousticFaultType
from .thermal import ThermalFrame, ThermalFaultType


# =============================================================================
# VIBRATION SIMULATION
# =============================================================================

def generate_normal(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
) -> NDArray[np.float32]:
    """Generate a healthy machine vibration signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    sig = 0.5 * np.sin(2 * np.pi * shaft_freq * t)
    sig += 0.2 * np.sin(2 * np.pi * 2 * shaft_freq * t)
    sig += 0.1 * np.sin(2 * np.pi * 3 * shaft_freq * t)
    sig += 0.05 * np.random.randn(len(t))

    return sig.astype(np.float32)


def generate_bearing_fault(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
    fault_type: str = "outer",
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate bearing fault vibration signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    n_balls = 9
    ball_diameter = 7.94e-3
    pitch_diameter = 39.04e-3

    bpfo = (n_balls / 2) * shaft_freq * (1 - ball_diameter / pitch_diameter)
    bpfi = (n_balls / 2) * shaft_freq * (1 + ball_diameter / pitch_diameter)
    bsf = (pitch_diameter / (2 * ball_diameter)) * shaft_freq

    if fault_type == "outer":
        fault_freq = bpfo
    elif fault_type == "inner":
        fault_freq = bpfi
    else:
        fault_freq = bsf

    sig = generate_normal(sample_rate, duration, rpm)

    impulse_period = 1.0 / fault_freq
    impulse_times = np.arange(0, duration, impulse_period)

    for imp_t in impulse_times:
        mask = (t >= imp_t) & (t < imp_t + 0.008)
        decay = np.exp(-500 * (t[mask] - imp_t))
        resonance = np.sin(2 * np.pi * 3000 * (t[mask] - imp_t))
        sig[mask] += severity * 5.0 * decay * resonance

    sig += severity * 0.3 * np.random.randn(len(t))

    return sig.astype(np.float32)


def generate_misalignment(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate misalignment vibration signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    sig = 0.5 * np.sin(2 * np.pi * shaft_freq * t)
    sig += (0.3 + severity * 0.8) * np.sin(2 * np.pi * 2 * shaft_freq * t)
    sig += (0.1 + severity * 0.4) * np.sin(2 * np.pi * 3 * shaft_freq * t)
    sig += 0.05 * np.random.randn(len(t))

    return sig.astype(np.float32)


def generate_imbalance(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate imbalance vibration signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    sig = (0.5 + severity * 2.0) * np.sin(2 * np.pi * shaft_freq * t)
    sig += 0.15 * np.sin(2 * np.pi * 2 * shaft_freq * t)
    sig += 0.05 * np.random.randn(len(t))

    return sig.astype(np.float32)


def generate_looseness(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate mechanical looseness vibration signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    sig = np.zeros(len(t), dtype=np.float32)

    for i in range(1, 8):
        amplitude = (0.3 + severity * 0.5) / i
        sig += amplitude * np.sin(2 * np.pi * i * shaft_freq * t + np.random.rand() * 2 * np.pi)

    sig += severity * 0.4 * np.sin(2 * np.pi * 0.5 * shaft_freq * t)
    sig += 0.1 * np.random.randn(len(t))

    return sig.astype(np.float32)


def generate_sample(
    station_id: str = "SIM-001",
    fault_type: FaultType = FaultType.NORMAL,
    severity: float = 0.5,
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
) -> VibrationSample:
    """Generate a complete simulated vibration sample."""
    generators = {
        FaultType.NORMAL: lambda: generate_normal(sample_rate, duration, rpm),
        FaultType.BEARING_INNER: lambda: generate_bearing_fault(sample_rate, duration, rpm, "inner", severity),
        FaultType.BEARING_OUTER: lambda: generate_bearing_fault(sample_rate, duration, rpm, "outer", severity),
        FaultType.BEARING_BALL: lambda: generate_bearing_fault(sample_rate, duration, rpm, "ball", severity),
        FaultType.MISALIGNMENT: lambda: generate_misalignment(sample_rate, duration, rpm, severity),
        FaultType.IMBALANCE: lambda: generate_imbalance(sample_rate, duration, rpm, severity),
        FaultType.LOOSENESS: lambda: generate_looseness(sample_rate, duration, rpm, severity),
    }

    raw_signal = generators[fault_type]()

    return VibrationSample(
        station_id=station_id,
        timestamp=time.time(),
        raw_signal=raw_signal,
        sample_rate=sample_rate,
        duration=duration,
    )


# =============================================================================
# ACOUSTIC SIMULATION
# =============================================================================

def generate_acoustic_normal(
    sample_rate: int = 96000,
    duration: float = 3.0,
) -> NDArray[np.float32]:
    """Generate normal ambient factory noise."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    # Low-frequency machinery hum
    sig = 0.02 * np.sin(2 * np.pi * 60 * t)
    sig += 0.01 * np.sin(2 * np.pi * 120 * t)
    # Broadband ambient
    sig += 0.005 * np.random.randn(len(t))
    return sig.astype(np.float32)


def generate_air_leak(
    sample_rate: int = 96000,
    duration: float = 3.0,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate pneumatic air leak signal.

    Air leaks produce broadband high-frequency noise (hissing)
    concentrated in 20-40 kHz with steady amplitude.
    """
    t = np.arange(0, duration, 1.0 / sample_rate)

    # Start with ambient
    sig = generate_acoustic_normal(sample_rate, duration)

    # Ultrasonic hissing: band-limited noise in 20-40 kHz
    noise = np.random.randn(len(t)).astype(np.float32)
    from scipy import signal as sp_signal
    sos = sp_signal.butter(4, [20000, 40000], btype="bandpass", fs=sample_rate, output="sos")
    leak_noise = sp_signal.sosfilt(sos, noise)

    # Steady amplitude (characteristic of leaks vs intermittent faults)
    sig += severity * 0.3 * leak_noise.astype(np.float32)

    return sig.astype(np.float32)


def generate_electrical_arcing(
    sample_rate: int = 96000,
    duration: float = 3.0,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate electrical arcing signal.

    Arcing produces intermittent broadband bursts (crackling).
    """
    t = np.arange(0, duration, 1.0 / sample_rate)

    sig = generate_acoustic_normal(sample_rate, duration)

    # Random bursts
    n_bursts = int(5 + severity * 20)
    for _ in range(n_bursts):
        burst_start = np.random.uniform(0, duration - 0.01)
        burst_duration = np.random.uniform(0.001, 0.005)
        mask = (t >= burst_start) & (t < burst_start + burst_duration)
        burst = severity * np.random.uniform(0.5, 2.0) * np.random.randn(int(mask.sum()))
        sig[mask] += burst.astype(np.float32)

    return sig.astype(np.float32)


def generate_metal_friction(
    sample_rate: int = 96000,
    duration: float = 3.0,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate metal-on-metal friction signal.

    High-frequency continuous with harmonic structure.
    """
    t = np.arange(0, duration, 1.0 / sample_rate)

    sig = generate_acoustic_normal(sample_rate, duration)

    # Friction: harmonic series at high frequency
    base_freq = 8000 + np.random.uniform(-1000, 1000)
    for harmonic in range(1, 5):
        freq = base_freq * harmonic
        if freq < sample_rate / 2:
            sig += severity * 0.1 / harmonic * np.sin(2 * np.pi * freq * t)

    # Add some modulation (scraping pattern)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    sig *= (1 + severity * 0.3 * mod).astype(np.float32)

    return sig.astype(np.float32)


def generate_acoustic_sample(
    station_id: str = "SIM-001",
    fault_type: AcousticFaultType = AcousticFaultType.NORMAL,
    severity: float = 0.5,
    sample_rate: int = 96000,
    duration: float = 3.0,
) -> AcousticSample:
    """Generate a simulated acoustic emission sample."""
    generators = {
        AcousticFaultType.NORMAL: lambda: generate_acoustic_normal(sample_rate, duration),
        AcousticFaultType.AIR_LEAK: lambda: generate_air_leak(sample_rate, duration, severity),
        AcousticFaultType.GAS_LEAK: lambda: generate_air_leak(sample_rate, duration, severity * 1.5),
        AcousticFaultType.FRICTION: lambda: generate_metal_friction(sample_rate, duration, severity),
        AcousticFaultType.ARCING: lambda: generate_electrical_arcing(sample_rate, duration, severity),
    }

    raw_signal = generators[fault_type]()

    return AcousticSample(
        station_id=station_id,
        timestamp=time.time(),
        raw_signal=raw_signal,
        sample_rate=sample_rate,
        duration=duration,
    )


# =============================================================================
# THERMAL SIMULATION
# =============================================================================

def generate_thermal_normal(
    resolution: tuple[int, int] = (24, 32),
    ambient: float = 22.0,
    machine_temp: float = 35.0,
) -> NDArray[np.float32]:
    """Generate a normal thermal frame — machine slightly warm, uniform."""
    h, w = resolution
    frame = np.full((h, w), ambient, dtype=np.float32)

    # Machine area (center of frame, slightly above ambient)
    cy, cx = h // 2, w // 2
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            frame[y, x] += max(0, (machine_temp - ambient) * np.exp(-dist / 8))

    # Small noise
    frame += np.random.randn(h, w).astype(np.float32) * 0.5

    return frame


def generate_thermal_hotspot(
    resolution: tuple[int, int] = (24, 32),
    ambient: float = 22.0,
    machine_temp: float = 35.0,
    hotspot_temp: float = 75.0,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate thermal frame with a hotspot (overheating component)."""
    frame = generate_thermal_normal(resolution, ambient, machine_temp)

    h, w = resolution
    # Place hotspot at a random but plausible location
    hy = np.random.randint(h // 4, 3 * h // 4)
    hx = np.random.randint(w // 4, 3 * w // 4)

    actual_temp = machine_temp + severity * (hotspot_temp - machine_temp)

    # Hotspot with thermal spread
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - hy) ** 2 + (x - hx) ** 2)
            if dist < 5:
                frame[y, x] = max(frame[y, x], actual_temp * np.exp(-dist / 2))

    return frame


def generate_thermal_frame(
    station_id: str = "SIM-001",
    fault_type: ThermalFaultType = ThermalFaultType.NORMAL,
    severity: float = 0.5,
    ambient: float = 22.0,
) -> ThermalFrame:
    """Generate a simulated thermal camera frame."""
    if fault_type == ThermalFaultType.NORMAL:
        pixels = generate_thermal_normal(ambient=ambient)
    elif fault_type in (ThermalFaultType.HOTSPOT, ThermalFaultType.OVERHEATING):
        hotspot_temp = 60.0 + severity * 40.0  # 60-100°C depending on severity
        pixels = generate_thermal_hotspot(
            ambient=ambient,
            hotspot_temp=hotspot_temp,
            severity=severity,
        )
    else:
        pixels = generate_thermal_normal(ambient=ambient)

    return ThermalFrame(
        station_id=station_id,
        timestamp=time.time(),
        pixels=pixels,
        ambient_temp=ambient,
    )
