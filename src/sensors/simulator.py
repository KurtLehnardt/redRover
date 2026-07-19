"""Simulated vibration data for development without hardware.

Generates realistic vibration signals for various fault conditions
based on published bearing fault characteristics (CWRU dataset patterns).
"""

import time
import numpy as np
from numpy.typing import NDArray

from .vibration import VibrationSample, FaultType


def generate_normal(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
) -> NDArray[np.float32]:
    """Generate a healthy machine vibration signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    # Fundamental + harmonics (normal running)
    sig = 0.5 * np.sin(2 * np.pi * shaft_freq * t)
    sig += 0.2 * np.sin(2 * np.pi * 2 * shaft_freq * t)
    sig += 0.1 * np.sin(2 * np.pi * 3 * shaft_freq * t)

    # Background noise
    sig += 0.05 * np.random.randn(len(t))

    return sig.astype(np.float32)


def generate_bearing_fault(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
    fault_type: str = "outer",
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate bearing fault vibration signal.

    Args:
        fault_type: "inner", "outer", or "ball"
        severity: 0.0 (incipient) to 1.0 (severe)
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    # Bearing geometry factors (typical for 6205 bearing)
    n_balls = 9
    ball_diameter = 7.94e-3
    pitch_diameter = 39.04e-3
    contact_angle = 0.0

    # Characteristic defect frequencies
    bpfo = (n_balls / 2) * shaft_freq * (1 - ball_diameter / pitch_diameter)
    bpfi = (n_balls / 2) * shaft_freq * (1 + ball_diameter / pitch_diameter)
    bsf = (pitch_diameter / (2 * ball_diameter)) * shaft_freq

    if fault_type == "outer":
        fault_freq = bpfo
    elif fault_type == "inner":
        fault_freq = bpfi
    else:
        fault_freq = bsf

    # Normal component
    sig = generate_normal(sample_rate, duration, rpm)

    # Fault impulses (repetitive impacts at defect frequency)
    impulse_period = 1.0 / fault_freq
    impulse_times = np.arange(0, duration, impulse_period)

    for imp_t in impulse_times:
        # Exponentially decaying impulse
        mask = (t >= imp_t) & (t < imp_t + 0.008)
        decay = np.exp(-500 * (t[mask] - imp_t))
        resonance = np.sin(2 * np.pi * 3000 * (t[mask] - imp_t))
        sig[mask] += severity * 5.0 * decay * resonance

    # Add randomness to amplitude (realistic)
    sig += severity * 0.3 * np.random.randn(len(t))

    return sig.astype(np.float32)


def generate_misalignment(
    sample_rate: int = 4000,
    duration: float = 5.0,
    rpm: float = 1800,
    severity: float = 0.5,
) -> NDArray[np.float32]:
    """Generate misalignment vibration signal.

    Characterized by strong 2x shaft frequency component.
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    sig = 0.5 * np.sin(2 * np.pi * shaft_freq * t)
    # Misalignment: elevated 2x component
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
    """Generate imbalance vibration signal.

    Characterized by dominant 1x shaft frequency.
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    # Imbalance: very strong 1x
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
    """Generate mechanical looseness vibration signal.

    Characterized by many harmonics + sub-harmonics.
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    shaft_freq = rpm / 60.0

    sig = np.zeros(len(t), dtype=np.float32)

    # Many harmonics (characteristic of looseness)
    for i in range(1, 8):
        amplitude = (0.3 + severity * 0.5) / i
        sig += amplitude * np.sin(2 * np.pi * i * shaft_freq * t + np.random.rand() * 2 * np.pi)

    # Sub-harmonics
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
