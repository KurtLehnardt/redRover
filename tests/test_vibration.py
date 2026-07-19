"""Tests for vibration signal processing."""

import numpy as np
import pytest

from src.sensors.vibration import (
    VibrationSample,
    compute_fft,
    compute_spectrogram,
    compute_envelope_spectrum,
    extract_features,
)
from src.sensors.simulator import (
    generate_sample,
    generate_normal,
    generate_bearing_fault,
    generate_misalignment,
    generate_imbalance,
    generate_looseness,
    FaultType,
)


def test_normal_signal_low_kurtosis():
    """Normal signals should have near-Gaussian kurtosis (~0)."""
    sample = generate_sample(fault_type=FaultType.NORMAL)
    assert abs(sample.kurtosis) < 2.0


def test_bearing_fault_high_kurtosis():
    """Bearing faults produce impulsive signals with high kurtosis."""
    sample = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.8)
    normal = generate_sample(fault_type=FaultType.NORMAL)
    # Bearing fault should have significantly higher kurtosis than normal
    assert sample.kurtosis > normal.kurtosis + 1.0


def test_bearing_fault_high_crest_factor():
    """Bearing faults should have elevated crest factor."""
    sample = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.8)
    normal = generate_sample(fault_type=FaultType.NORMAL)
    assert sample.crest_factor > normal.crest_factor


def test_misalignment_strong_2x():
    """Misalignment should have strong energy at 2x shaft frequency."""
    sample = generate_sample(fault_type=FaultType.MISALIGNMENT, severity=0.8)
    features = extract_features(sample)
    # 2x of 1800 RPM = 60 Hz → falls in 0-100 Hz band
    # Should have more low-freq energy than a normal signal
    normal = generate_sample(fault_type=FaultType.NORMAL)
    normal_features = extract_features(normal)
    assert features["energy_0_100hz"] > normal_features["energy_0_100hz"]


def test_imbalance_dominant_1x():
    """Imbalance should have dominant 1x shaft frequency (30 Hz for 1800 RPM)."""
    sample = generate_sample(fault_type=FaultType.IMBALANCE, severity=0.8)
    features = extract_features(sample)
    # 1x = 30 Hz, should dominate
    assert features["dominant_frequency_hz"] < 50.0


def test_fft_shape():
    """FFT output should be correctly shaped."""
    sample = generate_sample(fault_type=FaultType.NORMAL)
    freqs, fft_mag = compute_fft(sample)
    expected_len = len(sample.raw_signal) // 2 + 1
    assert len(freqs) == expected_len
    assert len(fft_mag) == expected_len


def test_spectrogram_output():
    """Spectrogram should return valid time-frequency data."""
    sample = generate_sample(fault_type=FaultType.NORMAL)
    freqs, times, sxx = compute_spectrogram(sample)
    assert len(freqs) > 0
    assert len(times) > 0
    assert sxx.shape == (len(freqs), len(times))


def test_extract_features_keys():
    """Feature extraction should return all expected keys."""
    sample = generate_sample(fault_type=FaultType.NORMAL)
    features = extract_features(sample)
    expected_keys = [
        "rms", "peak", "crest_factor", "kurtosis",
        "dominant_frequency_hz",
        "energy_0_100hz", "energy_100_500hz",
        "energy_500_1000hz", "energy_1000_2000hz",
    ]
    for key in expected_keys:
        assert key in features


def test_rms_increases_with_severity():
    """Higher fault severity should produce higher RMS."""
    mild = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.2)
    severe = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.9)
    assert severe.rms > mild.rms


def test_sample_duration():
    """Generated samples should match requested duration."""
    sample = generate_sample(duration=3.0, sample_rate=4000)
    expected_samples = 3.0 * 4000
    assert len(sample.raw_signal) == expected_samples
