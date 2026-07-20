"""Tests for acoustic and thermal sensor modules."""

import numpy as np
import pytest

from src.sensors.acoustic import (
    AcousticFaultType,
    AcousticSample,
    compute_mel_spectrogram,
    extract_acoustic_features,
)
from src.sensors.thermal import (
    ThermalFaultType,
    ThermalFrame,
    detect_hotspots,
    extract_thermal_features,
    classify_thermal_severity,
)
from src.sensors.simulator import (
    generate_acoustic_sample,
    generate_thermal_frame,
)


# =============================================================================
# ACOUSTIC TESTS
# =============================================================================


def test_acoustic_normal_low_ultrasonic():
    """Normal signal has low ultrasonic energy."""
    sample = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    assert sample.ultrasonic_energy < 0.01


def test_air_leak_high_ultrasonic():
    """Air leak signal has significantly higher ultrasonic energy than normal."""
    normal = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    leak = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.8)
    assert leak.ultrasonic_energy > normal.ultrasonic_energy * 5


def test_arcing_high_rms_variance():
    """Arcing signal has intermittent bursts (higher RMS variance than normal)."""
    normal = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    arcing = generate_acoustic_sample(fault_type=AcousticFaultType.ARCING, severity=0.8)

    normal_features = extract_acoustic_features(normal)
    arcing_features = extract_acoustic_features(arcing)

    assert arcing_features["rms_variance"] > normal_features["rms_variance"]


def test_acoustic_feature_extraction_keys():
    """All expected keys present in extracted features."""
    sample = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    features = extract_acoustic_features(sample)

    expected_keys = {
        "rms",
        "peak",
        "ultrasonic_energy",
        "rms_variance",
        "rms_std",
        "acoustic_audible_low",
        "acoustic_audible_mid",
        "acoustic_audible_high",
        "acoustic_ultrasonic_low",
        "acoustic_ultrasonic_mid",
        "acoustic_ultrasonic_high",
    }
    assert expected_keys.issubset(features.keys())


def test_mel_spectrogram_shape():
    """Mel spectrogram output has correct dimensions."""
    sample = generate_acoustic_sample(fault_type=AcousticFaultType.NORMAL)
    n_mels = 64
    mel_spec = compute_mel_spectrogram(sample, n_mels=n_mels)

    # First dimension should be n_mels
    assert mel_spec.shape[0] == n_mels
    # Second dimension (time frames) should be > 0
    assert mel_spec.shape[1] > 0


def test_acoustic_sample_properties():
    """RMS and peak are computed correctly (non-zero for non-silent signal)."""
    sample = generate_acoustic_sample(fault_type=AcousticFaultType.AIR_LEAK, severity=0.5)

    assert sample.rms > 0
    assert sample.peak > 0
    assert sample.peak >= sample.rms


# =============================================================================
# THERMAL TESTS
# =============================================================================


def test_thermal_normal_no_hotspots():
    """Normal thermal frame has no hotspots detected."""
    frame = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL, ambient=22.0)
    hotspots = detect_hotspots(frame)
    assert len(hotspots) == 0


def test_thermal_hotspot_detected():
    """Hotspot frame with severity 0.8 has at least one hotspot."""
    frame = generate_thermal_frame(
        fault_type=ThermalFaultType.HOTSPOT, severity=0.8, ambient=22.0
    )
    hotspots = detect_hotspots(frame)
    assert len(hotspots) >= 1


def test_thermal_hotspot_max_temp_elevated():
    """Hotspot frame max temp is well above ambient."""
    ambient = 22.0
    frame = generate_thermal_frame(
        fault_type=ThermalFaultType.HOTSPOT, severity=0.8, ambient=ambient
    )
    # With severity 0.8, hotspot_temp = 60 + 0.8*40 = 92 C
    # Max temp should be well above ambient
    assert frame.max_temp > ambient + 20


def test_thermal_severity_classification():
    """Overheating frame classified as critical/severe."""
    frame = generate_thermal_frame(
        fault_type=ThermalFaultType.OVERHEATING, severity=1.0, ambient=22.0
    )
    fault_type, severity_label = classify_thermal_severity(frame)

    assert severity_label in ("critical", "severe")


def test_thermal_feature_extraction_keys():
    """All expected keys present in thermal features."""
    frame = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL, ambient=22.0)
    features = extract_thermal_features(frame)

    expected_keys = {
        "max_temp_c",
        "min_temp_c",
        "mean_temp_c",
        "temp_range_c",
        "delta_above_ambient_c",
        "max_gradient_c_per_pixel",
        "hotspot_count",
        "hotspot_max_temp_c",
        "hotspot_total_pixels",
        "quadrant_temps",
    }
    assert expected_keys == set(features.keys())


def test_thermal_gradient_detection():
    """Hotspot frame has higher gradient than normal frame."""
    normal = generate_thermal_frame(fault_type=ThermalFaultType.NORMAL, ambient=22.0)
    hotspot = generate_thermal_frame(
        fault_type=ThermalFaultType.HOTSPOT, severity=0.8, ambient=22.0
    )

    normal_features = extract_thermal_features(normal)
    hotspot_features = extract_thermal_features(hotspot)

    assert hotspot_features["max_gradient_c_per_pixel"] > normal_features["max_gradient_c_per_pixel"]


def test_detect_hotspots_threshold():
    """Hotspots only returned when threshold is exceeded."""
    frame = generate_thermal_frame(
        fault_type=ThermalFaultType.HOTSPOT, severity=0.8, ambient=22.0
    )

    # With a very high threshold, no hotspots should be detected
    hotspots_strict = detect_hotspots(frame, threshold_delta=200.0)
    assert len(hotspots_strict) == 0

    # With default threshold (15.0), hotspots should be detected
    hotspots_default = detect_hotspots(frame, threshold_delta=15.0)
    assert len(hotspots_default) >= 1
