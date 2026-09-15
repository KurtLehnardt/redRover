"""Tests for vibration signal processing."""

import numpy as np
import pytest

from src.sensors.simulator import (
    FaultType,
    generate_bearing_fault,
    generate_imbalance,
    generate_looseness,
    generate_misalignment,
    generate_normal,
    generate_sample,
)
from src.sensors.vibration import (
    VibrationSample,
    bearing_defect_frequencies,
    compute_envelope_spectrum,
    compute_fft,
    compute_spectrogram,
    envelope_defect_energy,
    extract_features,
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
        "rms",
        "peak",
        "crest_factor",
        "kurtosis",
        "dominant_frequency_hz",
        "energy_0_100hz",
        "energy_100_500hz",
        "energy_500_1000hz",
        "energy_1000_2000hz",
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


# === Envelope analysis (the documented bearing technique) ===


def test_envelope_spectrum_shape():
    sample = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.8)
    freqs, envelope = compute_envelope_spectrum(sample)
    assert len(freqs) == len(envelope)
    assert len(freqs) == len(sample.raw_signal) // 2 + 1


def test_envelope_spectrum_rejects_unusable_sample_rate():
    """A 1 kHz sample cannot carry the 1 kHz+ resonance band."""
    sample = VibrationSample(
        station_id="M-1",
        timestamp=0.0,
        raw_signal=np.zeros(1000, dtype=np.float32),
        sample_rate=1000,
        duration=1.0,
    )
    assert sample.supports_bearing_analysis is False
    with pytest.raises(ValueError, match="envelope analysis needs"):
        compute_envelope_spectrum(sample)


def test_envelope_energy_higher_for_bearing_fault():
    """Envelope defect energy separates a bearing fault from a healthy machine."""
    faulty = generate_sample(fault_type=FaultType.BEARING_OUTER, severity=0.9)
    healthy = generate_sample(fault_type=FaultType.NORMAL)

    faulty_energy = envelope_defect_energy(faulty, rpm=1800.0)
    healthy_energy = envelope_defect_energy(healthy, rpm=1800.0)

    assert faulty_energy and healthy_energy
    assert faulty_energy["envelope_bpfo_hz"] > healthy_energy["envelope_bpfo_hz"]


def test_envelope_energy_absent_when_not_measurable():
    """Missing is reported as an empty dict, not as zero energy."""
    sample = VibrationSample(
        station_id="M-1",
        timestamp=0.0,
        raw_signal=np.zeros(500, dtype=np.float32),
        sample_rate=100,
        duration=5.0,
    )
    assert envelope_defect_energy(sample) == {}


def test_bearing_defect_frequencies_scale_with_rpm():
    slow = bearing_defect_frequencies(900.0)
    fast = bearing_defect_frequencies(1800.0)
    assert fast["bpfo_hz"] == pytest.approx(slow["bpfo_hz"] * 2)
    assert fast["bpfi_hz"] > fast["bpfo_hz"]  # inner race rides faster


# === Generator coverage ===


def test_each_generator_produces_the_requested_length():
    for gen in (generate_normal, generate_misalignment, generate_imbalance, generate_looseness):
        signal = gen(sample_rate=2000, duration=1.0)
        assert len(signal) == 2000


def test_bearing_generator_severity_increases_impulsiveness():
    mild = VibrationSample(
        "M",
        0.0,
        generate_bearing_fault(4000, 2.0, severity=0.2),
        4000,
        2.0,
    )
    severe = VibrationSample(
        "M",
        0.0,
        generate_bearing_fault(4000, 2.0, severity=0.9),
        4000,
        2.0,
    )
    assert severe.kurtosis > mild.kurtosis
