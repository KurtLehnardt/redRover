"""Sensor sources must never fabricate data in real mode."""

from __future__ import annotations

import pytest

from src.config import load_config
from src.sensors.acoustic import AcousticSample
from src.sensors.sources import (
    BEARING_ANALYSIS_MIN_RATE_HZ,
    SourceUnavailable,
    UnavailableSource,
    build_sensor_suite,
)
from src.sensors.vibration import VibrationSample, extract_features


@pytest.fixture
def config():
    return load_config()


def test_simulated_suite_is_entirely_simulated(config):
    suite = build_sensor_suite(config, simulate=True)
    assert suite.vibration.simulated
    assert suite.acoustic.simulated
    assert suite.thermal.simulated
    assert not suite.any_real


def test_real_suite_never_falls_back_to_the_simulator(config):
    """Regression: ``--real`` used to call the simulator on both branches.

    A modality with no driver must surface as unavailable, not as synthetic
    data indistinguishable from a measurement.
    """
    config.sensors.sensor_type = "usb_accel"   # no driver implemented
    config.sensors.microphone_enabled = False
    config.sensors.thermal_camera = "none"

    suite = build_sensor_suite(config, simulate=False, rover=None)
    assert isinstance(suite.vibration, UnavailableSource)
    assert isinstance(suite.acoustic, UnavailableSource)
    assert isinstance(suite.thermal, UnavailableSource)
    for source in (suite.vibration, suite.acoustic, suite.thermal):
        assert source.simulated is False


@pytest.mark.asyncio
async def test_unavailable_source_raises_rather_than_returning_data():
    source = UnavailableSource("thermal", "no camera attached")
    with pytest.raises(SourceUnavailable, match="no camera"):
        await source.read("M-001")


@pytest.mark.asyncio
async def test_imu_source_refuses_a_simulated_rover(config):
    from src.rover.controller import RoverController
    from src.sensors.sources import RoverIMUVibrationSource

    source = RoverIMUVibrationSource(RoverController(simulate=True))
    with pytest.raises(SourceUnavailable, match="simulate mode"):
        await source.read("M-001")


# === Bandwidth honesty ===


def test_low_rate_sample_reports_bands_as_unmeasured():
    """Bands above Nyquist are None (not measured), never 0.0 (measured zero)."""
    import numpy as np

    sample = VibrationSample(
        station_id="M-1", timestamp=0.0,
        raw_signal=np.random.randn(250).astype(np.float32),
        sample_rate=50, duration=5.0,
    )
    features = extract_features(sample)

    # 0-100 Hz straddles the 25 Hz Nyquist: measured, but only in part.
    assert features["energy_0_100hz"] is not None
    assert "energy_0_100hz" in features["partial_bands"]
    # These sit entirely above Nyquist and were never observed.
    assert features["energy_100_500hz"] is None
    assert features["energy_1000_2000hz"] is None
    assert features["bearing_analysis_available"] is False
    assert sample.sample_rate < BEARING_ANALYSIS_MIN_RATE_HZ


def test_high_rate_sample_reports_every_band():
    from src.sensors.simulator import generate_sample

    features = extract_features(generate_sample(sample_rate=4000, duration=1.0))
    for band in ("energy_0_100hz", "energy_100_500hz",
                 "energy_500_1000hz", "energy_1000_2000hz"):
        assert features[band] is not None
    assert features["partial_bands"] == []
    assert features["bearing_analysis_available"] is True


def test_bearing_verdict_suppressed_below_the_usable_rate():
    """An impulsive low-rate signal must not be called a bearing fault."""
    import numpy as np

    from src.ai.faults import FaultCode
    from src.ai.fusion import FusionAnalyzer

    rng = np.random.default_rng(3)
    signal = rng.normal(0, 0.05, 500).astype(np.float32)
    signal[::50] = 5.0  # strong impulses -> high kurtosis and crest factor

    low_rate = VibrationSample("M-1", 0.0, signal, sample_rate=100, duration=5.0)
    analyzer = FusionAnalyzer()
    result = analyzer._analyze_vibration(low_rate)

    assert result.fault_detected  # the energy is real...
    assert result.code is not FaultCode.BEARING_FAULT  # ...but unattributable
    assert "bearing_resonance_band" in result.unobservable


def test_sub_96khz_audio_marks_ultrasonic_unmeasured():
    import numpy as np

    from src.ai.fusion import FusionAnalyzer

    rng = np.random.default_rng(5)
    audio = rng.normal(0, 0.01, 44100).astype(np.float32)
    sample = AcousticSample("M-1", 0.0, audio, sample_rate=44100, duration=1.0)

    assert sample.ultrasonic_energy is None
    assert sample.supports_ultrasonic is False

    result = FusionAnalyzer()._analyze_acoustic(sample)
    assert "ultrasonic_band" in result.unobservable
    # A microphone that cannot hear the band must not be read as "no leak".
    assert result.confidence < 0.85


# === Sampling honesty ===


@pytest.mark.asyncio
async def test_imu_source_reports_delivered_rate_not_poll_rate():
    """Regression: the source used to poll on a timer.

    When BLE delivers slower than the poll loop, polling re-reads the same
    cached tuple, so the sample is padded with duplicates and reports the poll
    rate as the sample rate. That is fabricated bandwidth, and it is what the
    bearing-suppression logic downstream relies on being truthful.
    """
    import asyncio

    from src.rover.controller import RoverController
    from src.sensors.sources import RoverIMUVibrationSource

    rover = RoverController(simulate=False)
    rover._streaming = True  # pretend streaming is already running

    # Deliver 20 packets over ~0.4 s while the source samples for 0.4 s.
    delivered = 20

    async def deliver():
        for i in range(delivered):
            rover.inject_sensor_data(accelerometer=(0.0, 0.0, 1.0 + 0.01 * i))
            await asyncio.sleep(0.02)

    source = RoverIMUVibrationSource(rover, duration=0.4, period_ms=10)
    feeder = asyncio.create_task(deliver())
    sample = await source.read("M-001")
    feeder.cancel()

    # One sample per delivered packet, not one per 10 ms tick.
    assert len(sample.raw_signal) <= delivered
    assert len(sample.raw_signal) >= 8
    # And the reported rate matches what arrived.
    assert sample.sample_rate == pytest.approx(
        len(sample.raw_signal) / sample.duration, rel=0.25
    )
    # Duplicates would show up as repeated values; these are all distinct.
    assert len(set(sample.raw_signal.tolist())) == len(sample.raw_signal)


@pytest.mark.asyncio
async def test_imu_source_ignores_packets_from_other_sensors():
    """A locator packet must not be counted as an accelerometer sample."""
    import asyncio

    from src.rover.controller import RoverController
    from src.sensors.sources import RoverIMUVibrationSource

    rover = RoverController(simulate=False)
    rover._streaming = True

    async def deliver():
        for i in range(12):
            rover.inject_sensor_data(accelerometer=(0.0, 0.0, 1.0 + 0.01 * i))
            await asyncio.sleep(0.01)
            # Interleaved locator packets carry no new acceleration.
            rover.inject_sensor_data(locator=(float(i), 0.0))
            await asyncio.sleep(0.01)

    source = RoverIMUVibrationSource(rover, duration=0.3, period_ms=10)
    feeder = asyncio.create_task(deliver())
    sample = await source.read("M-001")
    feeder.cancel()

    assert 8 <= len(sample.raw_signal) <= 12
