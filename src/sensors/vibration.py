"""Vibration data acquisition and signal processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.stats import kurtosis as _scipy_kurtosis

logger = logging.getLogger(__name__)

# Frequency bands reported by `extract_features`, in Hz.
FEATURE_BANDS: tuple[tuple[int, int], ...] = ((0, 100), (100, 500), (500, 1000), (1000, 2000))

# Lower edge of the resonance band used for bearing envelope demodulation.
ENVELOPE_BAND_LOW_HZ = 1000.0
# Envelope analysis needs headroom above the resonance band to be meaningful.
ENVELOPE_MIN_SAMPLE_RATE_HZ = 2 * (ENVELOPE_BAND_LOW_HZ + 200.0)


class FaultType(str, Enum):
    NORMAL = "normal"
    BEARING_INNER = "bearing_inner_race"
    BEARING_OUTER = "bearing_outer_race"
    BEARING_BALL = "bearing_ball"
    MISALIGNMENT = "misalignment"
    LOOSENESS = "looseness"
    IMBALANCE = "imbalance"


@dataclass
class VibrationSample:
    """A single vibration measurement from a machine station.

    The scalar statistics are computed once on construction: they are read
    several times per station (logging, feature extraction, fusion) and
    recomputing an FFT-adjacent statistic on every attribute access was
    measurable in patrol timings.
    """

    station_id: str
    timestamp: float
    raw_signal: NDArray[np.float32]
    sample_rate: int
    duration: float

    _stats: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        sig = np.asarray(self.raw_signal, dtype=np.float64)
        rms = float(np.sqrt(np.mean(sig ** 2))) if sig.size else 0.0
        peak = float(np.max(np.abs(sig))) if sig.size else 0.0
        self._stats = {
            "rms": rms,
            "peak": peak,
            "crest_factor": (peak / rms) if rms else 0.0,
            "kurtosis": float(_scipy_kurtosis(sig)) if sig.size > 3 else 0.0,
        }

    @property
    def nyquist_hz(self) -> float:
        return self.sample_rate / 2.0

    @property
    def supports_bearing_analysis(self) -> bool:
        """True when the sample rate can actually resolve bearing signatures."""
        return self.sample_rate >= ENVELOPE_MIN_SAMPLE_RATE_HZ

    @property
    def rms(self) -> float:
        """Root mean square of the signal."""
        return self._stats["rms"]

    @property
    def peak(self) -> float:
        """Peak amplitude."""
        return self._stats["peak"]

    @property
    def crest_factor(self) -> float:
        """Peak / RMS — high values indicate impulsive faults."""
        return self._stats["crest_factor"]

    @property
    def kurtosis(self) -> float:
        """Statistical kurtosis — elevated in bearing faults."""
        return self._stats["kurtosis"]


def compute_fft(sample: VibrationSample) -> tuple[NDArray, NDArray]:
    """Compute single-sided FFT magnitude spectrum."""
    n = len(sample.raw_signal)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample.sample_rate)
    fft_mag = np.abs(np.fft.rfft(sample.raw_signal)) * 2.0 / n
    return freqs, fft_mag


def compute_spectrogram(
    sample: VibrationSample,
    nperseg: int = 256,
    noverlap: int = 128,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute spectrogram for time-frequency analysis."""
    nperseg = min(nperseg, len(sample.raw_signal)) or 1
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0
    freqs, times, sxx = signal.spectrogram(
        sample.raw_signal,
        fs=sample.sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return freqs, times, sxx


def compute_envelope_spectrum(sample: VibrationSample) -> tuple[NDArray, NDArray]:
    """Envelope analysis — key technique for bearing fault detection.

    Demodulates the signal to extract repetitive impact patterns that
    correspond to bearing defect frequencies (BPFO / BPFI / BSF).

    Raises:
        ValueError: if the sample rate cannot carry the resonance band.
    """
    if not sample.supports_bearing_analysis:
        raise ValueError(
            f"envelope analysis needs >= {ENVELOPE_MIN_SAMPLE_RATE_HZ:.0f} Hz, "
            f"sample is {sample.sample_rate} Hz"
        )

    high_edge = sample.nyquist_hz - 100.0
    sos = signal.butter(
        4,
        [ENVELOPE_BAND_LOW_HZ, high_edge],
        btype="bandpass",
        fs=sample.sample_rate,
        output="sos",
    )
    filtered = signal.sosfilt(sos, sample.raw_signal)

    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    n = len(envelope)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample.sample_rate)
    env_fft = np.abs(np.fft.rfft(envelope)) * 2.0 / n

    return freqs, env_fft


def bearing_defect_frequencies(rpm: float, n_balls: int = 9,
                               ball_diameter_m: float = 7.94e-3,
                               pitch_diameter_m: float = 39.04e-3) -> dict[str, float]:
    """Classical rolling-element defect frequencies for a given shaft speed."""
    shaft_hz = rpm / 60.0
    ratio = ball_diameter_m / pitch_diameter_m
    return {
        "shaft_hz": shaft_hz,
        "bpfo_hz": (n_balls / 2) * shaft_hz * (1 - ratio),
        "bpfi_hz": (n_balls / 2) * shaft_hz * (1 + ratio),
        "bsf_hz": (pitch_diameter_m / (2 * ball_diameter_m)) * shaft_hz,
    }


def envelope_defect_energy(sample: VibrationSample, rpm: float = 1800.0,
                           tolerance_hz: float = 3.0) -> dict[str, float]:
    """Energy at each bearing defect frequency in the envelope spectrum.

    Returns an empty dict when the sample rate is too low for envelope
    analysis, so callers can distinguish "no defect energy" from "not measured".
    """
    if not sample.supports_bearing_analysis:
        return {}

    freqs, env = compute_envelope_spectrum(sample)
    defects = bearing_defect_frequencies(rpm)
    out: dict[str, float] = {}
    for name, target in defects.items():
        if name == "shaft_hz" or target <= 0 or target >= sample.nyquist_hz:
            continue
        mask = np.abs(freqs - target) <= tolerance_hz
        out[f"envelope_{name}"] = float(np.sum(env[mask] ** 2)) if mask.any() else 0.0
    return out


def extract_features(sample: VibrationSample) -> dict:
    """Extract a feature vector from a vibration sample for AI analysis.

    Frequency bands above Nyquist are reported as ``None`` rather than ``0.0``:
    a band that was never observable is not the same as a band with no energy,
    and the difference decides whether a bearing verdict is trustworthy.
    """
    freqs, fft_mag = compute_fft(sample)
    nyquist = sample.nyquist_hz

    band_energy: dict[str, float | None] = {}
    partial_bands: list[str] = []
    for low, high in FEATURE_BANDS:
        key = f"energy_{low}_{high}hz"
        if low >= nyquist:
            # Entirely above Nyquist: never observed.
            band_energy[key] = None
            continue
        if high > nyquist:
            # Only the lower slice was observed; the value is real but it
            # under-reports the band, so callers are told not to read it as a
            # complete measurement.
            partial_bands.append(key)
        mask = (freqs >= low) & (freqs < min(high, nyquist))
        band_energy[key] = float(np.sum(fft_mag[mask] ** 2))

    if len(fft_mag) > 1:
        dominant_idx = int(np.argmax(fft_mag[1:])) + 1  # skip DC
        dominant_freq = float(freqs[dominant_idx])
    else:
        dominant_freq = 0.0

    features: dict = {
        "rms": sample.rms,
        "peak": sample.peak,
        "crest_factor": sample.crest_factor,
        "kurtosis": sample.kurtosis,
        "dominant_frequency_hz": dominant_freq,
        "sample_rate_hz": float(sample.sample_rate),
        "bearing_analysis_available": sample.supports_bearing_analysis,
        "partial_bands": partial_bands,
        **band_energy,
    }
    features.update(envelope_defect_energy(sample))
    return features
