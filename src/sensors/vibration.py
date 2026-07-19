"""Vibration data acquisition and signal processing."""

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from dataclasses import dataclass
from enum import Enum


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
    """A single vibration measurement from a machine station."""
    station_id: str
    timestamp: float
    raw_signal: NDArray[np.float32]
    sample_rate: int
    duration: float

    @property
    def rms(self) -> float:
        """Root mean square of the signal."""
        return float(np.sqrt(np.mean(self.raw_signal ** 2)))

    @property
    def peak(self) -> float:
        """Peak amplitude."""
        return float(np.max(np.abs(self.raw_signal)))

    @property
    def crest_factor(self) -> float:
        """Peak / RMS — high values indicate impulsive faults."""
        rms = self.rms
        if rms == 0:
            return 0.0
        return self.peak / rms

    @property
    def kurtosis(self) -> float:
        """Statistical kurtosis — elevated in bearing faults."""
        from scipy.stats import kurtosis as sp_kurtosis
        return float(sp_kurtosis(self.raw_signal))


def compute_fft(sample: VibrationSample) -> tuple[NDArray, NDArray]:
    """Compute single-sided FFT magnitude spectrum."""
    n = len(sample.raw_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample.sample_rate)
    fft_mag = np.abs(np.fft.rfft(sample.raw_signal)) * 2.0 / n
    return freqs, fft_mag


def compute_spectrogram(
    sample: VibrationSample,
    nperseg: int = 256,
    noverlap: int = 128,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute spectrogram for time-frequency analysis."""
    freqs, times, sxx = signal.spectrogram(
        sample.raw_signal,
        fs=sample.sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return freqs, times, sxx


def compute_envelope_spectrum(sample: VibrationSample) -> tuple[NDArray, NDArray]:
    """Envelope analysis — key technique for bearing fault detection.

    Demodulates the signal to extract repetitive impact patterns
    that correspond to bearing defect frequencies.
    """
    # Bandpass filter to isolate resonance band
    sos = signal.butter(
        4,
        [1000, sample.sample_rate / 2 - 100],
        btype="bandpass",
        fs=sample.sample_rate,
        output="sos",
    )
    filtered = signal.sosfilt(sos, sample.raw_signal)

    # Hilbert transform for envelope
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    # FFT of envelope
    n = len(envelope)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample.sample_rate)
    env_fft = np.abs(np.fft.rfft(envelope)) * 2.0 / n

    return freqs, env_fft


def extract_features(sample: VibrationSample) -> dict:
    """Extract a feature vector from a vibration sample for AI analysis."""
    freqs, fft_mag = compute_fft(sample)

    # Frequency band energies
    bands = [(0, 100), (100, 500), (500, 1000), (1000, 2000)]
    band_energy = {}
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_energy[f"energy_{low}_{high}hz"] = float(np.sum(fft_mag[mask] ** 2))

    # Dominant frequency
    dominant_idx = np.argmax(fft_mag[1:]) + 1  # skip DC
    dominant_freq = float(freqs[dominant_idx])

    return {
        "rms": sample.rms,
        "peak": sample.peak,
        "crest_factor": sample.crest_factor,
        "kurtosis": sample.kurtosis,
        "dominant_frequency_hz": dominant_freq,
        **band_energy,
    }
