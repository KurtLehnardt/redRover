"""Acoustic emission analysis — high-frequency audio for leak and friction detection.

Targets:
- Pneumatic air leaks (20-40 kHz ultrasonic)
- Pressurized gas leaks (hissing patterns)
- Metal-on-metal friction (broadband high-frequency)
- Electrical arcing (crackling, intermittent bursts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy import signal

# Ultrasonic leak signatures live at 20-48 kHz, so the capture must run at
# >= 96 kHz.  Below that the band is not attenuated — it is absent.
ULTRASONIC_MIN_SAMPLE_RATE_HZ = 96000
ULTRASONIC_BAND_LOW_HZ = 20000


class AcousticFaultType(str, Enum):
    NORMAL = "normal"
    AIR_LEAK = "air_leak"
    GAS_LEAK = "gas_leak"
    FRICTION = "metal_friction"
    ARCING = "electrical_arcing"


@dataclass
class AcousticSample:
    """A single acoustic emission measurement."""

    station_id: str
    timestamp: float
    raw_signal: NDArray[np.float32]
    sample_rate: int  # Typically 96kHz+ for ultrasonic
    duration: float

    _stats: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        sig = np.asarray(self.raw_signal, dtype=np.float64)
        self._stats = {
            "rms": float(np.sqrt(np.mean(sig ** 2))) if sig.size else 0.0,
            "peak": float(np.max(np.abs(sig))) if sig.size else 0.0,
            "ultrasonic_energy": self._compute_ultrasonic(sig),
        }

    def _compute_ultrasonic(self, sig: NDArray) -> float | None:
        """Energy in the 20-48kHz band, or None when unobservable at this rate."""
        if self.sample_rate < ULTRASONIC_MIN_SAMPLE_RATE_HZ or sig.size == 0:
            return None
        nyquist = self.sample_rate / 2
        high_cutoff = min(44000, nyquist - 1000)
        sos = signal.butter(
            4, [ULTRASONIC_BAND_LOW_HZ, high_cutoff],
            btype="bandpass", fs=self.sample_rate, output="sos",
        )
        filtered = signal.sosfilt(sos, sig)
        return float(np.sqrt(np.mean(filtered ** 2)))

    @property
    def supports_ultrasonic(self) -> bool:
        return self.sample_rate >= ULTRASONIC_MIN_SAMPLE_RATE_HZ

    @property
    def rms(self) -> float:
        return self._stats["rms"]

    @property
    def peak(self) -> float:
        return self._stats["peak"]

    @property
    def ultrasonic_energy(self) -> float | None:
        """Ultrasonic band energy, or ``None`` if the capture cannot see it."""
        return self._stats["ultrasonic_energy"]


def compute_mel_spectrogram(
    sample: AcousticSample,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NDArray:
    """Compute mel spectrogram for audio classification.

    Returns a 2D array suitable for CNN input.
    """
    # Manual mel spectrogram (avoids librosa dependency for real-time)
    nperseg = min(n_fft, len(sample.raw_signal)) or 1
    noverlap = max(0, min(nperseg - 1, nperseg - hop_length))
    freqs, times, sxx = signal.spectrogram(
        sample.raw_signal,
        fs=sample.sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    mel_filters = _mel_filterbank(n_mels, nperseg, sample.sample_rate)
    mel_spec = mel_filters @ sxx
    return np.log1p(mel_spec * 1000)


def extract_acoustic_features(sample: AcousticSample) -> dict:
    """Extract features for acoustic fault classification.

    Bands above Nyquist are reported as ``None`` so a rule engine can tell
    "no ultrasonic energy" from "no ultrasonic microphone".
    """
    n = len(sample.raw_signal)
    if n == 0:
        return {
            "rms": 0.0, "peak": 0.0, "ultrasonic_energy": None,
            "rms_variance": 0.0, "rms_std": 0.0,
            "supports_ultrasonic": sample.supports_ultrasonic,
            "partial_bands": [],
        }

    freqs = np.fft.rfftfreq(n, d=1.0 / sample.sample_rate)
    fft_mag = np.abs(np.fft.rfft(sample.raw_signal)) * 2.0 / n
    nyquist = sample.sample_rate / 2

    bands = {
        "audible_low": (20, 2000),
        "audible_mid": (2000, 8000),
        "audible_high": (8000, 20000),
        "ultrasonic_low": (20000, 30000),
        "ultrasonic_mid": (30000, 40000),
        "ultrasonic_high": (40000, 48000),
    }

    band_energy: dict[str, float | None] = {}
    partial_bands: list[str] = []
    for name, (low, high) in bands.items():
        key = f"acoustic_{name}"
        if low >= nyquist:
            band_energy[key] = None  # entirely above Nyquist: never observed
            continue
        if high > nyquist:
            partial_bands.append(key)  # observed only in part; under-reports
        mask = (freqs >= low) & (freqs < min(high, nyquist))
        band_energy[key] = float(np.sum(fft_mag[mask] ** 2))

    # Temporal features — leaks are steady-state, arcing is intermittent.
    frame_size = max(1, sample.sample_rate // 10)  # 100ms frames
    n_frames = n // frame_size
    if n_frames > 0:
        frames = sample.raw_signal[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    else:
        frame_rms = np.array([sample.rms])

    return {
        "rms": sample.rms,
        "peak": sample.peak,
        "ultrasonic_energy": sample.ultrasonic_energy,
        "supports_ultrasonic": sample.supports_ultrasonic,
        "rms_variance": float(np.var(frame_rms)),  # High = intermittent (arcing)
        "rms_std": float(np.std(frame_rms)),
        "partial_bands": partial_bands,
        **band_energy,
    }


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> NDArray:
    """Create a mel-scale filterbank matrix of shape (n_mels, n_fft//2 + 1)."""
    low_freq = 0
    high_freq = sample_rate / 2

    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)

    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    n_freqs = n_fft // 2 + 1
    # Clamp so the topmost filters cannot index past the spectrum.
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)

    filters = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            filters[i, j] = (right - j) / (right - center)

    return filters
