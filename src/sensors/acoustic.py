"""Acoustic emission analysis — high-frequency audio for leak and friction detection.

Targets:
- Pneumatic air leaks (20-40 kHz ultrasonic)
- Pressurized gas leaks (hissing patterns)
- Metal-on-metal friction (broadband high-frequency)
- Electrical arcing (crackling, intermittent bursts)
"""

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from dataclasses import dataclass
from enum import Enum


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

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.raw_signal ** 2)))

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.raw_signal)))

    @property
    def ultrasonic_energy(self) -> float:
        """Energy in the 20-48kHz band (ultrasonic leak indicator)."""
        if self.sample_rate < 96000:
            return 0.0
        nyquist = self.sample_rate / 2
        high_cutoff = min(44000, nyquist - 1000)
        sos = signal.butter(4, [20000, high_cutoff], btype="bandpass", fs=self.sample_rate, output="sos")
        filtered = signal.sosfilt(sos, self.raw_signal)
        return float(np.sqrt(np.mean(filtered ** 2)))


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
    freqs, times, sxx = signal.spectrogram(
        sample.raw_signal,
        fs=sample.sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )

    # Create mel filter bank
    mel_filters = _mel_filterbank(n_mels, n_fft, sample.sample_rate)

    # Apply mel filters
    mel_spec = mel_filters @ sxx[:mel_filters.shape[1], :]

    # Log scale
    mel_spec = np.log1p(mel_spec * 1000)

    return mel_spec


def extract_acoustic_features(sample: AcousticSample) -> dict:
    """Extract features for acoustic fault classification."""
    n = len(sample.raw_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample.sample_rate)
    fft_mag = np.abs(np.fft.rfft(sample.raw_signal)) * 2.0 / n

    # Band energies for acoustic classification
    bands = {
        "audible_low": (20, 2000),
        "audible_mid": (2000, 8000),
        "audible_high": (8000, 20000),
    }

    # Add ultrasonic bands if sample rate supports it
    if sample.sample_rate >= 96000:
        bands["ultrasonic_low"] = (20000, 30000)
        bands["ultrasonic_mid"] = (30000, 40000)
        bands["ultrasonic_high"] = (40000, 48000)

    band_energy = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy[f"acoustic_{name}"] = float(np.sum(fft_mag[mask] ** 2))

    # Temporal features
    # Leak detection: steady-state high-freq energy
    # Arcing: intermittent bursts
    frame_size = sample.sample_rate // 10  # 100ms frames
    n_frames = n // frame_size
    frame_rms = np.array([
        np.sqrt(np.mean(sample.raw_signal[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ])

    return {
        "rms": sample.rms,
        "peak": sample.peak,
        "ultrasonic_energy": sample.ultrasonic_energy,
        "rms_variance": float(np.var(frame_rms)),  # High = intermittent (arcing)
        "rms_std": float(np.std(frame_rms)),
        **band_energy,
    }


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> NDArray:
    """Create a mel-scale filterbank matrix."""
    low_freq = 0
    high_freq = sample_rate / 2

    # Mel scale conversion
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)

    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_freqs = n_fft // 2 + 1
    filters = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center != left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filters[i, j] = (right - j) / (right - center)

    return filters
