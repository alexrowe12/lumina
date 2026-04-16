"""Data models used across the package."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AudioData:
    """Normalized mono audio ready for tempo analysis."""

    sample_rate: int
    samples: list[float]
    duration_seconds: float
    start_offset_seconds: float


@dataclass(slots=True)
class TempoResult:
    """Represents a BPM detection result."""

    bpm: float
    rounded_bpm: int
    beat_timestamps: list[float]
    bar_timestamps: list[float]
    beats_per_bar: int


@dataclass(slots=True)
class BarFeatures:
    """Per-bar audio features for downstream section analysis."""

    timestamps: list[float]
    rms_energy: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    band_energies: np.ndarray

    @property
    def vectors(self) -> np.ndarray:
        """Return a stacked feature matrix with one row per bar."""

        return np.column_stack(
            (
                self.rms_energy,
                self.spectral_centroid,
                self.spectral_rolloff,
                self.band_energies,
            )
        )


@dataclass(slots=True)
class NoveltyCurve:
    """Bar-aligned structural change scores."""

    timestamps: list[float]
    scores: np.ndarray


@dataclass(slots=True)
class SectionBoundary:
    """A selected structural boundary candidate."""

    timestamp: float
    bar_index: int
    raw_score: float
    weighted_score: float
