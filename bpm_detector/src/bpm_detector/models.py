"""Data models used across the package."""

from dataclasses import dataclass


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
