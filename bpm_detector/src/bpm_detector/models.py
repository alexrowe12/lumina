"""Data models used across the package."""

from dataclasses import dataclass


@dataclass(slots=True)
class TempoResult:
    """Represents a BPM detection result."""

    bpm: float
    rounded_bpm: int
    beat_timestamps: list[float]

