"""Helpers for constructing beat-aligned musical grids."""

from __future__ import annotations


def build_bar_timestamps(
    beat_timestamps: list[float],
    beats_per_bar: int = 4,
) -> list[float]:
    """Return the timestamp of each bar start, assuming a fixed beats-per-bar grid."""

    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be positive.")
    if not beat_timestamps:
        return []

    return beat_timestamps[::beats_per_bar]
