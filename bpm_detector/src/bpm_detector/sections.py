"""Section-analysis primitives built on top of the tempo detector."""

from __future__ import annotations

import numpy as np

from .models import AudioData, BarFeatures, TempoResult


def extract_bar_features(
    audio: AudioData,
    tempo_result: TempoResult,
    band_edges_hz: tuple[float, ...] = (150.0, 400.0, 1_000.0, 2_500.0),
) -> BarFeatures:
    """Compute beat-grid-aligned per-bar features for structural analysis."""

    if audio.sample_rate <= 0:
        raise ValueError("Audio sample rate must be positive.")
    if not tempo_result.bar_timestamps:
        raise ValueError("Tempo result does not contain any bar timestamps.")

    sample_rate = audio.sample_rate
    samples = np.asarray(audio.samples, dtype=np.float64)
    bar_starts = tempo_result.bar_timestamps
    beat_timestamps = tempo_result.beat_timestamps
    band_count = len(band_edges_hz) + 1

    rms_energy = np.zeros(len(bar_starts), dtype=np.float64)
    spectral_centroid = np.zeros(len(bar_starts), dtype=np.float64)
    spectral_rolloff = np.zeros(len(bar_starts), dtype=np.float64)
    band_energies = np.zeros((len(bar_starts), band_count), dtype=np.float64)

    for bar_index, bar_start in enumerate(bar_starts):
        bar_end = _resolve_bar_end_timestamp(
            bar_index=bar_index,
            bar_starts=bar_starts,
            beat_timestamps=beat_timestamps,
            beats_per_bar=tempo_result.beats_per_bar,
            fallback_end_seconds=audio.start_offset_seconds + audio.duration_seconds,
        )
        bar_samples = _slice_audio_segment(
            samples=samples,
            sample_rate=sample_rate,
            start_seconds=bar_start,
            end_seconds=bar_end,
            start_offset_seconds=audio.start_offset_seconds,
        )

        if bar_samples.size == 0:
            continue

        spectrum = np.abs(np.fft.rfft(bar_samples * np.hanning(len(bar_samples))))
        frequencies = np.fft.rfftfreq(len(bar_samples), d=1.0 / sample_rate)

        rms_energy[bar_index] = np.sqrt(np.mean(np.square(bar_samples)))
        spectral_centroid[bar_index] = _compute_spectral_centroid(frequencies, spectrum)
        spectral_rolloff[bar_index] = _compute_spectral_rolloff(frequencies, spectrum)
        band_energies[bar_index] = _compute_band_energies(
            frequencies=frequencies,
            magnitude_spectrum=spectrum,
            band_edges_hz=band_edges_hz,
        )

    return BarFeatures(
        timestamps=bar_starts,
        rms_energy=_normalize_feature(rms_energy),
        spectral_centroid=_normalize_feature(spectral_centroid),
        spectral_rolloff=_normalize_feature(spectral_rolloff),
        band_energies=_normalize_feature_matrix_rows(band_energies),
    )


def _resolve_bar_end_timestamp(
    bar_index: int,
    bar_starts: list[float],
    beat_timestamps: list[float],
    beats_per_bar: int,
    fallback_end_seconds: float,
) -> float:
    if bar_index + 1 < len(bar_starts):
        return bar_starts[bar_index + 1]

    beat_index = bar_index * beats_per_bar
    if beat_index + beats_per_bar < len(beat_timestamps):
        return beat_timestamps[beat_index + beats_per_bar]

    beat_intervals = [
        current - previous
        for previous, current in zip(beat_timestamps, beat_timestamps[1:])
        if current > previous
    ]
    if beat_intervals:
        mean_beat_interval = sum(beat_intervals) / len(beat_intervals)
        return bar_starts[bar_index] + (mean_beat_interval * beats_per_bar)

    return fallback_end_seconds


def _slice_audio_segment(
    samples: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    start_offset_seconds: float,
) -> np.ndarray:
    relative_start = max(0.0, start_seconds - start_offset_seconds)
    relative_end = max(relative_start, end_seconds - start_offset_seconds)

    start_index = min(len(samples), int(round(relative_start * sample_rate)))
    end_index = min(len(samples), int(round(relative_end * sample_rate)))
    return samples[start_index:end_index]


def _compute_spectral_centroid(frequencies: np.ndarray, magnitude_spectrum: np.ndarray) -> float:
    spectral_mass = magnitude_spectrum.sum()
    if spectral_mass <= 0.0:
        return 0.0
    return float(np.dot(frequencies, magnitude_spectrum) / spectral_mass)


def _compute_spectral_rolloff(frequencies: np.ndarray, magnitude_spectrum: np.ndarray) -> float:
    spectral_mass = magnitude_spectrum.sum()
    if spectral_mass <= 0.0:
        return 0.0

    cumulative = np.cumsum(magnitude_spectrum)
    cutoff = spectral_mass * 0.85
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(index, len(frequencies) - 1)
    return float(frequencies[index])


def _compute_band_energies(
    frequencies: np.ndarray,
    magnitude_spectrum: np.ndarray,
    band_edges_hz: tuple[float, ...],
) -> np.ndarray:
    band_energies = np.zeros(len(band_edges_hz) + 1, dtype=np.float64)
    previous_edge = 0.0

    for band_index, edge in enumerate(band_edges_hz):
        mask = (frequencies >= previous_edge) & (frequencies < edge)
        band_energies[band_index] = float(magnitude_spectrum[mask].sum())
        previous_edge = edge

    band_energies[-1] = float(magnitude_spectrum[frequencies >= previous_edge].sum())
    total_energy = band_energies.sum()
    if total_energy <= 0.0:
        return band_energies

    return band_energies / total_energy


def _normalize_feature(values: np.ndarray) -> np.ndarray:
    peak = float(values.max(initial=0.0))
    if peak <= 0.0:
        return values
    return values / peak


def _normalize_feature_matrix_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values

    row_sums = values.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return values / safe_row_sums
