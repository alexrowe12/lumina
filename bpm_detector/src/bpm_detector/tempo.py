"""Tempo estimation logic."""

from __future__ import annotations

from .models import AudioData, TempoResult


class TempoDetectionError(ValueError):
    """Raised when tempo cannot be estimated from the provided audio."""


def detect_tempo(audio: AudioData, min_bpm: int = 60, max_bpm: int = 200) -> TempoResult:
    """Estimate the dominant global tempo for a preprocessed audio signal."""

    if audio.sample_rate <= 0:
        raise TempoDetectionError("Audio sample rate must be positive.")
    if len(audio.samples) < audio.sample_rate:
        raise TempoDetectionError("Audio is too short to estimate BPM reliably.")
    if min_bpm <= 0 or max_bpm <= min_bpm:
        raise TempoDetectionError("Invalid BPM search range.")

    hop_size = _choose_hop_size(audio.sample_rate)
    onset_envelope = compute_onset_envelope(audio.samples, hop_size)
    if len(onset_envelope) < 8:
        raise TempoDetectionError("Audio does not contain enough rhythmic information.")

    bpm = estimate_bpm_from_onset_envelope(
        onset_envelope,
        envelope_rate=audio.sample_rate / hop_size,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    return TempoResult(
        bpm=bpm,
        rounded_bpm=round(bpm),
        beat_timestamps=[],
    )


def compute_onset_envelope(samples: list[float], hop_size: int) -> list[float]:
    """Compute a simple onset envelope from short-window energy changes."""

    if hop_size <= 0:
        raise TempoDetectionError("Hop size must be positive.")
    if not samples:
        return []

    energy: list[float] = []
    for start in range(0, len(samples), hop_size):
        frame = samples[start : start + hop_size]
        if not frame:
            continue
        energy.append(sum(abs(sample) for sample in frame) / len(frame))

    if len(energy) < 2:
        return energy

    onset_envelope = [0.0]
    for index in range(1, len(energy)):
        change = energy[index] - energy[index - 1]
        onset_envelope.append(change if change > 0.0 else 0.0)

    return _normalize_series(_smooth_series(onset_envelope, radius=2))


def estimate_bpm_from_onset_envelope(
    onset_envelope: list[float],
    envelope_rate: float,
    min_bpm: int,
    max_bpm: int,
) -> float:
    """Pick the BPM whose lag has the strongest autocorrelation score."""

    if envelope_rate <= 0:
        raise TempoDetectionError("Envelope rate must be positive.")
    if not onset_envelope:
        raise TempoDetectionError("Onset envelope is empty.")

    centered = _center_series(onset_envelope)
    if not any(abs(value) > 1e-9 for value in centered):
        raise TempoDetectionError("Audio does not contain enough rhythmic variation.")

    min_lag = max(1, int(envelope_rate * 60.0 / max_bpm))
    max_lag = int(envelope_rate * 60.0 / min_bpm)
    max_valid_lag = len(centered) // 2
    if max_valid_lag < min_lag:
        raise TempoDetectionError("Audio is too short for the requested BPM range.")

    max_lag = min(max_lag, max_valid_lag)

    best_lag = min_lag
    best_score = float("-inf")

    for lag in range(min_lag, max_lag + 1):
        score = _autocorrelation(centered, lag) * _tempo_preference_weight(
            bpm=(60.0 * envelope_rate) / lag
        )
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_score <= 0.0:
        raise TempoDetectionError("Unable to find a stable tempo.")

    return (60.0 * envelope_rate) / best_lag


def _choose_hop_size(sample_rate: int) -> int:
    target_hz = 200
    return max(128, sample_rate // target_hz)


def _smooth_series(values: list[float], radius: int) -> list[float]:
    if radius <= 0 or len(values) < 3:
        return values

    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        window = values[start:end]
        smoothed.append(sum(window) / len(window))

    return smoothed


def _normalize_series(values: list[float]) -> list[float]:
    peak = max((abs(value) for value in values), default=0.0)
    if peak <= 0.0:
        return values
    return [value / peak for value in values]


def _center_series(values: list[float]) -> list[float]:
    mean_value = sum(values) / len(values)
    return [value - mean_value for value in values]


def _autocorrelation(values: list[float], lag: int) -> float:
    return sum(values[index] * values[index - lag] for index in range(lag, len(values)))


def _tempo_preference_weight(bpm: float) -> float:
    """Bias toward common dance/pop tempo ranges to reduce octave mistakes."""

    distance_from_center = abs(bpm - 120.0)
    return max(0.6, 1.0 - (distance_from_center / 240.0))
