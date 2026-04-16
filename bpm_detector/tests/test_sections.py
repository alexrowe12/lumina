from __future__ import annotations

import numpy as np
import pytest

from bpm_detector.audio import load_audio
from bpm_detector.models import BarFeatures, NoveltyCurve, TempoResult
from bpm_detector.sections import (
    compute_novelty_curve,
    extract_bar_features,
    select_section_boundaries,
)


def test_extract_bar_features_returns_expected_shapes(tone_sections_factory) -> None:
    wav_path = tone_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 1760.0, 1760.0],
        bar_duration_seconds=2.0,
    )
    audio = load_audio(wav_path)
    tempo_result = TempoResult(
        bpm=120.0,
        rounded_bpm=120,
        beat_timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        bar_timestamps=[0.0, 2.0, 4.0, 6.0],
        beats_per_bar=4,
    )

    features = extract_bar_features(audio, tempo_result)

    assert features.timestamps == [0.0, 2.0, 4.0, 6.0]
    assert features.rms_energy.shape == (4,)
    assert features.spectral_centroid.shape == (4,)
    assert features.spectral_rolloff.shape == (4,)
    assert features.band_energies.shape == (4, 5)
    assert features.vectors.shape == (4, 8)


def test_extract_bar_features_detects_spectral_change_between_bars(tone_sections_factory) -> None:
    wav_path = tone_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 1760.0, 1760.0],
        bar_duration_seconds=2.0,
    )
    audio = load_audio(wav_path)
    tempo_result = TempoResult(
        bpm=120.0,
        rounded_bpm=120,
        beat_timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        bar_timestamps=[0.0, 2.0, 4.0, 6.0],
        beats_per_bar=4,
    )

    features = extract_bar_features(audio, tempo_result)

    low_section_centroid = float(np.mean(features.spectral_centroid[:2]))
    high_section_centroid = float(np.mean(features.spectral_centroid[2:]))

    assert high_section_centroid > low_section_centroid
    assert float(np.mean(features.band_energies[:2, 0])) > float(np.mean(features.band_energies[2:, 0]))
    assert float(np.mean(features.band_energies[2:, 3])) > float(np.mean(features.band_energies[:2, 3]))


def test_extract_bar_features_respects_original_timestamps_after_trim(
    tone_sections_factory,
) -> None:
    wav_path = tone_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 1760.0, 1760.0],
        bar_duration_seconds=2.0,
        leading_silence_seconds=1.5,
    )
    audio = load_audio(wav_path)
    tempo_result = TempoResult(
        bpm=120.0,
        rounded_bpm=120,
        beat_timestamps=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
        bar_timestamps=[1.5, 3.5, 5.5, 7.5],
        beats_per_bar=4,
    )

    features = extract_bar_features(audio, tempo_result)

    assert features.timestamps == [1.5, 3.5, 5.5, 7.5]
    assert float(np.mean(features.spectral_centroid[2:])) > float(np.mean(features.spectral_centroid[:2]))


def test_compute_novelty_curve_peaks_at_section_change(tone_sections_factory) -> None:
    wav_path = tone_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 110.0, 1760.0, 1760.0, 1760.0],
        bar_duration_seconds=2.0,
    )
    audio = load_audio(wav_path)
    tempo_result = TempoResult(
        bpm=120.0,
        rounded_bpm=120,
        beat_timestamps=[
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
        ],
        bar_timestamps=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        beats_per_bar=4,
    )

    features = extract_bar_features(audio, tempo_result)
    novelty = compute_novelty_curve(features, window_bars=2)

    peak_index = int(np.argmax(novelty.scores))
    assert novelty.timestamps[peak_index] == pytest.approx(6.0, abs=0.01)
    assert novelty.scores[peak_index] == pytest.approx(1.0, abs=1e-9)
    assert novelty.scores[peak_index] > novelty.scores[1]


def test_compute_novelty_curve_stays_near_zero_for_uniform_bars() -> None:
    features = BarFeatures(
        timestamps=[0.0, 2.0, 4.0, 6.0],
        rms_energy=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        spectral_centroid=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
        spectral_rolloff=np.array([0.75, 0.75, 0.75, 0.75], dtype=np.float64),
        band_energies=np.array(
            [
                [0.4, 0.2, 0.2, 0.1, 0.1],
                [0.4, 0.2, 0.2, 0.1, 0.1],
                [0.4, 0.2, 0.2, 0.1, 0.1],
                [0.4, 0.2, 0.2, 0.1, 0.1],
            ],
            dtype=np.float64,
        ),
    )

    novelty = compute_novelty_curve(features, window_bars=2)

    assert novelty.timestamps == [0.0, 2.0, 4.0, 6.0]
    assert np.allclose(novelty.scores, 0.0)


def test_compute_novelty_curve_requires_positive_window() -> None:
    features = BarFeatures(
        timestamps=[0.0],
        rms_energy=np.array([1.0], dtype=np.float64),
        spectral_centroid=np.array([1.0], dtype=np.float64),
        spectral_rolloff=np.array([1.0], dtype=np.float64),
        band_energies=np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="window_bars must be positive"):
        compute_novelty_curve(features, window_bars=0)


def test_extract_bar_features_validates_band_edges(tone_sections_factory) -> None:
    wav_path = tone_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 1760.0, 1760.0],
        bar_duration_seconds=2.0,
    )
    audio = load_audio(wav_path)
    tempo_result = TempoResult(
        bpm=120.0,
        rounded_bpm=120,
        beat_timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        bar_timestamps=[0.0, 2.0, 4.0, 6.0],
        beats_per_bar=4,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        extract_bar_features(audio, tempo_result, band_edges_hz=(400.0, 150.0))

    with pytest.raises(ValueError, match="positive values"):
        extract_bar_features(audio, tempo_result, band_edges_hz=(150.0, 0.0, 400.0))


def test_select_section_boundaries_returns_transition_timestamp(tone_sections_factory) -> None:
    wav_path = tone_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 110.0, 1760.0, 1760.0, 1760.0],
        bar_duration_seconds=2.0,
    )
    audio = load_audio(wav_path)
    tempo_result = TempoResult(
        bpm=120.0,
        rounded_bpm=120,
        beat_timestamps=[
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
        ],
        bar_timestamps=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        beats_per_bar=4,
    )

    features = extract_bar_features(audio, tempo_result)
    novelty = compute_novelty_curve(features, window_bars=2)
    boundaries = select_section_boundaries(novelty, min_score=0.4, min_spacing_bars=3)

    assert len(boundaries) == 1
    assert boundaries[0].timestamp == pytest.approx(6.0, abs=0.01)
    assert boundaries[0].bar_index == 3
    assert boundaries[0].weighted_score >= boundaries[0].raw_score


def test_select_section_boundaries_prefers_phrase_aligned_peak() -> None:
    novelty = NoveltyCurve(
        timestamps=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        scores=np.array([0.0, 0.0, 0.72, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0], dtype=np.float64),
    )

    boundaries = select_section_boundaries(
        novelty,
        min_score=0.5,
        min_spacing_bars=8,
        max_boundaries=1,
    )

    assert len(boundaries) == 1
    assert boundaries[0].bar_index == 2
    assert boundaries[0].timestamp == pytest.approx(4.0, abs=0.01)


def test_select_section_boundaries_enforces_spacing_by_weighted_priority() -> None:
    novelty = NoveltyCurve(
        timestamps=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        scores=np.array([0.0, 0.55, 0.0, 0.78, 0.0, 0.9, 0.0], dtype=np.float64),
    )

    boundaries = select_section_boundaries(
        novelty,
        min_score=0.5,
        min_spacing_bars=3,
    )

    assert [boundary.bar_index for boundary in boundaries] == [1, 5]
    assert boundaries[0].timestamp == pytest.approx(2.0, abs=0.01)
    assert boundaries[1].timestamp == pytest.approx(10.0, abs=0.01)


def test_select_section_boundaries_validates_arguments() -> None:
    novelty = NoveltyCurve(
        timestamps=[0.0],
        scores=np.array([0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="min_score must be between 0.0 and 1.0"):
        select_section_boundaries(novelty, min_score=1.2)

    with pytest.raises(ValueError, match="min_spacing_bars must be at least 1"):
        select_section_boundaries(novelty, min_spacing_bars=0)

    with pytest.raises(ValueError, match="max_boundaries must be positive"):
        select_section_boundaries(novelty, max_boundaries=0)


def test_select_section_boundaries_validates_curve_lengths() -> None:
    novelty = NoveltyCurve(
        timestamps=[0.0, 2.0],
        scores=np.array([0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="same length"):
        select_section_boundaries(novelty)
