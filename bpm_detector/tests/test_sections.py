from __future__ import annotations

import numpy as np

from bpm_detector.audio import load_audio
from bpm_detector.models import TempoResult
from bpm_detector.sections import extract_bar_features


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
