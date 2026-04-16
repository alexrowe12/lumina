from __future__ import annotations

from bpm_detector.audio import load_audio
from bpm_detector.sections import detect_sections
from bpm_detector.tempo import detect_tempo


def test_detect_sections_returns_full_analysis_result(structured_sections_factory) -> None:
    wav_path = structured_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 110.0, 1760.0, 1760.0, 1760.0, 1760.0, 1760.0],
        bpm=120,
    )

    audio = load_audio(wav_path)
    tempo_result = detect_tempo(audio)
    analysis = detect_sections(audio, tempo_result)

    assert analysis.tempo is tempo_result
    assert analysis.bar_features.timestamps == tempo_result.bar_timestamps
    assert list(analysis.novelty_curve.timestamps) == tempo_result.bar_timestamps
    assert analysis.section_boundaries
    assert analysis.section_timestamps == [
        boundary.timestamp for boundary in analysis.section_boundaries
    ]
    assert analysis.section_boundaries[-1].timestamp > 0.0


def test_detect_sections_honors_selection_parameters(structured_sections_factory) -> None:
    wav_path = structured_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 110.0, 1760.0, 1760.0, 1760.0, 1760.0, 1760.0],
        bpm=120,
    )

    audio = load_audio(wav_path)
    tempo_result = detect_tempo(audio)
    permissive = detect_sections(
        audio,
        tempo_result,
        min_score=0.05,
        min_spacing_bars=2,
    )
    strict = detect_sections(
        audio,
        tempo_result,
        min_score=0.8,
        min_spacing_bars=6,
        max_boundaries=1,
    )

    assert len(permissive.section_boundaries) >= len(strict.section_boundaries)
    assert len(strict.section_boundaries) == 1
