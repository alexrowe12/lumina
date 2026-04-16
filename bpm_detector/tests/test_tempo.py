from __future__ import annotations

import pytest

from bpm_detector.audio import load_audio
from bpm_detector.tempo import detect_tempo


@pytest.mark.parametrize(
    ("bpm", "duration_seconds"),
    [
        (90, 12),
        (120, 10),
        (174, 10),
    ],
)
def test_detect_tempo_matches_click_track_bpm(
    click_track_factory,
    bpm: int,
    duration_seconds: int,
) -> None:
    wav_path = click_track_factory(bpm=bpm, duration_seconds=duration_seconds)

    result = detect_tempo(load_audio(wav_path))

    assert result.bpm == pytest.approx(bpm, abs=1.0)
    assert result.rounded_bpm == round(result.bpm)


@pytest.mark.parametrize(
    ("bpm", "duration_seconds"),
    [
        (90, 12),
        (120, 10),
    ],
)
def test_detect_tempo_returns_regular_beat_timestamps(
    click_track_factory,
    bpm: int,
    duration_seconds: int,
) -> None:
    wav_path = click_track_factory(bpm=bpm, duration_seconds=duration_seconds)

    result = detect_tempo(load_audio(wav_path))
    expected_interval = 60.0 / bpm

    assert result.beat_timestamps
    assert result.beat_timestamps[0] == pytest.approx(0.0, abs=0.05)

    intervals = [
        current - previous
        for previous, current in zip(result.beat_timestamps, result.beat_timestamps[1:])
    ]

    assert intervals
    for interval in intervals:
        assert interval == pytest.approx(expected_interval, abs=0.08)

    expected_beats = int(duration_seconds / expected_interval)
    assert len(result.beat_timestamps) == pytest.approx(expected_beats, abs=1)


def test_detect_tempo_preserves_original_file_timestamps(click_track_factory) -> None:
    wav_path = click_track_factory(bpm=120, duration_seconds=8, leading_silence_seconds=2.0)

    result = detect_tempo(load_audio(wav_path))

    assert result.beat_timestamps
    assert result.beat_timestamps[0] == pytest.approx(2.0, abs=0.06)


@pytest.mark.parametrize(
    ("bpm", "duration_seconds"),
    [
        (90, 12),
        (120, 12),
    ],
)
def test_detect_tempo_returns_regular_bar_timestamps(
    click_track_factory,
    bpm: int,
    duration_seconds: int,
) -> None:
    wav_path = click_track_factory(bpm=bpm, duration_seconds=duration_seconds)

    result = detect_tempo(load_audio(wav_path))
    expected_bar_interval = (60.0 / bpm) * 4

    assert result.beats_per_bar == 4
    assert result.bar_timestamps
    assert result.bar_timestamps[0] == pytest.approx(0.0, abs=0.05)

    bar_intervals = [
        current - previous
        for previous, current in zip(result.bar_timestamps, result.bar_timestamps[1:])
    ]

    assert bar_intervals
    for interval in bar_intervals:
        assert interval == pytest.approx(expected_bar_interval, abs=0.12)


def test_detect_tempo_preserves_original_bar_timestamps(click_track_factory) -> None:
    wav_path = click_track_factory(bpm=120, duration_seconds=12, leading_silence_seconds=2.0)

    result = detect_tempo(load_audio(wav_path))

    assert result.bar_timestamps
    assert result.bar_timestamps[0] == pytest.approx(2.0, abs=0.06)
