from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest


def write_click_track(
    path: Path,
    bpm: int,
    duration_seconds: int,
    sample_rate: int = 44_100,
    click_samples: int = 1_200,
    click_amplitude: int = 22_000,
    leading_silence_seconds: float = 0.0,
) -> None:
    interval = int(sample_rate * 60 / bpm)
    total_samples = sample_rate * duration_seconds
    start_sample = int(sample_rate * leading_silence_seconds)
    samples = [0] * total_samples

    for start in range(start_sample, total_samples, interval):
        for offset in range(min(click_samples, total_samples - start)):
            samples[start + offset] = click_amplitude if offset < 250 else 0

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def write_tone_sections(
    path: Path,
    section_frequencies_hz: list[float],
    bar_duration_seconds: float,
    sample_rate: int = 44_100,
    amplitude: int = 14_000,
    leading_silence_seconds: float = 0.0,
) -> None:
    total_tone_samples = int(round(len(section_frequencies_hz) * bar_duration_seconds * sample_rate))
    leading_silence_samples = int(round(leading_silence_seconds * sample_rate))
    total_samples = leading_silence_samples + total_tone_samples
    samples = [0] * total_samples

    for bar_index, frequency_hz in enumerate(section_frequencies_hz):
        start_sample = leading_silence_samples + int(round(bar_index * bar_duration_seconds * sample_rate))
        end_sample = leading_silence_samples + int(round((bar_index + 1) * bar_duration_seconds * sample_rate))

        for sample_index in range(start_sample, min(end_sample, total_samples)):
            phase = 2.0 * math.pi * frequency_hz * ((sample_index - leading_silence_samples) / sample_rate)
            samples[sample_index] = int(amplitude * math.sin(phase))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def write_structured_click_sections(
    path: Path,
    section_frequencies_hz: list[float],
    bpm: int,
    beats_per_bar: int = 4,
    sample_rate: int = 44_100,
    tone_amplitude: int = 10_000,
    click_amplitude: int = 22_000,
    click_samples: int = 1_200,
) -> None:
    beat_interval_samples = int(round(sample_rate * 60 / bpm))
    bar_duration_seconds = (60.0 / bpm) * beats_per_bar
    total_samples = int(round(len(section_frequencies_hz) * bar_duration_seconds * sample_rate))
    samples = [0] * total_samples

    for bar_index, frequency_hz in enumerate(section_frequencies_hz):
        start_sample = int(round(bar_index * bar_duration_seconds * sample_rate))
        end_sample = int(round((bar_index + 1) * bar_duration_seconds * sample_rate))

        for sample_index in range(start_sample, min(end_sample, total_samples)):
            phase = 2.0 * math.pi * frequency_hz * (sample_index / sample_rate)
            samples[sample_index] += int(tone_amplitude * math.sin(phase))

    for start in range(0, total_samples, beat_interval_samples):
        for offset in range(min(click_samples, total_samples - start)):
            if offset < 250:
                samples[start + offset] += click_amplitude

    clipped_samples = [max(-32768, min(32767, sample)) for sample in samples]

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in clipped_samples))


@pytest.fixture
def click_track_factory(tmp_path: Path):
    def factory(*, bpm: int, duration_seconds: int, leading_silence_seconds: float = 0.0) -> Path:
        silence_label = str(leading_silence_seconds).replace(".", "_")
        path = tmp_path / f"click_{bpm}_{duration_seconds}s_silence_{silence_label}.wav"
        write_click_track(
            path=path,
            bpm=bpm,
            duration_seconds=duration_seconds,
            leading_silence_seconds=leading_silence_seconds,
        )
        return path

    return factory


@pytest.fixture
def tone_sections_factory(tmp_path: Path):
    def factory(
        *,
        section_frequencies_hz: list[float],
        bar_duration_seconds: float,
        leading_silence_seconds: float = 0.0,
    ) -> Path:
        freq_label = "_".join(str(int(frequency)) for frequency in section_frequencies_hz)
        silence_label = str(leading_silence_seconds).replace(".", "_")
        path = tmp_path / f"tones_{freq_label}_bar_{bar_duration_seconds}_silence_{silence_label}.wav"
        write_tone_sections(
            path=path,
            section_frequencies_hz=section_frequencies_hz,
            bar_duration_seconds=bar_duration_seconds,
            leading_silence_seconds=leading_silence_seconds,
        )
        return path

    return factory


@pytest.fixture
def structured_sections_factory(tmp_path: Path):
    def factory(
        *,
        section_frequencies_hz: list[float],
        bpm: int,
        beats_per_bar: int = 4,
    ) -> Path:
        freq_label = "_".join(str(int(frequency)) for frequency in section_frequencies_hz)
        path = tmp_path / f"structured_{bpm}_{freq_label}.wav"
        write_structured_click_sections(
            path=path,
            section_frequencies_hz=section_frequencies_hz,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
        )
        return path

    return factory
