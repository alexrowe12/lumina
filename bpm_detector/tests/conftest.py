from __future__ import annotations

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
