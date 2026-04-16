"""Audio loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
import wave

from .models import AudioData


class AudioLoadError(ValueError):
    """Raised when a WAV file cannot be parsed into supported audio samples."""


def load_audio(path: str | Path) -> AudioData:
    """Load a WAV file, mix it to mono, and trim quiet edges."""

    wav_path = Path(path)
    _validate_wav_path(wav_path)

    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw_frames = wav_file.readframes(frame_count)

    interleaved_samples = _decode_pcm_frames(raw_frames, sample_width)
    mono_samples = _mix_to_mono(interleaved_samples, channel_count)
    trimmed_samples, start_sample_index = _trim_silence_with_offset(mono_samples)
    duration_seconds = len(trimmed_samples) / sample_rate if sample_rate else 0.0
    start_offset_seconds = start_sample_index / sample_rate if sample_rate else 0.0

    return AudioData(
        sample_rate=sample_rate,
        samples=trimmed_samples,
        duration_seconds=duration_seconds,
        start_offset_seconds=start_offset_seconds,
    )


def trim_silence(samples: list[float], threshold: float = 0.01) -> list[float]:
    """Remove leading and trailing samples below a simple absolute-amplitude threshold."""

    trimmed_samples, _ = _trim_silence_with_offset(samples, threshold=threshold)
    return trimmed_samples


def _trim_silence_with_offset(
    samples: list[float],
    threshold: float = 0.01,
) -> tuple[list[float], int]:
    """Trim quiet edges and return both the trimmed audio and the start sample offset."""

    if not samples:
        return [], 0

    start_index = 0
    end_index = len(samples) - 1

    while start_index <= end_index and abs(samples[start_index]) < threshold:
        start_index += 1

    while end_index >= start_index and abs(samples[end_index]) < threshold:
        end_index -= 1

    if start_index > end_index:
        return [], 0

    return samples[start_index : end_index + 1], start_index


def _validate_wav_path(path: Path) -> None:
    if path.suffix.lower() != ".wav":
        raise AudioLoadError(f"Expected a .wav file, got: {path.name}")
    if not path.exists():
        raise AudioLoadError(f"WAV file does not exist: {path}")
    if not path.is_file():
        raise AudioLoadError(f"Expected a file path, got: {path}")


def _decode_pcm_frames(raw_frames: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(_byte - 128) / 128.0 for _byte in raw_frames]
    if sample_width == 2:
        return [
            int.from_bytes(raw_frames[index : index + 2], byteorder="little", signed=True) / 32768.0
            for index in range(0, len(raw_frames), 2)
        ]
    if sample_width == 3:
        return [_decode_24_bit_sample(raw_frames, index) for index in range(0, len(raw_frames), 3)]
    if sample_width == 4:
        return [
            int.from_bytes(raw_frames[index : index + 4], byteorder="little", signed=True) / 2147483648.0
            for index in range(0, len(raw_frames), 4)
        ]

    raise AudioLoadError(f"Unsupported WAV sample width: {sample_width} bytes")


def _decode_24_bit_sample(raw_frames: bytes, start_index: int) -> float:
    chunk = raw_frames[start_index : start_index + 3]
    if len(chunk) != 3:
        raise AudioLoadError("Malformed 24-bit WAV frame data")

    sign_extension = b"\xff" if chunk[2] & 0x80 else b"\x00"
    sample = int.from_bytes(chunk + sign_extension, byteorder="little", signed=True)
    return sample / 8388608.0


def _mix_to_mono(samples: list[float], channel_count: int) -> list[float]:
    if channel_count < 1:
        raise AudioLoadError(f"Invalid channel count: {channel_count}")
    if channel_count == 1:
        return samples
    if len(samples) % channel_count != 0:
        raise AudioLoadError("Malformed WAV data: sample count does not match channel count")

    mono_samples: list[float] = []
    for index in range(0, len(samples), channel_count):
        frame = samples[index : index + channel_count]
        mono_samples.append(sum(frame) / channel_count)

    return mono_samples
