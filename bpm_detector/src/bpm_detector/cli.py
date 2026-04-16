"""Command-line interface for bpm_detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio import AudioLoadError, load_audio
from .models import TempoResult
from .tempo import TempoDetectionError, detect_tempo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpm-detector",
        description="Detect BPM and beat timestamps from WAV files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser(
        "detect",
        help="Analyze a WAV file and print BPM information.",
    )
    detect_parser.add_argument("path", type=Path, help="Path to a WAV file.")
    detect_parser.add_argument(
        "--show-beats",
        action="store_true",
        help="Print detected beat timestamps in raw seconds.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "detect":
        try:
            audio = load_audio(args.path)
        except AudioLoadError as exc:
            parser.exit(status=2, message=f"error: {exc}\n")

        try:
            result = detect_tempo(audio)
        except TempoDetectionError as exc:
            parser.exit(status=2, message=f"error: {exc}\n")

        _print_summary(result)
        if args.show_beats:
            _print_beat_timestamps(result.beat_timestamps)
        return 0

    return 0


def _print_summary(result: TempoResult) -> None:
    print(f"Estimated BPM: {result.bpm:.2f}")
    print(f"Rounded BPM: {result.rounded_bpm}")
    print(f"Beats detected: {len(result.beat_timestamps)}")


def _print_beat_timestamps(beat_timestamps: list[float]) -> None:
    print()
    print("Beat timestamps (seconds):")
    for timestamp in beat_timestamps:
        print(f"{timestamp:.3f}")
