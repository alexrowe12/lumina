"""Command-line interface for bpm_detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio import AudioLoadError, load_audio
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

        print(f"Estimated BPM: {result.bpm:.2f}")
        print(f"Rounded BPM: {result.rounded_bpm}")
        return 0

    return 0
