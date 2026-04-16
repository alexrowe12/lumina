"""Command-line interface for bpm_detector."""

from __future__ import annotations

import argparse
from pathlib import Path


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
        parser.error("BPM detection is not implemented yet.")

    return 0

