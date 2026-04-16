"""Command-line interface for bpm_detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio import AudioLoadError, load_audio
from .models import AudioData, NoveltyCurve, SectionBoundary, TempoResult
from .sections import compute_novelty_curve, extract_bar_features, select_section_boundaries
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
    detect_parser.add_argument(
        "--show-sections",
        action="store_true",
        help="Print detected section boundary timestamps in raw seconds.",
    )
    detect_parser.add_argument(
        "--debug-sections",
        action="store_true",
        help="Print section boundary debug information including scores and bar indices.",
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

        section_boundaries = []
        novelty_curve = None
        if args.show_sections or args.debug_sections:
            novelty_curve, section_boundaries = analyze_sections(audio, result)

        _print_summary(result)
        if args.show_beats:
            _print_beat_timestamps(result.beat_timestamps)
        if args.show_sections:
            _print_section_timestamps(section_boundaries)
        if args.debug_sections and novelty_curve is not None:
            _print_section_debug(novelty_curve, section_boundaries)
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


def analyze_sections(
    audio: AudioData,
    tempo_result: TempoResult,
) -> tuple[NoveltyCurve, list[SectionBoundary]]:
    """Run the full section-analysis pipeline for a loaded track."""

    bar_features = extract_bar_features(audio, tempo_result)
    novelty_curve = compute_novelty_curve(bar_features, window_bars=2)
    section_boundaries = select_section_boundaries(
        novelty_curve,
        min_score=0.4,
        min_spacing_bars=4,
    )
    return novelty_curve, section_boundaries


def _print_section_timestamps(section_boundaries: list[SectionBoundary]) -> None:
    print()
    print("Section boundaries (seconds):")
    for boundary in section_boundaries:
        print(f"{boundary.timestamp:.3f}")


def _print_section_debug(
    novelty_curve: NoveltyCurve,
    section_boundaries: list[SectionBoundary],
) -> None:
    selected_by_bar_index = {boundary.bar_index: boundary for boundary in section_boundaries}

    print()
    print("Section debug:")
    for bar_index, (timestamp, raw_score) in enumerate(
        zip(novelty_curve.timestamps, novelty_curve.scores)
    ):
        boundary = selected_by_bar_index.get(bar_index)
        if boundary is None:
            print(
                f"bar={bar_index + 1} time={timestamp:.3f} raw={raw_score:.3f} selected=no"
            )
            continue

        print(
            f"bar={bar_index + 1} time={timestamp:.3f} raw={raw_score:.3f} "
            f"weighted={boundary.weighted_score:.3f} selected=yes"
        )
