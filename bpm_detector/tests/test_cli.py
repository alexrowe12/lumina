from __future__ import annotations

from bpm_detector.cli import main


def test_cli_prints_section_timestamps(structured_sections_factory, capsys) -> None:
    wav_path = structured_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 110.0, 1760.0, 1760.0, 1760.0, 1760.0, 1760.0],
        bpm=120,
    )

    exit_code = main(["detect", str(wav_path), "--show-sections"])

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    header_index = lines.index("Section boundaries (seconds):")
    section_lines = [line for line in lines[header_index + 1 :] if line.strip()]

    assert exit_code == 0
    assert "Estimated BPM:" in captured.out
    assert "Section boundaries (seconds):" in captured.out
    assert section_lines
    assert all("." in line for line in section_lines)


def test_cli_prints_debug_section_rows(structured_sections_factory, capsys) -> None:
    wav_path = structured_sections_factory(
        section_frequencies_hz=[110.0, 110.0, 110.0, 1760.0, 1760.0, 1760.0, 1760.0, 1760.0],
        bpm=120,
    )

    exit_code = main(["detect", str(wav_path), "--debug-sections"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Section debug:" in captured.out
    assert "selected=yes" in captured.out
    assert "weighted=" in captured.out
