# bpm-detector

CLI tool for detecting BPM and beat timestamps from WAV files.

## Status

Project scaffold is in place. WAV loading, preprocessing, and first-pass global BPM
detection are implemented. The CLI can print either a compact summary or the full beat
timestamp list.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Planned usage

```bash
PYTHONPATH=src python -m bpm_detector detect path/to/file.wav
PYTHONPATH=src python -m bpm_detector detect path/to/file.wav --show-beats
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The test suite generates synthetic click-track WAV files and checks both BPM accuracy
and beat timestamp spacing.

Current CLI output:

```text
Estimated BPM: 127.83
Rounded BPM: 128
Beats detected: 243
```

With `--show-beats`:

```text
Estimated BPM: 127.83
Rounded BPM: 128
Beats detected: 243

Beat timestamps (seconds):
0.472
0.941
1.408
...
```

## Notes

In a fully provisioned environment, you can install the package in editable mode and run
`python -m bpm_detector ...` normally. In this sandbox, offline verification uses
`PYTHONPATH=src` because the local virtual environment does not include packaging tools.

Current WAV support is limited to uncompressed PCM sample widths that Python's standard
library `wave` module can read.
