# bpm-detector

CLI tool for detecting BPM and beat timestamps from WAV files.

## Status

Project scaffold in progress. Detection logic is not implemented yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Planned usage

```bash
PYTHONPATH=src python -m bpm_detector detect path/to/file.wav
```

## Notes

In a fully provisioned environment, you can install the package in editable mode and run
`python -m bpm_detector ...` normally. In this sandbox, offline verification uses
`PYTHONPATH=src` because the local virtual environment does not include packaging tools.
