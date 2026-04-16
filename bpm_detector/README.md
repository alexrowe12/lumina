# bpm-detector

CLI tool for detecting BPM, beat timestamps, and eventually section boundaries from WAV files.

## Status

The BPM-detection foundation is in place. WAV loading, preprocessing, global tempo
detection, and beat timestamp extraction are implemented. The next stage is structural
section detection for prerecorded tracks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

After that, use the installed console command directly:

```bash
bpm-detector detect path/to/file.wav
bpm-detector detect path/to/file.wav --show-beats
```

## Tests

```bash
pytest -q
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

Beat timestamps are reported in raw seconds relative to the original WAV file, even if
leading or trailing silence is trimmed during preprocessing.

## Notes

The repo is set up to be run from a local virtual environment. In this sandbox, network
and packaging constraints may require verification with `PYTHONPATH=src`, but the intended
local workflow is editable install plus the `bpm-detector` command.

Current WAV support is limited to uncompressed PCM sample widths that Python's standard
library `wave` module can read.
