# bpm-detector

CLI tool for detecting BPM, beat timestamps, and section boundaries from WAV files.

## Status

The prerecorded analysis foundation is in place. The CLI can detect:

- global BPM
- beat timestamps
- bar-aligned structural section boundaries
- section debug output with novelty scores and selected bars

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
bpm-detector detect path/to/file.wav --show-sections
bpm-detector detect path/to/file.wav --show-sections --debug-sections
```

## Tests

```bash
pytest -q
```

The test suite generates synthetic WAV fixtures and checks BPM accuracy, beat/bar timing,
feature extraction, novelty scoring, section selection, and CLI output.

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

With `--show-sections --debug-sections`:

```text
Estimated BPM: 120.27
Rounded BPM: 120
Beats detected: 33

Section boundaries (seconds):
15.999

Section debug:
bar=1 time=0.115 raw=0.000 selected=no
bar=2 time=2.005 raw=0.098 selected=no
...
bar=9 time=15.999 raw=1.000 weighted=1.388 selected=yes
```

Beat timestamps are reported in raw seconds relative to the original WAV file, even if
leading or trailing silence is trimmed during preprocessing.

## Notes

The repo is set up to be run from a local virtual environment. In this sandbox, network
and packaging constraints may require verification with `PYTHONPATH=src`, but the intended
local workflow is editable install plus the `bpm-detector` command.

Current WAV support is limited to uncompressed PCM sample widths that Python's standard
library `wave` module can read.
