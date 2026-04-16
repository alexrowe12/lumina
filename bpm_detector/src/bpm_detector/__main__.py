"""Module entrypoint for `python -m bpm_detector`."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

