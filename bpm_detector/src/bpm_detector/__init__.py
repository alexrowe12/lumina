"""Top-level package for bpm_detector."""

from .models import AudioData, TempoResult
from .tempo import TempoDetectionError, detect_tempo

__all__ = ["AudioData", "TempoDetectionError", "TempoResult", "detect_tempo"]
