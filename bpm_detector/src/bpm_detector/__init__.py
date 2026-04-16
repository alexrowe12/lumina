"""Top-level package for bpm_detector."""

from .grid import build_bar_timestamps
from .models import AudioData, BarFeatures, TempoResult
from .sections import extract_bar_features
from .tempo import TempoDetectionError, detect_tempo

__all__ = [
    "AudioData",
    "BarFeatures",
    "TempoDetectionError",
    "TempoResult",
    "build_bar_timestamps",
    "extract_bar_features",
    "detect_tempo",
]
