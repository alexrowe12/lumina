"""Top-level package for bpm_detector."""

from .grid import build_bar_timestamps
from .models import AudioData, BarFeatures, NoveltyCurve, TempoResult
from .sections import compute_novelty_curve, extract_bar_features
from .tempo import TempoDetectionError, detect_tempo

__all__ = [
    "AudioData",
    "BarFeatures",
    "NoveltyCurve",
    "TempoDetectionError",
    "TempoResult",
    "build_bar_timestamps",
    "compute_novelty_curve",
    "extract_bar_features",
    "detect_tempo",
]
