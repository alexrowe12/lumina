"""Top-level package for bpm_detector."""

from .grid import build_bar_timestamps
from .models import (
    AudioData,
    BarFeatures,
    NoveltyCurve,
    SectionAnalysisResult,
    SectionBoundary,
    TempoResult,
)
from .sections import (
    compute_novelty_curve,
    detect_sections,
    extract_bar_features,
    select_section_boundaries,
)
from .tempo import TempoDetectionError, detect_tempo

__all__ = [
    "AudioData",
    "BarFeatures",
    "NoveltyCurve",
    "SectionAnalysisResult",
    "SectionBoundary",
    "TempoDetectionError",
    "TempoResult",
    "build_bar_timestamps",
    "compute_novelty_curve",
    "detect_sections",
    "extract_bar_features",
    "select_section_boundaries",
    "detect_tempo",
]
