#!/usr/bin/env python3
"""
Detect section boundaries in a song using multiple audio analysis signals.

Combines three novelty detection approaches:
1. Harmonic novelty (chroma + MFCC self-similarity)
2. Energy novelty (RMS energy changes)
3. Onset novelty (percussive density changes)

Usage:
    python detect_boundaries.py <file> [--sensitivity low|medium|high] [--weights H,E,O]
"""

import argparse
import librosa
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


# Sensitivity presets
PRESETS = {
    "low": {
        "kernel_size": 24,
        "threshold_scale": 1.0,
        "min_section_sec": 20,
    },
    "medium": {
        "kernel_size": 16,
        "threshold_scale": 0.5,
        "min_section_sec": 15,
    },
    "high": {
        "kernel_size": 12,
        "threshold_scale": 0.25,
        "min_section_sec": 10,
    },
}

# Default weights for combining novelty signals (harmonic, energy, onset)
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)


def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    x = np.asarray(x, dtype=np.float32)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 0:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x)


def compute_harmonic_novelty(y: np.ndarray, sr: int, hop_length: int, kernel_size: int) -> np.ndarray:
    """
    Compute novelty based on harmonic/timbral self-similarity.
    Detects structural changes like key shifts, timbral transitions.
    """
    # Compute chroma and MFCC features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    features = np.vstack([chroma, mfcc])

    # Self-similarity via recurrence matrix
    rec = librosa.segment.recurrence_matrix(
        features, width=3, mode='affinity', sym=True
    )
    rec_filtered = median_filter(rec, size=(9, 9))

    # Checkerboard kernel novelty detection
    novelty = np.zeros(rec.shape[0], dtype=np.float32)
    for i in range(kernel_size, rec.shape[0] - kernel_size):
        before = rec_filtered[i - kernel_size:i, i - kernel_size:i]
        after = rec_filtered[i:i + kernel_size, i:i + kernel_size]
        cross = rec_filtered[i - kernel_size:i, i:i + kernel_size]
        novelty[i] = np.mean(before) + np.mean(after) - 2 * np.mean(cross)

    return np.maximum(novelty, 0)


def compute_energy_novelty(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Compute novelty based on energy changes.
    Detects sudden energy jumps (breakdown→drop) and rising energy (buildups).
    """
    # Compute RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Smooth the energy curve
    rms_smooth = uniform_filter1d(rms, size=20)

    # First derivative - detects sudden changes
    energy_diff = np.diff(rms_smooth, prepend=rms_smooth[0])

    # We care about both sudden increases (positive) and the magnitude of change
    # Take absolute value but weight positive changes more (energy increases are more salient)
    novelty = np.abs(energy_diff)
    novelty[energy_diff > 0] *= 1.5  # Boost energy increases

    # Also detect sustained rises (buildups) by looking at longer-term trend
    window = 50  # frames
    rising_score = np.zeros_like(rms_smooth)
    for i in range(window, len(rms_smooth)):
        segment = rms_smooth[i - window:i]
        # Check if consistently rising
        rises = np.sum(np.diff(segment) > 0) / window
        if rises > 0.6:  # More than 60% of frames are rising
            rising_score[i] = rises * np.std(segment)

    # Combine instant changes with rising detection
    # Peaks at the END of a rise (where buildup ends / drop begins)
    rising_diff = np.diff(rising_score, prepend=0)
    rising_peaks = np.maximum(-rising_diff, 0)  # Negative diff = end of rise

    combined = novelty + rising_peaks
    return combined.astype(np.float32)


def compute_onset_novelty(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """
    Compute novelty based on onset/percussive density changes.
    Detects transitions from sparse to dense (breakdown→drop) or vice versa.
    """
    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Compute local density (smoothed onset strength)
    window = 40  # frames
    density = uniform_filter1d(onset_env, size=window)

    # Novelty = change in density
    density_diff = np.abs(np.diff(density, prepend=density[0]))

    # Also look for sudden onset bursts
    onset_peaks = onset_env - uniform_filter1d(onset_env, size=10)
    onset_peaks = np.maximum(onset_peaks, 0)

    # Combine density changes with onset peaks
    combined = density_diff + 0.5 * onset_peaks
    return combined.astype(np.float32)


def combine_novelty_curves(
    harmonic: np.ndarray,
    energy: np.ndarray,
    onset: np.ndarray,
    weights: tuple[float, float, float]
) -> np.ndarray:
    """Combine multiple novelty signals into one."""
    # Ensure same length (pad shorter ones)
    max_len = max(len(harmonic), len(energy), len(onset))

    def pad_to_length(arr, length):
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)), mode='edge')
        return arr[:length]

    harmonic = pad_to_length(harmonic, max_len)
    energy = pad_to_length(energy, max_len)
    onset = pad_to_length(onset, max_len)

    # Normalize each to [0, 1]
    h_norm = normalize(harmonic)
    e_norm = normalize(energy)
    o_norm = normalize(onset)

    # Weighted combination
    w_h, w_e, w_o = weights
    combined = w_h * h_norm + w_e * e_norm + w_o * o_norm

    return combined


def detect_boundaries(
    file_path: str,
    sensitivity: str = "medium",
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    verbose: bool = True
) -> list[float]:
    """Detect section boundaries using multi-signal novelty detection."""
    params = PRESETS[sensitivity]
    hop_length = 512

    if verbose:
        print(f"\nAnalyzing: {file_path}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Weights (H/E/O): {weights[0]:.1f}/{weights[1]:.1f}/{weights[2]:.1f}")
        print("Loading audio...")

    # Load audio
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)

    if verbose:
        print(f"Duration: {format_time(duration)} ({duration:.1f}s)")
        print("Computing novelty signals...")

    # Compute all three novelty signals
    if verbose:
        print("  - Harmonic novelty (structure)...")
    harmonic = compute_harmonic_novelty(y, sr, hop_length, params["kernel_size"])

    if verbose:
        print("  - Energy novelty (intensity)...")
    energy = compute_energy_novelty(y, sr, hop_length)

    if verbose:
        print("  - Onset novelty (rhythm)...")
    onset = compute_onset_novelty(y, sr, hop_length)

    # Combine signals
    if verbose:
        print("Combining signals and finding boundaries...\n")
    combined = combine_novelty_curves(harmonic, energy, onset, weights)

    # Find peaks
    min_frames = int(params["min_section_sec"] * sr / hop_length)
    threshold = np.mean(combined) + params["threshold_scale"] * np.std(combined)

    peaks = librosa.util.peak_pick(
        combined,
        pre_max=min_frames // 2,
        post_max=min_frames // 2,
        pre_avg=min_frames // 2,
        post_avg=min_frames // 2,
        delta=threshold,
        wait=min_frames
    )

    # Convert to timestamps
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # Always include start
    if len(times) == 0 or times[0] > 2.0:
        times = np.concatenate([[0.0], times])

    # Print results
    if verbose:
        print(f"Section Boundaries for: {file_path}")
        print("=" * 50)
        for i, boundary in enumerate(times, 1):
            print(f"  {i:2d}.  {format_time(boundary):>5}  ({boundary:.2f}s)")
        print("=" * 50)
        print(f"Total sections detected: {len(times)}\n")

    return times.tolist()


def parse_weights(weights_str: str) -> tuple[float, float, float]:
    """Parse weights string like '0.4,0.3,0.3' into tuple."""
    parts = [float(x.strip()) for x in weights_str.split(",")]
    if len(parts) != 3:
        raise ValueError("Weights must have exactly 3 values (harmonic,energy,onset)")
    # Normalize to sum to 1
    total = sum(parts)
    return tuple(p / total for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="Detect section boundaries using multi-signal analysis"
    )
    parser.add_argument(
        "file", nargs="?", default="losing_it.wav",
        help="Audio file path"
    )
    parser.add_argument(
        "--sensitivity", "-s",
        choices=["low", "medium", "high"],
        default="medium",
        help="Detection sensitivity (default: medium)"
    )
    parser.add_argument(
        "--weights", "-w",
        default="0.4,0.3,0.3",
        help="Weights for harmonic,energy,onset signals (default: 0.4,0.3,0.3)"
    )
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    detect_boundaries(args.file, args.sensitivity, weights)


if __name__ == "__main__":
    main()
