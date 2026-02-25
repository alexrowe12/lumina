#!/usr/bin/env python3
"""
Detect section boundaries in a song using a three-stage pipeline:
1. Coarse detection (multi-signal novelty)
2. Local refinement (energy jump + spectral flux + onset strength)
3. Beat quantization (snap to nearest downbeat)

Usage:
    python detect_boundaries.py <file> [options]
"""

import argparse
import librosa
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


# Sensitivity presets for coarse detection
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

DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)


def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS.ms format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    x = np.asarray(x, dtype=np.float32)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 0:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x)


# =============================================================================
# STAGE 1: Coarse Detection
# =============================================================================

def compute_harmonic_novelty(y: np.ndarray, sr: int, hop_length: int, kernel_size: int) -> np.ndarray:
    """Compute novelty based on harmonic/timbral self-similarity."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    features = np.vstack([chroma, mfcc])

    rec = librosa.segment.recurrence_matrix(
        features, width=3, mode='affinity', sym=True
    )
    rec_filtered = median_filter(rec, size=(9, 9))

    novelty = np.zeros(rec.shape[0], dtype=np.float32)
    for i in range(kernel_size, rec.shape[0] - kernel_size):
        before = rec_filtered[i - kernel_size:i, i - kernel_size:i]
        after = rec_filtered[i:i + kernel_size, i:i + kernel_size]
        cross = rec_filtered[i - kernel_size:i, i:i + kernel_size]
        novelty[i] = np.mean(before) + np.mean(after) - 2 * np.mean(cross)

    return np.maximum(novelty, 0)


def compute_energy_novelty(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Compute novelty based on energy changes."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_smooth = uniform_filter1d(rms, size=20)
    energy_diff = np.diff(rms_smooth, prepend=rms_smooth[0])

    novelty = np.abs(energy_diff)
    novelty[energy_diff > 0] *= 1.5

    window = 50
    rising_score = np.zeros_like(rms_smooth)
    for i in range(window, len(rms_smooth)):
        segment = rms_smooth[i - window:i]
        rises = np.sum(np.diff(segment) > 0) / window
        if rises > 0.6:
            rising_score[i] = rises * np.std(segment)

    rising_diff = np.diff(rising_score, prepend=0)
    rising_peaks = np.maximum(-rising_diff, 0)

    combined = novelty + rising_peaks
    return combined.astype(np.float32)


def compute_onset_novelty(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Compute novelty based on onset/percussive density changes."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    window = 40
    density = uniform_filter1d(onset_env, size=window)
    density_diff = np.abs(np.diff(density, prepend=density[0]))

    onset_peaks = onset_env - uniform_filter1d(onset_env, size=10)
    onset_peaks = np.maximum(onset_peaks, 0)

    combined = density_diff + 0.5 * onset_peaks
    return combined.astype(np.float32)


def combine_novelty_curves(
    harmonic: np.ndarray,
    energy: np.ndarray,
    onset: np.ndarray,
    weights: tuple[float, float, float]
) -> np.ndarray:
    """Combine multiple novelty signals into one."""
    max_len = max(len(harmonic), len(energy), len(onset))

    def pad_to_length(arr, length):
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)), mode='edge')
        return arr[:length]

    harmonic = pad_to_length(harmonic, max_len)
    energy = pad_to_length(energy, max_len)
    onset = pad_to_length(onset, max_len)

    h_norm = normalize(harmonic)
    e_norm = normalize(energy)
    o_norm = normalize(onset)

    w_h, w_e, w_o = weights
    return w_h * h_norm + w_e * e_norm + w_o * o_norm


def coarse_detection(
    y: np.ndarray,
    sr: int,
    sensitivity: str,
    weights: tuple[float, float, float],
    hop_length: int,
    verbose: bool
) -> np.ndarray:
    """Stage 1: Find approximate boundary candidates."""
    params = PRESETS[sensitivity]

    if verbose:
        print("Stage 1: Coarse Detection")
        print("  Computing harmonic novelty...")
    harmonic = compute_harmonic_novelty(y, sr, hop_length, params["kernel_size"])

    if verbose:
        print("  Computing energy novelty...")
    energy = compute_energy_novelty(y, sr, hop_length)

    if verbose:
        print("  Computing onset novelty...")
    onset = compute_onset_novelty(y, sr, hop_length)

    if verbose:
        print("  Combining signals...")
    combined = combine_novelty_curves(harmonic, energy, onset, weights)

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

    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    if len(times) == 0 or times[0] > 2.0:
        times = np.concatenate([[0.0], times])

    if verbose:
        print(f"  Found {len(times)} candidate boundaries\n")

    return times


# =============================================================================
# STAGE 2: Local Refinement
# =============================================================================

def compute_transition_score(y: np.ndarray, sr: int, hop_length: int = 128) -> np.ndarray:
    """
    Compute a high-resolution transition score combining:
    - Energy jump (RMS derivative)
    - Spectral flux
    - Onset strength
    """
    # Energy jump: derivative of RMS
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    energy_diff = np.abs(np.diff(rms, prepend=rms[0]))

    # Spectral flux: frame-to-frame spectral difference
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Ensure same length
    min_len = min(len(energy_diff), len(spectral_flux), len(onset_env))
    energy_diff = energy_diff[:min_len]
    spectral_flux = spectral_flux[:min_len]
    onset_env = onset_env[:min_len]

    # Normalize each
    energy_norm = normalize(energy_diff)
    flux_norm = normalize(spectral_flux)
    onset_norm = normalize(onset_env)

    # Combine with weights favoring energy and flux
    transition_score = 0.4 * energy_norm + 0.4 * flux_norm + 0.2 * onset_norm

    return transition_score.astype(np.float32)


def refine_boundary(
    y: np.ndarray,
    sr: int,
    candidate_time: float,
    window_sec: float = 5.0
) -> float:
    """
    Search within ±window_sec of candidate to find exact transition point.
    Uses high-resolution analysis of energy, spectral flux, and onset.
    """
    duration = len(y) / sr

    # Define search window
    start_time = max(0, candidate_time - window_sec)
    end_time = min(duration, candidate_time + window_sec)

    # Extract audio segment
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_segment = y[start_sample:end_sample]

    if len(y_segment) < sr * 0.5:  # Too short
        return candidate_time

    # Compute high-resolution transition score
    hop_length = 128  # ~5.8ms resolution at 22050 Hz
    transition_score = compute_transition_score(y_segment, sr, hop_length)

    # Find the peak (maximum transition)
    peak_frame = np.argmax(transition_score)
    peak_time = librosa.frames_to_time(peak_frame, sr=sr, hop_length=hop_length)

    # Convert back to absolute time
    refined_time = start_time + peak_time

    return refined_time


def refine_boundaries(
    y: np.ndarray,
    sr: int,
    candidates: np.ndarray,
    window_sec: float,
    verbose: bool
) -> np.ndarray:
    """Stage 2: Refine all candidate boundaries."""
    if verbose:
        print(f"Stage 2: Local Refinement (±{window_sec}s window)")

    refined = []
    for i, candidate in enumerate(candidates):
        if candidate == 0.0:  # Keep start time as-is
            refined.append(0.0)
            continue

        new_time = refine_boundary(y, sr, candidate, window_sec)
        if verbose:
            diff = new_time - candidate
            direction = "+" if diff >= 0 else ""
            print(f"  {format_time(candidate)} → {format_time(new_time)} ({direction}{diff:.2f}s)")
        refined.append(new_time)

    if verbose:
        print()

    return np.array(refined)


# =============================================================================
# STAGE 3: Beat Quantization
# =============================================================================

def quantize_to_beats(
    times: np.ndarray,
    beat_times: np.ndarray,
    downbeat_times: np.ndarray,
    max_snap_sec: float = 0.5,
    verbose: bool = True
) -> np.ndarray:
    """
    Stage 3: Snap boundaries to nearest downbeat (or beat if no downbeat is close).
    Only snaps if a beat is within max_snap_sec.
    """
    if verbose:
        print(f"Stage 3: Beat Quantization (max snap: {max_snap_sec}s)")

    quantized = []
    for t in times:
        if t == 0.0:
            quantized.append(0.0)
            continue

        # Find nearest downbeat
        downbeat_dists = np.abs(downbeat_times - t)
        nearest_downbeat_idx = np.argmin(downbeat_dists)
        nearest_downbeat = downbeat_times[nearest_downbeat_idx]
        downbeat_dist = downbeat_dists[nearest_downbeat_idx]

        # Find nearest beat
        beat_dists = np.abs(beat_times - t)
        nearest_beat_idx = np.argmin(beat_dists)
        nearest_beat = beat_times[nearest_beat_idx]
        beat_dist = beat_dists[nearest_beat_idx]

        # Prefer downbeat if close enough, else try beat, else keep original
        if downbeat_dist <= max_snap_sec:
            snapped = nearest_downbeat
            snap_type = "downbeat"
        elif beat_dist <= max_snap_sec:
            snapped = nearest_beat
            snap_type = "beat"
        else:
            snapped = t
            snap_type = "no snap"

        if verbose and snap_type != "no snap":
            diff = snapped - t
            direction = "+" if diff >= 0 else ""
            print(f"  {format_time(t)} → {format_time(snapped)} ({snap_type}, {direction}{diff:.3f}s)")
        elif verbose:
            print(f"  {format_time(t)} → {format_time(t)} (no beat within {max_snap_sec}s)")

        quantized.append(snapped)

    if verbose:
        print()

    return np.array(quantized)


def get_beat_grid(y: np.ndarray, sr: int, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Compute beat and downbeat times."""
    if verbose:
        print("  Computing beat grid...")

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')

    # Estimate downbeats (every 4th beat, assuming 4/4 time)
    # Start from the first beat
    if len(beats) >= 4:
        # Try to find the best downbeat alignment by looking at onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_frames = librosa.time_to_frames(beats, sr=sr)

        # Score each possible downbeat phase (0, 1, 2, 3)
        best_phase = 0
        best_score = 0
        for phase in range(4):
            downbeat_frames = beat_frames[phase::4]
            if len(downbeat_frames) > 0:
                score = np.mean(onset_env[downbeat_frames[downbeat_frames < len(onset_env)]])
                if score > best_score:
                    best_score = score
                    best_phase = phase

        downbeats = beats[best_phase::4]
    else:
        downbeats = beats

    if verbose:
        tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo
        print(f"  Tempo: {tempo_val:.1f} BPM, {len(beats)} beats, {len(downbeats)} downbeats")

    return beats, downbeats


# =============================================================================
# Main Pipeline
# =============================================================================

def detect_boundaries(
    file_path: str,
    sensitivity: str = "medium",
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    do_refine: bool = True,
    refine_window: float = 5.0,
    do_quantize: bool = True,
    max_snap: float = 0.5,
    verbose: bool = True
) -> list[float]:
    """
    Full three-stage boundary detection pipeline.

    Stage 1: Coarse detection using multi-signal novelty
    Stage 2: Local refinement using high-res energy/spectral/onset
    Stage 3: Beat quantization to snap to downbeats
    """
    hop_length = 512

    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {file_path}")
        print(f"Sensitivity: {sensitivity} | Weights: {weights[0]:.1f}/{weights[1]:.1f}/{weights[2]:.1f}")
        print(f"Refinement: {'ON' if do_refine else 'OFF'} | Quantization: {'ON' if do_quantize else 'OFF'}")
        print(f"{'='*60}\n")
        print("Loading audio...")

    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)

    if verbose:
        print(f"Duration: {format_time(duration)} ({duration:.1f}s)\n")

    # Stage 1: Coarse detection
    candidates = coarse_detection(y, sr, sensitivity, weights, hop_length, verbose)

    # Stage 2: Local refinement
    if do_refine:
        refined = refine_boundaries(y, sr, candidates, refine_window, verbose)
    else:
        refined = candidates

    # Stage 3: Beat quantization
    if do_quantize:
        if verbose:
            print("Computing beat grid for quantization...")
        beats, downbeats = get_beat_grid(y, sr, verbose)
        if verbose:
            print()
        final = quantize_to_beats(refined, beats, downbeats, max_snap, verbose)
    else:
        final = refined

    # Print final results
    if verbose:
        print(f"{'='*60}")
        print(f"Final Section Boundaries")
        print(f"{'='*60}")
        for i, boundary in enumerate(final, 1):
            print(f"  {i:2d}.  {format_time(boundary)}")
        print(f"{'='*60}")
        print(f"Total sections: {len(final)}\n")

    return final.tolist()


def parse_weights(weights_str: str) -> tuple[float, float, float]:
    """Parse weights string like '0.4,0.3,0.3' into tuple."""
    parts = [float(x.strip()) for x in weights_str.split(",")]
    if len(parts) != 3:
        raise ValueError("Weights must have exactly 3 values (harmonic,energy,onset)")
    total = sum(parts)
    return tuple(p / total for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="Detect section boundaries using three-stage pipeline"
    )
    parser.add_argument(
        "file", nargs="?", default="losing_it.wav",
        help="Audio file path"
    )
    parser.add_argument(
        "--sensitivity", "-s",
        choices=["low", "medium", "high"],
        default="medium",
        help="Coarse detection sensitivity (default: medium)"
    )
    parser.add_argument(
        "--weights", "-w",
        default="0.4,0.3,0.3",
        help="Weights for harmonic,energy,onset (default: 0.4,0.3,0.3)"
    )
    parser.add_argument(
        "--refine/--no-refine",
        dest="refine",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable local refinement (default: enabled)"
    )
    parser.add_argument(
        "--refine-window",
        type=float,
        default=5.0,
        help="Refinement search window in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--quantize/--no-quantize",
        dest="quantize",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable beat quantization (default: enabled)"
    )
    parser.add_argument(
        "--max-snap",
        type=float,
        default=0.5,
        help="Maximum snap distance for quantization in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()
    weights = parse_weights(args.weights)

    detect_boundaries(
        args.file,
        sensitivity=args.sensitivity,
        weights=weights,
        do_refine=args.refine,
        refine_window=args.refine_window,
        do_quantize=args.quantize,
        max_snap=args.max_snap,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
