"""
Inference script for boundary detection.

Takes an audio file, computes mel spectrogram, runs the trained model,
and outputs detected boundary timestamps.
"""

import os
import sys
import argparse
import numpy as np
import torch
import librosa
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import get_model
from src.utils import get_device, format_time, SAMPLE_RATE, HOP_LENGTH, N_MELS, N_FFT, FMIN, FMAX


def compute_mel_spectrogram(audio_path: str) -> np.ndarray:
    """
    Compute mel spectrogram from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Mel spectrogram of shape (n_mels, time_frames)
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )

    return mel.astype(np.float32)


def preprocess_mel(mel: np.ndarray) -> np.ndarray:
    """Apply same preprocessing as training data."""
    mel = np.log1p(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint.get('model_type', 'default')
    context_frames = checkpoint.get('context_frames', 64)

    model = get_model(model_type, context_frames=context_frames)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, context_frames


def predict_boundaries(
    mel: np.ndarray,
    model: torch.nn.Module,
    context_frames: int,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run boundary detection on a mel spectrogram.

    Args:
        mel: Preprocessed mel spectrogram (n_mels, time_frames)
        model: Trained model
        context_frames: Context size used during training
        device: Torch device
        threshold: Probability threshold for boundary detection
        batch_size: Batch size for inference

    Returns:
        Array of boundary probabilities for each frame
    """
    n_frames = mel.shape[1]
    probabilities = np.zeros(n_frames, dtype=np.float32)

    # Pad mel spectrogram for context
    mel_padded = np.pad(mel, ((0, 0), (context_frames, context_frames)), mode='edge')

    # Process in batches
    model.eval()
    with torch.no_grad():
        for start_idx in range(0, n_frames, batch_size):
            end_idx = min(start_idx + batch_size, n_frames)
            batch_size_actual = end_idx - start_idx

            # Extract patches
            patches = np.zeros((batch_size_actual, 1, N_MELS, 2 * context_frames + 1), dtype=np.float32)
            for i, frame_idx in enumerate(range(start_idx, end_idx)):
                padded_idx = frame_idx + context_frames
                patches[i, 0, :, :] = mel_padded[:, padded_idx - context_frames:padded_idx + context_frames + 1]

            # Run model
            patches_tensor = torch.from_numpy(patches).to(device)
            outputs = model(patches_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()

            probabilities[start_idx:end_idx] = probs

    return probabilities


def peak_pick_boundaries(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_distance_sec: float = 5.0,
    frame_rate: float = SAMPLE_RATE / HOP_LENGTH,
) -> List[float]:
    """
    Pick boundary peaks from probability curve.

    Args:
        probabilities: Boundary probability per frame
        threshold: Minimum probability threshold
        min_distance_sec: Minimum distance between boundaries in seconds
        frame_rate: Frames per second

    Returns:
        List of boundary timestamps in seconds
    """
    min_distance_frames = int(min_distance_sec * frame_rate)

    # Find peaks above threshold
    above_threshold = probabilities > threshold

    boundaries = []
    i = 0
    while i < len(probabilities):
        if above_threshold[i]:
            # Find the peak in this region
            start = i
            while i < len(probabilities) and above_threshold[i]:
                i += 1
            end = i

            # Get the frame with maximum probability in this region
            peak_idx = start + np.argmax(probabilities[start:end])
            peak_time = peak_idx / frame_rate

            # Check minimum distance from last boundary
            if not boundaries or peak_time - boundaries[-1] >= min_distance_sec:
                boundaries.append(peak_time)
        else:
            i += 1

    return boundaries


def detect_boundaries(
    audio_path: str,
    model_path: str,
    threshold: float = 0.5,
    min_distance_sec: float = 5.0,
    output_json: bool = False,
) -> List[float]:
    """
    Detect section boundaries in an audio file.

    Args:
        audio_path: Path to audio file
        model_path: Path to trained model checkpoint
        threshold: Probability threshold
        min_distance_sec: Minimum distance between boundaries
        output_json: Whether to output JSON format

    Returns:
        List of boundary timestamps
    """
    device = get_device()

    # Load model
    model, context_frames = load_model(model_path, device)

    # Compute mel spectrogram
    mel = compute_mel_spectrogram(audio_path)
    mel = preprocess_mel(mel)

    duration = mel.shape[1] * HOP_LENGTH / SAMPLE_RATE

    # Run inference
    probabilities = predict_boundaries(mel, model, context_frames, device)

    # Pick peaks
    boundaries = peak_pick_boundaries(
        probabilities,
        threshold=threshold,
        min_distance_sec=min_distance_sec,
    )

    # Always include start
    if not boundaries or boundaries[0] > 1.0:
        boundaries = [0.0] + boundaries

    if output_json:
        import json
        result = {
            "file": audio_path,
            "duration": duration,
            "boundaries": boundaries,
            "threshold": threshold,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\nSection Boundaries for: {audio_path}")
        print(f"Duration: {format_time(duration)} ({duration:.1f}s)")
        print("=" * 50)
        for i, b in enumerate(boundaries, 1):
            print(f"  {i:2d}. {format_time(b):>5} ({b:.2f}s)")
        print("=" * 50)
        print(f"Total sections: {len(boundaries)}")

    return boundaries


def main():
    parser = argparse.ArgumentParser(description="Detect section boundaries in audio")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="models/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for boundary detection")
    parser.add_argument("--min-distance", type=float, default=5.0,
                        help="Minimum distance between boundaries in seconds")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON format")

    args = parser.parse_args()

    detect_boundaries(
        args.audio,
        args.model,
        threshold=args.threshold,
        min_distance_sec=args.min_distance,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
