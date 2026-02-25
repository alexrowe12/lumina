"""Shared utilities for Lumina boundary detection."""

import torch

# Audio processing parameters (matching Harmonix dataset)
SAMPLE_RATE = 22050
HOP_LENGTH = 1024
N_MELS = 80
N_FFT = 2048
FMIN = 0
FMAX = 8000
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # ~21.5 fps


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"
