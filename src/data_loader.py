"""
Data loader for Harmonix Set mel spectrograms and segment annotations.

Loads pre-computed mel spectrograms and converts segment annotations
to frame-level binary labels for boundary detection.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


# Harmonix spectrogram parameters (derived from data)
SAMPLE_RATE = 22050
HOP_LENGTH = 1024
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # ~21.5 fps
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE  # ~0.0464 seconds per frame

# Boundary detection parameters
BOUNDARY_TOLERANCE_FRAMES = 2  # Frames within this distance are labeled as boundary


class HarmonixDataset(Dataset):
    """
    Dataset for boundary detection training.

    Returns mel spectrogram patches and binary boundary labels.
    """

    def __init__(
        self,
        mel_dir: str,
        segments_dir: str,
        metadata_path: str,
        file_ids: Optional[List[str]] = None,
        context_frames: int = 64,  # ~3 seconds of context
        normalize: bool = True,
        boundary_tolerance_sec: float = 1.0,
    ):
        """
        Args:
            mel_dir: Directory containing mel spectrogram .npy files
            segments_dir: Directory containing segment annotation .txt files
            metadata_path: Path to metadata.csv
            file_ids: List of file IDs to include (e.g., ['0001_12step']). If None, use all.
            context_frames: Number of frames on each side of target frame
            normalize: Whether to normalize spectrograms
            boundary_tolerance_sec: Tolerance in seconds for labeling boundary frames
        """
        self.mel_dir = mel_dir
        self.segments_dir = segments_dir
        self.context_frames = context_frames
        self.normalize = normalize
        self.boundary_tolerance_frames = int(boundary_tolerance_sec * FRAME_RATE)

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)

        # Get list of files to use
        if file_ids is not None:
            self.file_ids = file_ids
        else:
            # Use all files that have both mel and segments
            mel_files = set(f.replace('-mel.npy', '') for f in os.listdir(mel_dir) if f.endswith('-mel.npy'))
            seg_files = set(f.replace('.txt', '') for f in os.listdir(segments_dir) if f.endswith('.txt'))
            self.file_ids = sorted(list(mel_files & seg_files))

        # Pre-load all data into memory for faster training
        self.data = []
        self._load_all_data()

    def _load_all_data(self):
        """Load all spectrograms and create frame-level samples."""
        for file_id in self.file_ids:
            # Load mel spectrogram
            mel_path = os.path.join(self.mel_dir, f"{file_id}-mel.npy")
            mel = np.load(mel_path).astype(np.float32)

            # Normalize if requested
            if self.normalize:
                mel = np.log1p(mel)  # Log compression
                mel = (mel - mel.mean()) / (mel.std() + 1e-8)

            # Load segment boundaries
            seg_path = os.path.join(self.segments_dir, f"{file_id}.txt")
            boundaries = self._load_boundaries(seg_path, mel.shape[1])

            # Create frame-level binary labels
            labels = self._create_boundary_labels(boundaries, mel.shape[1])

            # Store with padding for context
            self.data.append({
                'file_id': file_id,
                'mel': mel,
                'labels': labels,
                'boundaries': boundaries,
            })

        # Precompute cumulative frame counts for fast indexing
        self._cumsum = [0]
        for d in self.data:
            track_frames = d['mel'].shape[1] - 2 * self.context_frames
            self._cumsum.append(self._cumsum[-1] + track_frames)

    def _load_boundaries(self, seg_path: str, num_frames: int) -> List[int]:
        """Load segment file and convert timestamps to frame indices."""
        boundaries = []
        with open(seg_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    frame_idx = int(timestamp * FRAME_RATE)
                    # Skip boundaries at very start or end, and 'end' labels
                    if parts[1].lower() != 'end' and 0 < frame_idx < num_frames - 1:
                        boundaries.append(frame_idx)
        return sorted(boundaries)

    def _create_boundary_labels(self, boundaries: List[int], num_frames: int) -> np.ndarray:
        """Create binary labels array with 1s near boundaries."""
        labels = np.zeros(num_frames, dtype=np.float32)
        for b in boundaries:
            start = max(0, b - self.boundary_tolerance_frames)
            end = min(num_frames, b + self.boundary_tolerance_frames + 1)
            labels[start:end] = 1.0
        return labels

    def __len__(self) -> int:
        """Return total number of frames across all tracks."""
        return self._cumsum[-1] if self._cumsum else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single frame with context and its label."""
        # Binary search to find which track this index belongs to
        lo, hi = 0, len(self.data)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumsum[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        track_idx = lo

        if track_idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range")

        d = self.data[track_idx]
        local_idx = idx - self._cumsum[track_idx] + self.context_frames

        # Extract context window
        start = local_idx - self.context_frames
        end = local_idx + self.context_frames + 1
        mel_patch = d['mel'][:, start:end]

        # Get label for center frame
        label = d['labels'][local_idx]

        # Add channel dimension
        mel_patch = mel_patch[np.newaxis, :, :]  # (1, mel_bins, time)

        return torch.from_numpy(mel_patch), torch.tensor(label)

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        total_positive = sum(d['labels'].sum() for d in self.data)
        total_frames = sum(d['labels'].shape[0] for d in self.data)
        total_negative = total_frames - total_positive

        # Weight inversely proportional to frequency
        pos_weight = total_negative / (total_positive + 1e-8)
        return torch.tensor([pos_weight], dtype=torch.float32)


def create_data_splits(
    mel_dir: str,
    segments_dir: str,
    metadata_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split file IDs into train/val/test sets.

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    # Get all valid file IDs
    mel_files = set(f.replace('-mel.npy', '') for f in os.listdir(mel_dir) if f.endswith('-mel.npy'))
    seg_files = set(f.replace('.txt', '') for f in os.listdir(segments_dir) if f.endswith('.txt'))
    all_ids = sorted(list(mel_files & seg_files))

    # Shuffle with fixed seed
    rng = np.random.RandomState(seed)
    rng.shuffle(all_ids)

    # Split
    n = len(all_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]

    return train_ids, val_ids, test_ids


def get_data_loaders(
    mel_dir: str,
    segments_dir: str,
    metadata_path: str,
    batch_size: int = 64,
    context_frames: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    samples_per_epoch: int = 50000,  # Limit samples per epoch for faster training
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create train/val/test data loaders.

    Args:
        samples_per_epoch: Maximum samples per training epoch (for faster training)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights)
    """
    from torch.utils.data import SubsetRandomSampler

    train_ids, val_ids, test_ids = create_data_splits(
        mel_dir, segments_dir, metadata_path, seed=seed
    )

    print(f"Data split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    train_dataset = HarmonixDataset(
        mel_dir, segments_dir, metadata_path,
        file_ids=train_ids,
        context_frames=context_frames,
    )
    val_dataset = HarmonixDataset(
        mel_dir, segments_dir, metadata_path,
        file_ids=val_ids,
        context_frames=context_frames,
    )
    test_dataset = HarmonixDataset(
        mel_dir, segments_dir, metadata_path,
        file_ids=test_ids,
        context_frames=context_frames,
    )

    class_weights = train_dataset.get_class_weights()

    # Use random subset for training to limit epoch size
    rng = np.random.RandomState(seed)
    train_indices = rng.choice(len(train_dataset), size=min(samples_per_epoch, len(train_dataset)), replace=False)
    val_indices = rng.choice(len(val_dataset), size=min(samples_per_epoch // 5, len(val_dataset)), replace=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Test the data loader
    mel_dir = "data/harmonix/melspecs"
    segments_dir = "data/harmonix/harmonixset/dataset/segments"
    metadata_path = "data/harmonix/harmonixset/dataset/metadata.csv"

    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        mel_dir, segments_dir, metadata_path, batch_size=32
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights (pos_weight): {class_weights.item():.2f}")

    # Check a batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch X shape: {batch_x.shape}")
    print(f"Batch Y shape: {batch_y.shape}")
    print(f"Positive ratio in batch: {batch_y.mean().item():.4f}")
