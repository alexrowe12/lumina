"""
Training script for boundary detection CNN.

Supports MPS (Apple Silicon), CUDA, and CPU.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import get_data_loaders
from src.model import get_model, count_parameters
from src.utils import get_device


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Down-weights easy negatives so the model focuses on hard examples.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: Weighting factor for positive class (handles class imbalance)
        gamma: Focusing parameter - higher = more focus on hard examples (typically 2.0)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Probability of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting for class imbalance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight: down-weight easy examples
        focal_weight = (1 - p_t) ** self.gamma

        # Binary cross entropy (without reduction)
        bce = -targets * torch.log(probs + 1e-8) - (1 - targets) * torch.log(1 - probs + 1e-8)

        # Combine
        loss = alpha_t * focal_weight * bce

        return loss.mean()


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        # Calculate accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total = 0
    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total += batch_y.size(0)

            # Calculate metrics
            preds = (torch.sigmoid(outputs) > 0.5).float()
            tp += ((preds == 1) & (batch_y == 1)).sum().item()
            fp += ((preds == 1) & (batch_y == 0)).sum().item()
            fn += ((preds == 0) & (batch_y == 1)).sum().item()
            tn += ((preds == 0) & (batch_y == 0)).sum().item()

    avg_loss = total_loss / total
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, accuracy, precision, recall, f1


def train(
    mel_dir: str,
    segments_dir: str,
    metadata_path: str,
    output_dir: str = "models",
    model_type: str = "default",
    batch_size: int = 64,
    context_frames: int = 64,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    samples_per_epoch: int = 50000,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
):
    """
    Train the boundary detection model.

    Args:
        mel_dir: Directory containing mel spectrogram files
        segments_dir: Directory containing segment annotation files
        metadata_path: Path to metadata CSV
        output_dir: Directory to save model checkpoints
        model_type: 'default' or 'small'
        batch_size: Training batch size
        context_frames: Context frames for each sample
        epochs: Maximum training epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience
        seed: Random seed
        samples_per_epoch: Maximum training samples per epoch
        loss_type: 'bce' or 'focal'
        focal_gamma: Gamma parameter for focal loss
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    start_time = time.time()
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        mel_dir, segments_dir, metadata_path,
        batch_size=batch_size,
        context_frames=context_frames,
        seed=seed,
        samples_per_epoch=samples_per_epoch,
    )
    print(f"Data loaded in {time.time() - start_time:.1f}s")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = get_model(model_type, context_frames=context_frames)
    model = model.to(device)
    print(f"\nModel: {model_type} ({count_parameters(model):,} parameters)")

    # Loss function
    pos_weight = class_weights.to(device)
    if loss_type == "focal":
        # For focal loss, alpha is the weight for positive class
        # Convert pos_weight to alpha: alpha = pos_weight / (1 + pos_weight)
        alpha = pos_weight.item() / (1 + pos_weight.item())
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
        print(f"Using Focal Loss (alpha={alpha:.3f}, gamma={focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCE Loss (pos_weight={pos_weight.item():.2f})")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    print(f"\nTraining for up to {epochs} epochs...")
    print("-" * 70)

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"P: {val_prec:.3f} R: {val_recall:.3f} | "
              f"Time: {epoch_time:.1f}s")

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  -> Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'model_type': model_type,
                'context_frames': context_frames,
                'loss_type': loss_type,
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> New best model saved (F1: {val_f1:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {patience} epochs)")
            break

    total_time = time.time() - training_start
    print("-" * 70)
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Best validation F1: {best_val_f1:.4f}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train boundary detection model")
    parser.add_argument("--mel-dir", default="data/harmonix/melspecs",
                        help="Directory containing mel spectrograms")
    parser.add_argument("--segments-dir", default="data/harmonix/harmonixset/dataset/segments",
                        help="Directory containing segment annotations")
    parser.add_argument("--metadata", default="data/harmonix/harmonixset/dataset/metadata.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--output-dir", default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--model-type", choices=["default", "small"], default="default",
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--context-frames", type=int, default=64,
                        help="Context frames on each side")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--samples-per-epoch", type=int, default=50000,
                        help="Max training samples per epoch (for faster training)")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce",
                        help="Loss function: bce or focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")

    args = parser.parse_args()

    train(
        mel_dir=args.mel_dir,
        segments_dir=args.segments_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        samples_per_epoch=args.samples_per_epoch,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
    )


if __name__ == "__main__":
    main()
