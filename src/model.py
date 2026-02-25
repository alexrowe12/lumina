"""
CNN model for music section boundary detection.

Takes mel spectrogram patches and outputs boundary probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryDetectorCNN(nn.Module):
    """
    CNN for detecting section boundaries in mel spectrograms.

    Architecture:
    - 3 conv blocks with batch norm and max pooling
    - Global average pooling
    - Dense layers with dropout
    - Sigmoid output for boundary probability
    """

    def __init__(self, dropout: float = 0.3, **kwargs):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))

        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        # Conv block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Mel spectrogram patch of shape (batch, 1, mel_bins, time)

        Returns:
            Boundary probability of shape (batch,)
        """
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x.squeeze(-1)


class BoundaryDetectorCNNSmall(nn.Module):
    """Smaller/faster CNN variant for quicker training."""

    def __init__(self, dropout: float = 0.3, **kwargs):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))

        # Conv block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))

        # Conv block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense layers
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)


def get_model(model_type: str = "default", dropout: float = 0.3, **kwargs) -> nn.Module:
    """
    Factory function to create model.

    Args:
        model_type: 'default' or 'small'
        dropout: Dropout probability

    Returns:
        Model instance
    """
    if model_type == "small":
        return BoundaryDetectorCNNSmall(dropout=dropout)
    else:
        return BoundaryDetectorCNN(dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = get_model("default")
    print(f"Default model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 1, 80, 129)  # batch=4, channels=1, mel_bins=80, time=129
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test small model
    model_small = get_model("small")
    print(f"Small model parameters: {count_parameters(model_small):,}")
