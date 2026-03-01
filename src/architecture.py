"""
Multi-Organ Cancer Detection Framework
Classification-focused 3D CNN Architecture

Optimized for RTX 4050 (6GB VRAM):
  - Removed segmentation head (saves ~25% VRAM)
  - Removed aggressive early downsampling (preserves small nodule detail)
  - Binary classification (2 classes: nodule vs non-nodule)
  - Higher dropout (0.5) for regularization with limited data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """Conv3D + InstanceNorm + ReLU"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    """3D residual block with optional downsampling"""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = Conv3DBlock(in_c, out_c, stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride, bias=False),
                nn.InstanceNorm3d(out_c),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class VolumetricEncoder(nn.Module):
    """
    3D ResNet-style encoder.

    Input:  (B, 1, D, H, W)  -- e.g. (B, 1, 64, 64, 64) for cubic patches
    Flow:   initial(stride=1) -> b1 -> b2(stride=2) -> b3(stride=2) -> b4(stride=2)
    Output: (B, 256, D/8, H/8, W/8)

    No MaxPool after initial conv -- preserves spatial resolution for
    small nodule detection (3-6mm nodules are only a few voxels).
    """

    def __init__(self, in_channels=1):
        super().__init__()
        self.initial = Conv3DBlock(in_channels, 32, stride=1)
        self.b1 = ResidualBlock3D(32, 32)
        self.b2 = ResidualBlock3D(32, 64, stride=2)
        self.b3 = ResidualBlock3D(64, 128, stride=2)
        self.b4 = ResidualBlock3D(128, 256, stride=2)

    def forward(self, x):
        x = self.initial(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return x


class ClassificationHead(nn.Module):
    """Global average pooling -> FC layers -> class logits"""

    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


class LungNoduleClassifier(nn.Module):
    """
    3D CNN for LUNA16 nodule classification.
    Input:  (B, 1, D, H, W) CT patch -- e.g. (B, 1, 64, 64, 64)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = VolumetricEncoder()
        self.classifier = ClassificationHead(num_classes=num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


# ======================================================================
# BASELINE MODEL: Simple 3D VGG-style network
# ======================================================================

class SimpleVGG3D(nn.Module):
    """
    Lightweight 3D VGG-style classifier for baseline comparison.

    Architecture: 3 conv blocks (each: Conv3d -> IN -> ReLU -> Conv3d -> IN -> ReLU -> MaxPool)
    followed by global average pooling and a small FC head.

    Input:  (B, 1, D, H, W)  -- e.g. (B, 1, 64, 64, 64)
    Flow:   block1(32) -> pool -> block2(64) -> pool -> block3(128) -> pool -> GAP -> FC
    Output: (B, num_classes)

    ~250K parameters (10x smaller than ResNet). Serves as a simpler
    baseline to demonstrate that the ResNet architecture adds value.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ======================================================================
# MODEL FACTORY
# ======================================================================

def get_model(name, num_classes=2):
    """
    Create a model by name.

    Args:
        name: "resnet3d" or "vgg3d"
        num_classes: Number of output classes (default 2)

    Returns:
        nn.Module instance
    """
    if name == "resnet3d":
        return LungNoduleClassifier(num_classes=num_classes)
    elif name == "vgg3d":
        return SimpleVGG3D(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'resnet3d' or 'vgg3d'.")
