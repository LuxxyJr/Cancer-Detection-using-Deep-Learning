"""
LUNA16 CT scan loading utilities
"""

import numpy as np
import SimpleITK as sitk
import torch


def load_mhd(path):
    """Load a LUNA16 .mhd scan, returns (volume, spacing, origin)"""
    image = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    spacing = image.GetSpacing()            # (X, Y, Z) in mm
    origin = image.GetOrigin()              # (X, Y, Z) in mm
    return volume, spacing, origin


def normalize_ct(volume):
    """HU normalization: clip to lung window [-1000, 400], scale to [0, 1]"""
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400
    return volume.astype(np.float32)


def to_tensor(volume):
    """Convert numpy volume to torch tensor with channel dim: (1, D, H, W)"""
    return torch.from_numpy(volume).unsqueeze(0)


def extract_patch(volume, center=None, patch_size=(64, 96, 96)):
    """
    Extract a 3D patch from a volume.

    Args:
        volume: numpy array (D, H, W)
        center: (z, y, x) voxel coordinates for patch center, or None for random
        patch_size: (pd, ph, pw) patch dimensions

    Returns:
        numpy array of shape patch_size, zero-padded if volume is too small
    """
    import random

    d, h, w = volume.shape
    pd, ph, pw = patch_size

    if center is not None:
        cz, cy, cx = center
        z = max(0, min(cz - pd // 2, d - pd))
        y = max(0, min(cy - ph // 2, h - ph))
        x = max(0, min(cx - pw // 2, w - pw))
    else:
        z = random.randint(0, max(0, d - pd))
        y = random.randint(0, max(0, h - ph))
        x = random.randint(0, max(0, w - pw))

    patch = volume[z:z + pd, y:y + ph, x:x + pw]

    # Zero-pad if volume is smaller than patch size
    if patch.shape != (pd, ph, pw):
        padded = np.zeros((pd, ph, pw), dtype=volume.dtype)
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded

    return patch
