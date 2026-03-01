"""
Fast dataset that loads pre-extracted .npy patches from disk.

Requires running preextract.py first to create the patch files.
Each patch is a small .npy file (~2.4 MB) that loads in milliseconds,
eliminating the CT-scan I/O bottleneck entirely.

Supports two modes:
  1. Standalone: reads manifest.csv from a directory (original single-split mode)
  2. K-fold CV:  receives pre-filtered samples list from the training script
"""

import os
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class FastPatchDataset(Dataset):
    """
    Loads pre-extracted 3D patches from .npy files on disk.

    Expects patches saved by preextract.py as individual .npy files.
    """

    def __init__(self, patches_dir, augment=False, samples=None, name=None):
        """
        Args:
            patches_dir: Path to directory containing .npy patch files
            augment: Apply random 3D augmentations (True for training only)
            samples: Optional list of dicts with 'filename' and 'label' keys.
                     If provided, uses these instead of reading manifest.csv.
                     Used by k-fold CV to pass pre-filtered patient subsets.
            name: Display name for logging (e.g. "train", "val", "test").
                  If None, uses the directory basename.
        """
        self.patches_dir = patches_dir
        self.augment = augment

        if samples is not None:
            # K-fold mode: use pre-filtered samples
            self.samples = samples
        else:
            # Standalone mode: read manifest.csv from patches_dir
            manifest_path = os.path.join(patches_dir, "manifest.csv")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"Manifest not found: {manifest_path}\n"
                    f"Run preextract.py first to create patch files."
                )

            self.samples = []
            with open(manifest_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append({
                        "filename": row["filename"],
                        "label": int(row["label"]),
                    })

        # Print stats
        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        n_neg = len(self.samples) - n_pos
        ratio = n_neg // max(n_pos, 1)
        display_name = name or os.path.basename(patches_dir)
        print(f"    {display_name:5s}: {len(self.samples):5d} patches "
              f"({n_pos} nodule, {n_neg} non-nodule, ratio 1:{ratio})")

    def __len__(self):
        return len(self.samples)

    def _augment_patch(self, patch):
        """
        Apply random 3D augmentations (medical imaging standard).

        Geometric: 90-degree rotations in axial plane + flips along all axes.
        Photometric: Gaussian noise, intensity shift, intensity scale.
        """
        # Random 90-degree rotation around a random axis
        # All 3 planes are safe because patch is cubic (D=H=W=64)
        if random.random() > 0.5:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            axis = random.choice([(0, 1), (0, 2), (1, 2)])
            patch = np.rot90(patch, k=k, axes=axis).copy()

        # Random flips along each spatial axis
        if random.random() > 0.5:
            patch = np.flip(patch, axis=0).copy()  # Z (axial)
        if random.random() > 0.5:
            patch = np.flip(patch, axis=1).copy()  # Y (coronal)
        if random.random() > 0.5:
            patch = np.flip(patch, axis=2).copy()  # X (sagittal)

        # Additive Gaussian noise (simulates scanner noise variation)
        if random.random() > 0.5:
            sigma = random.uniform(0.01, 0.03)
            noise = np.random.normal(0, sigma, patch.shape).astype(np.float32)
            patch = np.clip(patch + noise, 0.0, 1.0)

        # Random intensity shift (simulates contrast variation)
        if random.random() > 0.5:
            shift = random.uniform(-0.05, 0.05)
            patch = np.clip(patch + shift, 0.0, 1.0)

        # Random intensity scale (simulates brightness variation)
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)
            patch = np.clip(patch * scale, 0.0, 1.0)

        return patch

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load pre-extracted patch (fast -- just a small .npy file)
        path = os.path.join(self.patches_dir, sample["filename"])
        patch = np.load(path)  # shape: (D, H, W), dtype: float32

        # Augmentation (training only)
        if self.augment:
            patch = self._augment_patch(patch)

        # Convert to tensor: (1, D, H, W) -- single channel CT
        patch_tensor = torch.from_numpy(patch.copy()).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return patch_tensor, label
