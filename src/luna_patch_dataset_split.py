"""
LUNA16 Patch Dataset with REAL labels from candidates.csv
Extracts 3D patches centered on candidate nodule locations
Handles class imbalance via negative subsampling
Includes data augmentation and volume caching
"""

import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class LunaPatchDatasetSplit(Dataset):
    """
    LUNA16 candidate-based dataset with real ground truth labels.

    Each sample is a 3D patch centered on a candidate location from
    candidates.csv, with the label (0=non-nodule, 1=nodule) from that CSV.
    """

    def __init__(self, candidates, uid_to_path, patch_size=(64, 96, 96),
                 neg_ratio=5, augment=False, cache_size=10):
        """
        Args:
            candidates: List of dicts {seriesuid, coordX, coordY, coordZ, label}
            uid_to_path: Dict mapping seriesuid -> .mhd file path
            patch_size: (D, H, W) patch dimensions
            neg_ratio: Max negatives per positive to keep (for balancing)
            augment: Apply random augmentation (True for training only)
            cache_size: Number of CT volumes to cache in memory
        """
        self.patch_size = patch_size
        self.uid_to_path = uid_to_path
        self.augment = augment
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []

        # Filter to candidates that have a matching .mhd file
        valid = [c for c in candidates if c["seriesuid"] in uid_to_path]

        # Separate positives and negatives
        positives = [c for c in valid if c["label"] == 1]
        negatives = [c for c in valid if c["label"] == 0]

        # Balance: keep ALL positives, subsample negatives
        n_neg = min(len(negatives), len(positives) * neg_ratio)
        rng = random.Random(42)
        sampled_negatives = rng.sample(negatives, n_neg) if n_neg > 0 else []

        self.samples = positives + sampled_negatives
        rng.shuffle(self.samples)

        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        n_neg = len(self.samples) - n_pos
        print(f"  Dataset: {len(self.samples)} samples "
              f"({n_pos} positive, {n_neg} negative, "
              f"ratio 1:{n_neg // max(n_pos, 1)})")

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, seriesuid):
        """Load and cache a CT volume"""
        if seriesuid in self._cache:
            return self._cache[seriesuid]

        path = self.uid_to_path[seriesuid]
        img = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X)
        origin = img.GetOrigin()            # order: (X, Y, Z) in mm
        spacing = img.GetSpacing()          # order: (X, Y, Z) in mm

        # HU normalization: clip to lung window, scale to [0, 1]
        vol = np.clip(vol, -1000, 400).astype(np.float32)
        vol = (vol + 1000.0) / 1400.0

        entry = {"volume": vol, "origin": origin, "spacing": spacing}

        # Evict oldest if cache is full
        if len(self._cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[seriesuid] = entry
        self._cache_order.append(seriesuid)

        return entry

    def _world_to_voxel(self, coord_x, coord_y, coord_z, origin, spacing):
        """Convert world coordinates (mm) to voxel indices (Z, Y, X)"""
        vx = int(round((coord_x - origin[0]) / spacing[0]))
        vy = int(round((coord_y - origin[1]) / spacing[1]))
        vz = int(round((coord_z - origin[2]) / spacing[2]))
        return vz, vy, vx  # numpy array is indexed (Z, Y, X)

    def _extract_centered_patch(self, vol, cz, cy, cx):
        """Extract patch centered on (cz, cy, cx) with boundary handling"""
        pd, ph, pw = self.patch_size
        d, h, w = vol.shape

        # Calculate start indices, clamped to valid range
        z0 = max(0, min(cz - pd // 2, d - pd))
        y0 = max(0, min(cy - ph // 2, h - ph))
        x0 = max(0, min(cx - pw // 2, w - pw))

        patch = vol[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]

        # Pad if volume is smaller than patch size in any dimension
        if patch.shape != (pd, ph, pw):
            padded = np.zeros((pd, ph, pw), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded

        return patch

    def _augment_patch(self, patch):
        """Apply random 3D augmentations"""
        # Random flips along each spatial axis
        if random.random() > 0.5:
            patch = np.flip(patch, axis=0).copy()  # Z (axial)
        if random.random() > 0.5:
            patch = np.flip(patch, axis=1).copy()  # Y (coronal)
        if random.random() > 0.5:
            patch = np.flip(patch, axis=2).copy()  # X (sagittal)

        # Random intensity shift
        if random.random() > 0.5:
            shift = random.uniform(-0.05, 0.05)
            patch = np.clip(patch + shift, 0.0, 1.0)

        # Random intensity scale
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)
            patch = np.clip(patch * scale, 0.0, 1.0)

        return patch

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load volume (with caching)
        data = self._load_volume(sample["seriesuid"])
        vol = data["volume"]
        origin = data["origin"]
        spacing = data["spacing"]

        # Convert world coordinates to voxel indices
        vz, vy, vx = self._world_to_voxel(
            sample["coordX"], sample["coordY"], sample["coordZ"],
            origin, spacing
        )

        # Extract patch centered on candidate location
        patch = self._extract_centered_patch(vol, vz, vy, vx)

        # Augmentation (training only)
        if self.augment:
            patch = self._augment_patch(patch)

        # Convert to tensor: (1, D, H, W)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return patch_tensor, label
