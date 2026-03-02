"""
MSD TASK03 LIVER TUMOR - PATCH PRE-EXTRACTION (run this ONCE before training)

Loads each CT volume + segmentation mask, resamples to isotropic 1mm spacing,
extracts tumor-centered patches (positive) and liver-only patches (negative),
saves as .npy files.

Output format is identical to the lung pipeline (manifest.csv + .npy patches),
so the same training code (main_liver.py) works without changes.

Data format:
  Medical Segmentation Decathlon Task03_Liver provides NIfTI files:
    - imagesTr/liver_{id}.nii.gz   CT scan (131 training volumes)
    - labelsTr/liver_{id}.nii.gz   Mask (0=bg, 1=liver, 2=tumor)

  Download: http://medicaldecathlon.com/dataaws/
  Extract Task03_Liver.tar into data/Task03_Liver/

Usage:  python preextract_liver.py
Time:   ~30-60 minutes
Disk:   ~10-20 GB in data/liver_patches/
"""

import os
import re
import csv
import random
import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage

from utils import set_seed

# ── Config ────────────────────────────────────────────────────────────
SEED = 42
NEG_RATIO = 5                      # Negatives per positive
PATCH_SIZE = (96, 96, 96)          # Larger cubic patch for liver (tumors are bigger than lung nodules)
TARGET_SPACING = (1.0, 1.0, 1.0)  # Isotropic resampling in mm (Z, Y, X)
MIN_TUMOR_VOXELS = 10              # Skip tumor fragments smaller than this
MAX_PATCHES_PER_TUMOR = 3          # Max patches per tumor (centroid + random offsets)
LARGE_TUMOR_THRESHOLD = 1000       # Tumors larger than this (in voxels) get extra patches

# HU windowing for abdominal CT (liver-optimized)
# Liver parenchyma: ~40-70 HU, tumors: variable (20-100+ HU)
# Range [-200, 300] captures all relevant soft tissue contrast
HU_MIN = -200.0
HU_MAX = 300.0

# Paths (relative to project root -- works on both Windows and Linux)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSD_LIVER_FOLDER = os.path.join(PROJECT_ROOT, "data", "Task03_Liver")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "liver_patches")
# ──────────────────────────────────────────────────────────────────────


def find_volume_pairs(msd_folder):
    """
    Find all image-label pairs in MSD Task03_Liver folder.

    Expected structure:
      Task03_Liver/
        imagesTr/liver_0.nii.gz, liver_1.nii.gz, ...
        labelsTr/liver_0.nii.gz, liver_1.nii.gz, ...

    Returns:
        dict: {volume_id: {"volume": path, "segmentation": path}}
    """
    pairs = {}
    images_dir = os.path.join(msd_folder, "imagesTr")
    labels_dir = os.path.join(msd_folder, "labelsTr")

    if not os.path.isdir(images_dir):
        print(f"  ERROR: imagesTr directory not found at {images_dir}")
        return {}
    if not os.path.isdir(labels_dir):
        print(f"  ERROR: labelsTr directory not found at {labels_dir}")
        return {}

    for fname in os.listdir(images_dir):
        match = re.match(r"liver_(\d+)\.nii(?:\.gz)?$", fname)
        if match:
            vid = int(match.group(1))
            pairs[vid] = {"volume": os.path.join(images_dir, fname)}

    # Match with label files
    complete = {}
    for vid, paths in sorted(pairs.items()):
        # Try both .nii.gz and .nii
        label_found = False
        for ext in [".nii.gz", ".nii"]:
            label_path = os.path.join(labels_dir, f"liver_{vid}{ext}")
            if os.path.isfile(label_path):
                paths["segmentation"] = label_path
                complete[vid] = paths
                label_found = True
                break

        if not label_found:
            print(f"  WARNING: Volume {vid} has no matching label file, skipping")

    return complete


def load_and_resample(vol_path, mask_path):
    """
    Load CT volume + segmentation mask, resample both to isotropic spacing.

    Volume is resampled with linear interpolation (order=1).
    Mask is resampled with nearest-neighbor (order=0) to preserve labels.

    Returns:
        vol:  resampled volume, normalized to [0,1] (float32, shape Z,Y,X)
        mask: resampled mask (int8, labels 0/1/2, shape Z,Y,X)
    """
    # Load volume
    img = sitk.ReadImage(vol_path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    spacing = img.GetSpacing()  # (X, Y, Z) -- SimpleITK convention

    # Load mask
    mask_img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask_img).astype(np.int8)  # (Z, Y, X)

    # Compute zoom factors for resampling
    # spacing is (X, Y, Z), volume axes are (Z, Y, X) -- convert to match
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    zoom_factors = tuple(s / t for s, t in zip(spacing_zyx, TARGET_SPACING))

    if not all(abs(f - 1.0) < 0.01 for f in zoom_factors):
        # Resample volume (linear interpolation)
        vol = np.clip(vol, HU_MIN, HU_MAX)
        vol = scipy.ndimage.zoom(vol, zoom_factors, order=1).astype(np.float32)

        # Resample mask (nearest-neighbor to preserve label integrity)
        mask = scipy.ndimage.zoom(mask, zoom_factors, order=0).astype(np.int8)
    else:
        vol = np.clip(vol, HU_MIN, HU_MAX)

    # Normalize volume to [0, 1]
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)

    return vol, mask


def find_tumor_components(mask, min_voxels=10):
    """
    Find connected tumor components and their centroids.

    Args:
        mask: 3D array (Z, Y, X) with labels 0/1/2
        min_voxels: Skip components smaller than this

    Returns:
        list of dicts: [{"centroid": (z,y,x), "n_voxels": int, "id": int}, ...]
        labeled_array: labeled tumor mask (for finding voxels per component)
    """
    tumor_mask = (mask == 2).astype(np.int8)
    labeled, n_components = scipy.ndimage.label(tumor_mask)

    components = []
    for i in range(1, n_components + 1):
        component = (labeled == i)
        n_voxels = int(component.sum())

        if n_voxels < min_voxels:
            continue

        centroid = scipy.ndimage.center_of_mass(component)
        components.append({
            "centroid": (int(round(centroid[0])),
                         int(round(centroid[1])),
                         int(round(centroid[2]))),
            "n_voxels": n_voxels,
            "id": i,
        })

    return components, labeled


def get_positive_positions(components, labeled_tumors, max_per_tumor=3,
                           large_threshold=1000, rng=None):
    """
    Get patch center positions for positive (tumor) patches.

    For each tumor:
      - Always includes the centroid
      - For large tumors (>large_threshold voxels): adds random offsets within tumor

    Returns:
        list of (z, y, x) tuples
    """
    if rng is None:
        rng = random.Random(42)

    positions = []

    for comp in components:
        cz, cy, cx = comp["centroid"]
        positions.append((cz, cy, cx))

        # Extra patches for large tumors
        if comp["n_voxels"] > large_threshold and max_per_tumor > 1:
            tumor_voxels = np.argwhere(labeled_tumors == comp["id"])
            n_extra = min(max_per_tumor - 1, 2)

            if len(tumor_voxels) > n_extra:
                indices = rng.sample(range(len(tumor_voxels)), n_extra)
                for idx in indices:
                    pos = tumor_voxels[idx]
                    positions.append((int(pos[0]), int(pos[1]), int(pos[2])))

    return positions


def sample_negative_positions(mask, n_neg, patch_size, rng=None):
    """
    Sample positions for negative (liver-only) patches.

    Samples random liver voxels (mask==1) and verifies the full patch
    region contains no tumor voxels (mask==2). This guarantees clean negatives.

    Returns:
        list of (z, y, x) tuples
    """
    if rng is None:
        rng = random.Random(42)

    pd, ph, pw = patch_size
    vol_shape = mask.shape

    # Get all liver-only voxels (label 1, not tumor)
    liver_coords = np.argwhere(mask == 1)

    if len(liver_coords) == 0:
        return []

    # Filter to voxels that can be valid patch centers (enough margin from edges)
    margin_z, margin_y, margin_x = pd // 2, ph // 2, pw // 2

    valid = (
        (liver_coords[:, 0] >= margin_z) &
        (liver_coords[:, 0] < vol_shape[0] - margin_z) &
        (liver_coords[:, 1] >= margin_y) &
        (liver_coords[:, 1] < vol_shape[1] - margin_y) &
        (liver_coords[:, 2] >= margin_x) &
        (liver_coords[:, 2] < vol_shape[2] - margin_x)
    )
    valid_coords = liver_coords[valid]

    if len(valid_coords) == 0:
        return []

    # Sample candidate positions (oversample then filter)
    n_candidates = min(len(valid_coords), n_neg * 20)
    np_rng = np.random.RandomState(rng.randint(0, 2**31))
    candidate_indices = np_rng.choice(len(valid_coords), size=n_candidates,
                                      replace=False)

    positions = []
    for idx in candidate_indices:
        if len(positions) >= n_neg:
            break

        coord = valid_coords[idx]
        cz, cy, cx = int(coord[0]), int(coord[1]), int(coord[2])

        # Check patch region for tumor voxels
        z0 = max(0, cz - pd // 2)
        y0 = max(0, cy - ph // 2)
        x0 = max(0, cx - pw // 2)

        sub_mask = mask[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
        if np.any(sub_mask == 2):
            continue  # Patch contains tumor -- reject

        positions.append((cz, cy, cx))

    return positions


def extract_centered_patch(vol, cz, cy, cx, patch_size):
    """Extract patch centered on (cz, cy, cx) with boundary/padding handling"""
    pd, ph, pw = patch_size
    d, h, w = vol.shape

    z0 = max(0, min(cz - pd // 2, d - pd))
    y0 = max(0, min(cy - ph // 2, h - ph))
    x0 = max(0, min(cx - pw // 2, w - pw))

    patch = vol[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]

    if patch.shape != (pd, ph, pw):
        padded = np.zeros((pd, ph, pw), dtype=np.float32)
        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = padded

    return patch


def main():
    set_seed(SEED)
    start = time.time()
    rng = random.Random(SEED)

    print("\n" + "=" * 70)
    print("  MSD TASK03 LIVER TUMOR - PATCH PRE-EXTRACTION")
    print("  Run this ONCE. Then run main_liver.py for training.")
    print("=" * 70 + "\n")

    # Find volume-segmentation pairs
    pairs = find_volume_pairs(MSD_LIVER_FOLDER)
    print(f"  Found {len(pairs)} volume-segmentation pairs in {MSD_LIVER_FOLDER}")
    print(f"  Resampling to:   {TARGET_SPACING[0]:.0f}x{TARGET_SPACING[1]:.0f}x{TARGET_SPACING[2]:.0f} mm isotropic spacing")
    print(f"  Patch size:      {PATCH_SIZE[0]}x{PATCH_SIZE[1]}x{PATCH_SIZE[2]}")
    print(f"  HU window:       [{HU_MIN:.0f}, {HU_MAX:.0f}]")

    if len(pairs) == 0:
        print("\n  ERROR: No volume-segmentation pairs found!")
        print(f"  Expected structure:")
        print(f"    {MSD_LIVER_FOLDER}/imagesTr/liver_0.nii.gz")
        print(f"    {MSD_LIVER_FOLDER}/labelsTr/liver_0.nii.gz")
        print(f"  Download Task03_Liver.tar from http://medicaldecathlon.com/dataaws/")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = []
    patch_idx = 0
    total_scans = len(pairs)
    total_tumors = 0
    total_pos_patches = 0
    total_neg_patches = 0
    skipped_small = 0
    t0 = time.time()

    for scan_num, (vid, paths) in enumerate(sorted(pairs.items())):
        # Load and resample volume + mask
        vol, mask = load_and_resample(paths["volume"], paths["segmentation"])

        # Find tumor connected components
        components, labeled_tumors = find_tumor_components(
            mask, min_voxels=MIN_TUMOR_VOXELS
        )
        total_tumors += len(components)

        # Count skipped small tumors
        tumor_mask_raw = (mask == 2).astype(np.int8)
        _, n_raw = scipy.ndimage.label(tumor_mask_raw)
        skipped_small += n_raw - len(components)

        # Get positive patch positions
        pos_positions = get_positive_positions(
            components, labeled_tumors,
            max_per_tumor=MAX_PATCHES_PER_TUMOR,
            large_threshold=LARGE_TUMOR_THRESHOLD,
            rng=rng,
        )

        # Extract positive patches
        for cz, cy, cx in pos_positions:
            patch = extract_centered_patch(vol, cz, cy, cx, PATCH_SIZE)
            filename = f"patch_{patch_idx:06d}.npy"
            np.save(os.path.join(OUTPUT_DIR, filename), patch)
            manifest.append({
                "filename": filename,
                "label": 1,
                "seriesuid": f"liver_{vid}",
            })
            patch_idx += 1
            total_pos_patches += 1

        # Sample and extract negative patches (liver-only, no tumor in patch)
        n_neg = len(pos_positions) * NEG_RATIO
        if n_neg > 0:
            neg_positions = sample_negative_positions(
                mask, n_neg, PATCH_SIZE, rng=rng
            )

            for cz, cy, cx in neg_positions:
                patch = extract_centered_patch(vol, cz, cy, cx, PATCH_SIZE)
                filename = f"patch_{patch_idx:06d}.npy"
                np.save(os.path.join(OUTPUT_DIR, filename), patch)
                manifest.append({
                    "filename": filename,
                    "label": 0,
                    "seriesuid": f"liver_{vid}",
                })
                patch_idx += 1
                total_neg_patches += 1

        # Progress reporting
        if (scan_num + 1) % 10 == 0 or scan_num == 0 or (scan_num + 1) == total_scans:
            elapsed = time.time() - t0
            rate = (scan_num + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_scans - scan_num - 1) / rate if rate > 0 else 0
            print(f"  {scan_num+1}/{total_scans} volumes | "
                  f"tumors: {len(components)} | "
                  f"+{total_pos_patches} -{total_neg_patches} patches | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    # Save manifest CSV
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "seriesuid"])
        writer.writeheader()
        writer.writerows(manifest)

    elapsed = time.time() - start
    disk_gb = (patch_idx * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * 4) / 1e9
    unique_patients = len(set(m["seriesuid"] for m in manifest))

    print(f"\n" + "=" * 70)
    print(f"  EXTRACTION COMPLETE")
    print(f"  Total volumes:     {total_scans}")
    print(f"  Total tumors:      {total_tumors} ({skipped_small} small fragments skipped)")
    print(f"  Unique patients:   {unique_patients}")
    print(f"  Total patches:     {patch_idx} "
          f"({total_pos_patches} tumor, {total_neg_patches} liver-only)")
    print(f"  Ratio:             1:{total_neg_patches // max(total_pos_patches, 1)}")
    print(f"  Manifest:          {manifest_path}")
    print(f"  Disk usage:        ~{disk_gb:.1f} GB")
    print(f"  Total time:        {elapsed/60:.1f} minutes")
    print(f"\n  Next step:  python main_liver.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
