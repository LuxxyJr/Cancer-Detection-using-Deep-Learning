"""
PATCH PRE-EXTRACTION SCRIPT (run this ONCE before training)

Loads each CT scan once, resamples to isotropic 1mm spacing,
extracts all candidate patches, saves as .npy files.
This eliminates the I/O bottleneck during training.

Extracts ALL positives + balanced negatives into a single directory.
Patient-level splitting for 3-fold CV happens at training time (in main.py).

Usage:  python preextract.py
Time:   ~20-40 minutes (includes resampling)
Disk:   ~8-10 GB in data/patches/
"""

import os
import csv
import random
import time
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import scipy.ndimage

from utils import set_seed

# ── Config ────────────────────────────────────────────────────────────
SEED = 42
NEG_RATIO = 5
PATCH_SIZE = (64, 64, 64)   # Cube -- enables full 3D rotation augmentation
TARGET_SPACING = (1.0, 1.0, 1.0)  # Isotropic resampling in mm (Z, Y, X)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUNA_FOLDER = os.path.join(PROJECT_ROOT, "data", "LUNA16")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "patches")
# ──────────────────────────────────────────────────────────────────────


def find_mhd_files(luna_folder):
    """Find all .mhd files in subset folders, return seriesuid -> path mapping"""
    uid_to_path = {}
    for subset in sorted(os.listdir(luna_folder)):
        subset_dir = os.path.join(luna_folder, subset)
        if not os.path.isdir(subset_dir) or not subset.startswith("subset"):
            continue
        for fname in os.listdir(subset_dir):
            if fname.endswith(".mhd"):
                uid = fname.replace(".mhd", "")
                uid_to_path[uid] = os.path.join(subset_dir, fname)
    return uid_to_path


def read_candidates(luna_folder):
    """Read candidates.csv and return list of candidate dicts"""
    candidates = []
    csv_path = os.path.join(luna_folder, "candidates.csv")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append({
                "seriesuid": row["seriesuid"],
                "coordX": float(row["coordX"]),
                "coordY": float(row["coordY"]),
                "coordZ": float(row["coordZ"]),
                "label": int(row["class"]),
            })
    return candidates


def balance_candidates(candidates, neg_ratio, seed=42):
    """Keep all positives, subsample negatives to neg_ratio:1"""
    rng = random.Random(seed)
    positives = [c for c in candidates if c["label"] == 1]
    negatives = [c for c in candidates if c["label"] == 0]
    n_neg = min(len(negatives), len(positives) * neg_ratio)
    sampled_neg = rng.sample(negatives, n_neg) if n_neg > 0 else []
    result = positives + sampled_neg
    rng.shuffle(result)
    return result


def load_and_normalize(path):
    """Load .mhd CT scan, resample to isotropic spacing, normalize HU to [0, 1]

    Isotropic resampling is CRITICAL for 3D CNNs:
    - LUNA16 scans have varying Z-spacing (0.625mm to 2.5mm+)
    - A 64x96x96 patch covers 40-160mm in Z depending on scan spacing
    - Without resampling, the same nodule appears at wildly different scales
    - The model cannot learn consistent features across such variation
    """
    img = sitk.ReadImage(path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    origin = img.GetOrigin()            # (X, Y, Z) -- SimpleITK convention
    spacing = img.GetSpacing()          # (X, Y, Z) -- SimpleITK convention

    # Resample to isotropic target spacing
    # spacing is (X, Y, Z), volume axes are (Z, Y, X) -- convert to match
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    zoom_factors = tuple(s / t for s, t in zip(spacing_zyx, TARGET_SPACING))

    if not all(abs(f - 1.0) < 0.01 for f in zoom_factors):
        # Clip HU before resampling to prevent interpolation artifacts at extremes
        vol = np.clip(vol, -1000.0, 400.0)
        vol = scipy.ndimage.zoom(vol, zoom_factors, order=1).astype(np.float32)
    else:
        vol = np.clip(vol, -1000.0, 400.0)

    # Normalize to [0, 1]
    vol = (vol + 1000.0) / 1400.0

    # After resampling, use target spacing for world-to-voxel conversion
    # Origin is unchanged (physical position of voxel [0,0,0] is the same)
    resampled_spacing = (TARGET_SPACING[2], TARGET_SPACING[1], TARGET_SPACING[0])  # XYZ

    return vol, origin, resampled_spacing


def world_to_voxel(cx, cy, cz, origin, spacing):
    """Convert world mm coords to voxel indices (Z, Y, X)"""
    vx = int(round((cx - origin[0]) / spacing[0]))
    vy = int(round((cy - origin[1]) / spacing[1]))
    vz = int(round((cz - origin[2]) / spacing[2]))
    return vz, vy, vx


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

    print("\n" + "=" * 70)
    print("  PATCH PRE-EXTRACTION (for 3-fold cross-validation)")
    print("  Run this ONCE. Then run main.py for training.")
    print("=" * 70 + "\n")

    # Find scans
    uid_to_path = find_mhd_files(LUNA_FOLDER)
    print(f"  Found {len(uid_to_path)} CT scans in {LUNA_FOLDER}")
    print(f"  Resampling to:   {TARGET_SPACING[0]:.0f}x{TARGET_SPACING[1]:.0f}x{TARGET_SPACING[2]:.0f} mm isotropic spacing")

    # Read and filter candidates
    all_candidates = read_candidates(LUNA_FOLDER)
    valid = [c for c in all_candidates if c["seriesuid"] in uid_to_path]
    n_total_pos = sum(1 for c in valid if c["label"] == 1)
    n_total_neg = sum(1 for c in valid if c["label"] == 0)
    print(f"  Candidates with matching scans: {len(valid)} "
          f"({n_total_pos} nodule, {n_total_neg} non-nodule)")

    # Balance globally
    balanced = balance_candidates(valid, NEG_RATIO, SEED)
    n_pos = sum(1 for c in balanced if c["label"] == 1)
    n_neg = len(balanced) - n_pos
    print(f"\n  After balancing (neg_ratio={NEG_RATIO}):")
    print(f"    Positives (nodule):     {n_pos}")
    print(f"    Negatives (non-nodule): {n_neg}")
    print(f"    Ratio:                  1:{n_neg // max(n_pos, 1)}")

    # Group by seriesuid for efficient extraction (load each scan once)
    groups = defaultdict(list)
    for c in balanced:
        groups[c["seriesuid"]].append(c)

    unique_uids = sorted(groups.keys())
    print(f"    Unique patients:        {len(unique_uids)}")
    print(f"\n  Saving patches to: {OUTPUT_DIR}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = []
    patch_idx = 0
    total_scans = len(unique_uids)
    t0 = time.time()

    for scan_num, uid in enumerate(unique_uids):
        cands = groups[uid]

        # Load this scan once
        vol, origin, spacing = load_and_normalize(uid_to_path[uid])

        # Extract all candidate patches from this scan
        for c in cands:
            vz, vy, vx = world_to_voxel(
                c["coordX"], c["coordY"], c["coordZ"], origin, spacing
            )
            patch = extract_centered_patch(vol, vz, vy, vx, PATCH_SIZE)

            filename = f"patch_{patch_idx:06d}.npy"
            np.save(os.path.join(OUTPUT_DIR, filename), patch)
            manifest.append({
                "filename": filename,
                "label": c["label"],
                "seriesuid": c["seriesuid"],
            })
            patch_idx += 1

        if (scan_num + 1) % 50 == 0 or scan_num == 0 or (scan_num + 1) == total_scans:
            elapsed = time.time() - t0
            rate = (scan_num + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_scans - scan_num - 1) / rate if rate > 0 else 0
            print(f"  {scan_num+1}/{total_scans} scans | "
                  f"{patch_idx} patches | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    # Save manifest CSV (filename, label, seriesuid)
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "seriesuid"])
        writer.writeheader()
        writer.writerows(manifest)

    elapsed = time.time() - start
    n_pos_final = sum(1 for m in manifest if m["label"] == 1)
    disk_gb = (patch_idx * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * 4) / 1e9  # float32

    print(f"\n" + "=" * 70)
    print(f"  EXTRACTION COMPLETE")
    print(f"  Total patches:  {patch_idx} "
          f"({n_pos_final} nodule, {patch_idx - n_pos_final} non-nodule)")
    print(f"  Manifest:       {manifest_path}")
    print(f"  Disk usage:     ~{disk_gb:.1f} GB")
    print(f"  Total time:     {elapsed/60:.1f} minutes")
    print(f"\n  Next step:  python main.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
