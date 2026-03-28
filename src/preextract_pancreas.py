"""
TCIA PANCREAS - PATCH PRE-EXTRACTION (run this ONCE before training)

Loads each CT volume + segmentation mask, resamples to isotropic 1 mm spacing,
extracts positive-centered and negative patches, and saves .npy files.

This script is designed for the local dataset layout:

  data/Pancreas Dataset/
    manifest-1599750808610/Pancreas-CT/PANCREAS_XXXX/.../*.dcm
    TCIA_pancreas_labels-02-05-2017/labelXXXX.nii.gz

Notes on labels:
  - If mask contains label > 1, positives are extracted from those voxels
    (tumor-first behavior).
  - Otherwise positives are extracted from label == 1 (pancreas mask).

Output:
  data/pancreas_patches/
    patch_000000.npy
    patch_000001.npy
    ...
    manifest.csv  (filename,label,seriesuid)

Usage:
  python src/preextract_pancreas.py
"""

import os
import re
import csv
import time
import random

import numpy as np
import scipy.ndimage
import SimpleITK as sitk

from utils import set_seed


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEED = 42
NEG_RATIO = 5
PATCH_SIZE = (96, 96, 96)
TARGET_SPACING = (1.0, 1.0, 1.0)  # (Z, Y, X) mm

MIN_COMPONENT_VOXELS = 20
MAX_PATCHES_PER_COMPONENT = 3
LARGE_COMPONENT_THRESHOLD = 1200

HU_MIN = -150.0
HU_MAX = 250.0
BODY_HU_THRESHOLD = -500.0

# Research target configuration
# - "tumor":      positives are tumor voxels
# - "pancreas":   positives are pancreas organ voxels
# - "auto":       infer from mask values
CLASSIFICATION_TARGET = "tumor"

# Optional hard override for positive labels.
# For this TCIA pancreas tumor label set, positives are encoded as 1.
FORCE_POSITIVE_VALUES = {1}

# Optional explicit label folder override (absolute or project-relative path).
LABEL_ROOT_OVERRIDE = None
DIAGNOSTIC_LABEL_PRINT_CASES = 3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANCREAS_ROOT = os.path.join(PROJECT_ROOT, "data", "Pancreas Dataset")
DEFAULT_CT_ROOT = os.path.join(PANCREAS_ROOT, "manifest-1599750808610", "Pancreas-CT")
DEFAULT_LABEL_ROOT = os.path.join(PANCREAS_ROOT, "TCIA_pancreas_labels-02-05-2017")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "pancreas_patches")


def resolve_path(path_value):
    """Resolve optional absolute or project-relative path"""
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value)


def parse_label_case_id(filename):
    """Extract integer case id from labelXXXX.nii.gz"""
    m = re.match(r"label(\d+)\.nii(?:\.gz)?$", filename)
    if not m:
        return None
    return int(m.group(1))


def find_dicom_series_dir(case_root):
    """Find deepest directory containing .dcm files for a case"""
    for root, _, files in os.walk(case_root):
        if any(f.lower().endswith(".dcm") for f in files):
            return root
    return None


def find_case_pairs(ct_root, label_root):
    """
    Build case mapping between CT DICOM folders and NIfTI labels.

    Returns:
        dict[case_id] = {
            "seriesuid": "pancreas_XXXX",
            "dicom_dir": <path>,
            "label_path": <path>
        }
    """
    if not os.path.isdir(ct_root):
        print(f"  ERROR: CT root not found: {ct_root}")
        return {}
    if not os.path.isdir(label_root):
        print(f"  ERROR: Label root not found: {label_root}")
        return {}

    labels = {}
    for fname in os.listdir(label_root):
        case_id = parse_label_case_id(fname)
        if case_id is not None:
            labels[case_id] = os.path.join(label_root, fname)

    pairs = {}
    missing_ct = 0
    missing_series = 0

    for case_id, label_path in sorted(labels.items()):
        case_name = f"PANCREAS_{case_id:04d}"
        case_root = os.path.join(ct_root, case_name)
        if not os.path.isdir(case_root):
            missing_ct += 1
            continue

        dicom_dir = find_dicom_series_dir(case_root)
        if dicom_dir is None:
            missing_series += 1
            continue

        pairs[case_id] = {
            "seriesuid": f"pancreas_{case_id:04d}",
            "dicom_dir": dicom_dir,
            "label_path": label_path,
        }

    print(f"  Label files found: {len(labels)}")
    print(f"  Matched CT+label cases: {len(pairs)}")
    if missing_ct > 0:
        print(f"  WARNING: {missing_ct} labels without CT folder")
    if missing_series > 0:
        print(f"  WARNING: {missing_series} CT folders without DICOM series")

    return pairs


def discover_input_roots(dataset_root):
    """Auto-discover CT and label roots to avoid hardcoded timestamp folders"""
    ct_root = DEFAULT_CT_ROOT if os.path.isdir(DEFAULT_CT_ROOT) else None
    label_root = DEFAULT_LABEL_ROOT if os.path.isdir(DEFAULT_LABEL_ROOT) else None

    label_override = resolve_path(LABEL_ROOT_OVERRIDE)
    if label_override is not None:
        if os.path.isdir(label_override):
            label_root = label_override
        else:
            print(f"  WARNING: LABEL_ROOT_OVERRIDE does not exist: {label_override}")

    if ct_root is None and os.path.isdir(dataset_root):
        for name in sorted(os.listdir(dataset_root)):
            candidate = os.path.join(dataset_root, name, "Pancreas-CT")
            if name.startswith("manifest-") and os.path.isdir(candidate):
                ct_root = candidate
                break

    if label_root is None and os.path.isdir(dataset_root):
        for name in sorted(os.listdir(dataset_root)):
            candidate = os.path.join(dataset_root, name)
            if os.path.isdir(candidate) and "pancreas_labels" in name.lower():
                label_root = candidate
                break

    return ct_root, label_root


def load_dicom_volume(dicom_dir):
    """Load DICOM series with SimpleITK"""
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if len(file_names) == 0:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")
    reader.SetFileNames(file_names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)
    return image, volume


def align_label_to_ct(label_img, ct_img):
    """Resample label to CT geometry when metadata does not match"""
    same_size = label_img.GetSize() == ct_img.GetSize()
    same_spacing = all(
        abs(a - b) < 1e-6 for a, b in zip(label_img.GetSpacing(), ct_img.GetSpacing())
    )
    same_origin = all(
        abs(a - b) < 1e-3 for a, b in zip(label_img.GetOrigin(), ct_img.GetOrigin())
    )
    same_direction = all(
        abs(a - b) < 1e-6 for a, b in zip(label_img.GetDirection(), ct_img.GetDirection())
    )

    if same_size and same_spacing and same_origin and same_direction:
        return label_img

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(label_img)


def resample_to_isotropic(vol_hu, mask, spacing_xyz):
    """Resample volume and mask from native spacing to TARGET_SPACING"""
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    zoom_factors = tuple(s / t for s, t in zip(spacing_zyx, TARGET_SPACING))

    if not all(abs(f - 1.0) < 0.01 for f in zoom_factors):
        vol_hu = scipy.ndimage.zoom(vol_hu, zoom_factors, order=1).astype(np.float32)
        mask = scipy.ndimage.zoom(mask, zoom_factors, order=0).astype(np.int16)

    return vol_hu, mask


def select_positive_mask(mask, target="auto", forced_values=None):
    """
    Select positive voxels from segmentation.

    Priority:
      1) label > 1   (tumor-first behavior when available)
      2) label == 1  (pancreas mask)
      3) label > 0   (fallback)
    """
    unique_vals = set(np.unique(mask).tolist())

    if forced_values is not None:
        values = sorted({int(v) for v in forced_values})
        return np.isin(mask, values), f"custom_{'_'.join(str(v) for v in values)}"

    target = target.lower().strip()
    if target == "tumor":
        return (mask > 1), "tumor"
    if target == "pancreas":
        return (mask == 1), "pancreas"
    if target == "nonzero":
        return (mask > 0), "nonzero"

    if any(v > 1 for v in unique_vals):
        return (mask > 1), "tumor"
    if 1 in unique_vals:
        return (mask == 1), "pancreas"
    return (mask > 0), "nonzero"


def build_negative_candidate_mask(vol_hu, mask, mode):
    """Create candidate region for negative patch centers"""
    body_mask = vol_hu > BODY_HU_THRESHOLD

    if mode == "tumor":
        # Prefer pancreas tissue without tumor; fallback to body without labels
        neg_mask = (mask == 1)
        if int(neg_mask.sum()) < 200:
            neg_mask = np.logical_and(body_mask, mask == 0)
    else:
        # If positives are pancreas, negatives are body tissue outside pancreas
        neg_mask = np.logical_and(body_mask, mask == 0)

    return neg_mask


def find_components(pos_mask):
    """Connected components + centroids for positive regions"""
    labeled, n_components = scipy.ndimage.label(pos_mask.astype(np.int8))

    components = []
    for comp_id in range(1, n_components + 1):
        comp = labeled == comp_id
        voxels = int(comp.sum())
        if voxels < MIN_COMPONENT_VOXELS:
            continue

        cz, cy, cx = scipy.ndimage.center_of_mass(comp)
        components.append({
            "id": comp_id,
            "n_voxels": voxels,
            "centroid": (int(round(cz)), int(round(cy)), int(round(cx))),
        })

    return components, labeled


def get_positive_positions(components, labeled, rng):
    """Generate positive patch centers from components"""
    positions = []

    for comp in components:
        positions.append(comp["centroid"])

        if comp["n_voxels"] > LARGE_COMPONENT_THRESHOLD and MAX_PATCHES_PER_COMPONENT > 1:
            coords = np.argwhere(labeled == comp["id"])
            n_extra = min(MAX_PATCHES_PER_COMPONENT - 1, 2)
            if len(coords) > n_extra:
                pick = rng.sample(range(len(coords)), n_extra)
                for idx in pick:
                    z, y, x = coords[idx]
                    positions.append((int(z), int(y), int(x)))

    return positions


def sample_negative_positions(neg_mask, pos_mask, n_neg, patch_size, rng):
    """Sample negative centers and reject patches overlapping positive voxels"""
    pd, ph, pw = patch_size
    d, h, w = neg_mask.shape

    coords = np.argwhere(neg_mask)
    if len(coords) == 0:
        return []

    mz, my, mx = pd // 2, ph // 2, pw // 2
    valid = (
        (coords[:, 0] >= mz) & (coords[:, 0] < d - mz) &
        (coords[:, 1] >= my) & (coords[:, 1] < h - my) &
        (coords[:, 2] >= mx) & (coords[:, 2] < w - mx)
    )
    coords = coords[valid]
    if len(coords) == 0:
        return []

    n_candidates = min(len(coords), max(n_neg * 25, n_neg))
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    picked = np_rng.choice(len(coords), size=n_candidates, replace=False)

    positions = []
    for i in picked:
        if len(positions) >= n_neg:
            break

        cz, cy, cx = [int(v) for v in coords[i]]
        z0 = max(0, cz - pd // 2)
        y0 = max(0, cy - ph // 2)
        x0 = max(0, cx - pw // 2)

        sub_pos = pos_mask[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
        if np.any(sub_pos):
            continue

        positions.append((cz, cy, cx))

    return positions


def extract_centered_patch(vol, cz, cy, cx, patch_size):
    """Extract patch centered at (cz, cy, cx) with boundary handling"""
    pd, ph, pw = patch_size
    d, h, w = vol.shape

    z0 = max(0, min(cz - pd // 2, d - pd))
    y0 = max(0, min(cy - ph // 2, h - ph))
    x0 = max(0, min(cx - pw // 2, w - pw))

    patch = vol[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    if patch.shape != (pd, ph, pw):
        out = np.zeros((pd, ph, pw), dtype=np.float32)
        out[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
        patch = out
    return patch


def main():
    set_seed(SEED)
    rng = random.Random(SEED)
    start = time.time()

    print("\n" + "=" * 72)
    print("  TCIA PANCREAS - PATCH PRE-EXTRACTION")
    print("  Run this ONCE. Then train with a pancreas training script.")
    print("=" * 72 + "\n")

    ct_root, label_root = discover_input_roots(PANCREAS_ROOT)
    if ct_root is None or label_root is None:
        print(f"  ERROR: Could not auto-discover dataset folders under {PANCREAS_ROOT}")
        print("  Expected structure:")
        print("    Pancreas Dataset/manifest-*/Pancreas-CT/PANCREAS_XXXX/.../*.dcm")
        print("    Pancreas Dataset/TCIA_pancreas_labels-*/labelXXXX.nii.gz")
        return

    print(f"  CT root:          {ct_root}")
    print(f"  Label root:       {label_root}")
    print(f"  Output:           {OUTPUT_DIR}")
    print(f"  Target spacing:   {TARGET_SPACING[0]:.0f}x{TARGET_SPACING[1]:.0f}x{TARGET_SPACING[2]:.0f} mm")
    print(f"  Patch size:       {PATCH_SIZE[0]}x{PATCH_SIZE[1]}x{PATCH_SIZE[2]}")
    print(f"  HU window:        [{HU_MIN:.0f}, {HU_MAX:.0f}]")
    print(f"  Target mode:      {CLASSIFICATION_TARGET}")
    if FORCE_POSITIVE_VALUES is not None:
        print(f"  Forced labels:    {sorted(int(v) for v in FORCE_POSITIVE_VALUES)}")

    pairs = find_case_pairs(ct_root, label_root)
    if len(pairs) == 0:
        print("\n  ERROR: No valid pancreas CT+label pairs found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = []
    patch_idx = 0
    total_pos = 0
    total_neg = 0
    total_components = 0
    mode_counter = {"tumor": 0, "pancreas": 0, "nonzero": 0}
    mode_counter_extra = {}
    cases_with_components = 0
    skipped_failed = 0
    skipped_no_components = 0
    skipped_no_positive_voxels = 0

    total_cases = len(pairs)
    t0 = time.time()

    for case_num, (case_id, info) in enumerate(sorted(pairs.items()), start=1):
        try:
            ct_img, vol_hu = load_dicom_volume(info["dicom_dir"])

            label_img = sitk.ReadImage(info["label_path"])
            label_img = align_label_to_ct(label_img, ct_img)
            mask = sitk.GetArrayFromImage(label_img).astype(np.int16)
        except Exception as e:
            skipped_failed += 1
            print(f"  WARNING: case {case_id:04d} failed to load ({e}), skipping")
            continue

        vol_hu, mask = resample_to_isotropic(vol_hu, mask, ct_img.GetSpacing())

        if case_num <= DIAGNOSTIC_LABEL_PRINT_CASES:
            unique_vals = np.unique(mask)
            vals_str = ", ".join(str(int(v)) for v in unique_vals[:10])
            more = " ..." if len(unique_vals) > 10 else ""
            print(f"  label stats case {case_id:04d}: unique labels = [{vals_str}]{more}")

        # Use raw HU for body masking in negative sampling.
        vol_hu_raw = vol_hu.copy()

        # Window + normalize for model input patches.
        vol_hu_windowed = np.clip(vol_hu, HU_MIN, HU_MAX)
        vol = ((vol_hu_windowed - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)

        pos_mask, mode = select_positive_mask(
            mask,
            target=CLASSIFICATION_TARGET,
            forced_values=FORCE_POSITIVE_VALUES,
        )
        if mode in mode_counter:
            mode_counter[mode] += 1
        else:
            mode_counter_extra[mode] = mode_counter_extra.get(mode, 0) + 1

        if int(pos_mask.sum()) == 0:
            skipped_no_positive_voxels += 1
            continue

        components, labeled = find_components(pos_mask)
        total_components += len(components)
        if len(components) == 0:
            skipped_no_components += 1
            continue

        cases_with_components += 1

        pos_positions = get_positive_positions(components, labeled, rng)
        for cz, cy, cx in pos_positions:
            patch = extract_centered_patch(vol, cz, cy, cx, PATCH_SIZE)
            filename = f"patch_{patch_idx:06d}.npy"
            np.save(os.path.join(OUTPUT_DIR, filename), patch)
            manifest.append({
                "filename": filename,
                "label": 1,
                "seriesuid": info["seriesuid"],
            })
            patch_idx += 1
            total_pos += 1

        n_neg = len(pos_positions) * NEG_RATIO
        if n_neg > 0:
            neg_mask = build_negative_candidate_mask(vol_hu_raw, mask, mode)
            neg_positions = sample_negative_positions(neg_mask, pos_mask, n_neg, PATCH_SIZE, rng)

            for cz, cy, cx in neg_positions:
                patch = extract_centered_patch(vol, cz, cy, cx, PATCH_SIZE)
                filename = f"patch_{patch_idx:06d}.npy"
                np.save(os.path.join(OUTPUT_DIR, filename), patch)
                manifest.append({
                    "filename": filename,
                    "label": 0,
                    "seriesuid": info["seriesuid"],
                })
                patch_idx += 1
                total_neg += 1

        if case_num % 10 == 0 or case_num == 1 or case_num == total_cases:
            elapsed = time.time() - t0
            rate = case_num / elapsed if elapsed > 0 else 0.0
            remain = (total_cases - case_num) / rate if rate > 0 else 0.0
            print(
                f"  {case_num}/{total_cases} cases | "
                f"components: {len(components)} | "
                f"+{total_pos} -{total_neg} patches | "
                f"{elapsed:.0f}s elapsed | ~{remain:.0f}s remaining"
            )

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "seriesuid"])
        writer.writeheader()
        writer.writerows(manifest)

    elapsed = time.time() - start
    unique_patients = len(set(m["seriesuid"] for m in manifest))
    disk_gb = (patch_idx * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * 4) / 1e9

    print("\n" + "=" * 72)
    print("  EXTRACTION COMPLETE")
    print(f"  Total cases:       {total_cases}")
    print(f"  Cases with comps:  {cases_with_components}")
    print(f"  Skipped cases:     {skipped_failed} load failures, {skipped_no_positive_voxels} with no positive voxels, {skipped_no_components} with no components")
    print(f"  Positive mode use: tumor={mode_counter.get('tumor', 0)}, "
          f"pancreas={mode_counter.get('pancreas', 0)}, "
          f"nonzero={mode_counter.get('nonzero', 0)}")
    if mode_counter_extra:
        extras = ", ".join(f"{k}={v}" for k, v in sorted(mode_counter_extra.items()))
        print(f"  Positive mode use (extra): {extras}")
    print(f"  Total components:  {total_components}")
    print(f"  Unique patients:   {unique_patients}")
    print(f"  Total patches:     {patch_idx} ({total_pos} positive, {total_neg} negative)")
    print(f"  Ratio:             1:{total_neg // max(total_pos, 1)}")
    print(f"  Manifest:          {manifest_path}")
    print(f"  Disk usage:        ~{disk_gb:.1f} GB")
    print(f"  Total time:        {elapsed/60:.1f} minutes")
    if total_pos == 0:
        print("  ERROR: No positive patches generated. Check label encoding and CLASSIFICATION_TARGET.")
        print("         For binary tumor masks, set FORCE_POSITIVE_VALUES = {1}.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
