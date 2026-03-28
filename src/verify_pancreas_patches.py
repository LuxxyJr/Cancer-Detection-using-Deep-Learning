"""
Quick verification for pancreas pre-extracted patches.

Checks:
  - manifest exists and is readable
  - patch files listed in manifest exist
  - label distribution and patient count
  - patch file count matches manifest
  - sample patch shape/dtype/value range

Usage:
  python src/verify_pancreas_patches.py
"""

import csv
import glob
import os

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR = os.path.join(PROJECT_ROOT, "data", "pancreas_patches")
MANIFEST_PATH = os.path.join(PATCH_DIR, "manifest.csv")
LOG_PATH = os.path.join(PROJECT_ROOT, "pancreas_preextract.log")


def main():
    print("=" * 70)
    print("PANCREAS PATCH VERIFICATION")
    print("=" * 70)
    print(f"Patch dir:   {PATCH_DIR}")
    print(f"Manifest:    {MANIFEST_PATH}")

    if not os.path.isfile(MANIFEST_PATH):
        print("\nERROR: manifest.csv not found.")
        print("Run: python src/preextract_pancreas.py")
        return

    rows = []
    with open(MANIFEST_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        print("\nERROR: manifest.csv is empty.")
        return

    labels = []
    patients = set()
    missing = []

    for r in rows:
        try:
            label = int(r["label"])
        except Exception:
            label = None
        labels.append(label)

        patients.add(r.get("seriesuid", ""))

        fpath = os.path.join(PATCH_DIR, r["filename"])
        if not os.path.isfile(fpath):
            missing.append(r["filename"])

    n_total = len(rows)
    n_pos = sum(1 for x in labels if x == 1)
    n_neg = sum(1 for x in labels if x == 0)
    invalid_labels = sum(1 for x in labels if x not in (0, 1))

    disk_files = glob.glob(os.path.join(PATCH_DIR, "patch_*.npy"))

    print("\nSUMMARY")
    print("-" * 70)
    print(f"Total manifest rows:     {n_total}")
    print(f"Positive patches (1):    {n_pos}")
    print(f"Negative patches (0):    {n_neg}")
    print(f"Unique patients:         {len(patients)}")
    print(f"Patch files on disk:     {len(disk_files)}")
    print(f"Missing files in disk:   {len(missing)}")
    print(f"Invalid labels:          {invalid_labels}")

    if n_pos > 0:
        print(f"Class ratio (neg:pos):   1:{n_neg // max(n_pos, 1)}")

    sample_n = min(5, n_total)
    print(f"\nSAMPLE PATCH CHECK ({sample_n} files)")
    print("-" * 70)
    for i in range(sample_n):
        fname = rows[i]["filename"]
        fpath = os.path.join(PATCH_DIR, fname)
        if not os.path.isfile(fpath):
            print(f"{fname}: MISSING")
            continue

        arr = np.load(fpath)
        print(
            f"{fname}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={arr.min():.4f}, max={arr.max():.4f}"
        )

    if os.path.isfile(LOG_PATH):
        print("\nPREEXTRACT LOG NOTE")
        print("-" * 70)
        with open(LOG_PATH, "r", errors="ignore") as f:
            lines = f.readlines()
        mode_lines = [ln.strip() for ln in lines if "Positive mode use:" in ln]
        if mode_lines:
            print(mode_lines[-1])
        else:
            print("No 'Positive mode use' line found in pancreas_preextract.log")
    else:
        print("\nTip: save extraction output to pancreas_preextract.log to record mode usage.")

    print("\nSTATUS")
    print("-" * 70)
    if missing:
        print("FAIL: Some manifest files are missing on disk.")
    elif invalid_labels > 0:
        print("FAIL: Found labels outside {0,1}.")
    elif len(disk_files) != n_total:
        print("WARN: Disk file count differs from manifest row count.")
        print("      This can happen if old patches remain in folder.")
    else:
        print("PASS: Manifest and patches look consistent.")

    print("=" * 70)


if __name__ == "__main__":
    main()
