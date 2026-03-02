"""
Data Splitter - Split LUNA16 candidates by patient (seriesuid)
Ensures no data leakage: all candidates from one patient stay in the same split
"""

import os
import csv
import random


class DataSplitter:
    """Split LUNA16 candidates into train/val/test by patient"""

    def __init__(self, luna_folder, train_ratio=0.7, val_ratio=0.15,
                 test_ratio=0.15, seed=42):
        self.luna_folder = luna_folder
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Load candidates.csv (real ground truth labels)
        self.candidates = []
        csv_path = os.path.join(luna_folder, "candidates.csv")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.candidates.append({
                    "seriesuid": row["seriesuid"],
                    "coordX": float(row["coordX"]),
                    "coordY": float(row["coordY"]),
                    "coordZ": float(row["coordZ"]),
                    "label": int(row["class"]),
                })

        # Get unique patient IDs
        self.series_uids = sorted(set(c["seriesuid"] for c in self.candidates))

        n_pos = sum(1 for c in self.candidates if c["label"] == 1)
        n_neg = sum(1 for c in self.candidates if c["label"] == 0)
        print(f"Loaded {len(self.candidates)} candidates "
              f"({n_pos} nodules, {n_neg} non-nodules) "
              f"from {len(self.series_uids)} patients")

        # Map seriesuid -> .mhd file path (exclude segmentation masks)
        self.uid_to_path = {}
        for root, _, files in os.walk(luna_folder):
            if "seg-lungs" in root:
                continue
            for f in files:
                if f.endswith(".mhd"):
                    uid = f.replace(".mhd", "")
                    self.uid_to_path[uid] = os.path.join(root, f)

        print(f"Mapped {len(self.uid_to_path)} CT scan files")

    def split(self):
        """
        Split by seriesuid to prevent data leakage.
        Returns (train_candidates, val_candidates, test_candidates, sizes_dict)
        """
        rng = random.Random(self.seed)

        uids = list(self.series_uids)
        rng.shuffle(uids)

        total = len(uids)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_uids = set(uids[:train_end])
        val_uids = set(uids[train_end:val_end])
        test_uids = set(uids[val_end:])

        train_cands = [c for c in self.candidates if c["seriesuid"] in train_uids]
        val_cands = [c for c in self.candidates if c["seriesuid"] in val_uids]
        test_cands = [c for c in self.candidates if c["seriesuid"] in test_uids]

        sizes = {
            "train_uids": len(train_uids),
            "val_uids": len(val_uids),
            "test_uids": len(test_uids),
            "train_candidates": len(train_cands),
            "val_candidates": len(val_cands),
            "test_candidates": len(test_cands),
            "train_pos": sum(1 for c in train_cands if c["label"] == 1),
            "val_pos": sum(1 for c in val_cands if c["label"] == 1),
            "test_pos": sum(1 for c in test_cands if c["label"] == 1),
        }

        return train_cands, val_cands, test_cands, sizes


if __name__ == "__main__":
    splitter = DataSplitter(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "LUNA16")
    )
    train_cands, val_cands, test_cands, sizes = splitter.split()

    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name:5s}: {sizes[f'{split_name}_uids']} patients, "
              f"{sizes[f'{split_name}_candidates']} candidates "
              f"({sizes[f'{split_name}_pos']} positive)")
    print("=" * 60)
