"""
TCIA Pancreas Classification - Multi-Model 3-Fold Cross-Validation Pipeline.

This reuses the proven liver training pipeline and swaps only organ-specific
paths/labels so training behavior remains consistent across organs.

Run preextract_pancreas.py first to create:
  data/pancreas_patches/manifest.csv + patch_*.npy

Usage:
  python src/main_pancreas.py
"""

import os

import main_liver as base


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Organ-specific paths
base.PATCHES_DIR = os.path.join(PROJECT_ROOT, "data", "pancreas_patches")
base.RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "pancreas")
base.CHECKPOINT_DIR = os.path.join(base.RESULTS_DIR, "checkpoints")
base.PLOT_DIR = os.path.join(base.RESULTS_DIR, "plots")

# Organ-specific labels/metadata
base.ORGAN_NAME = "TCIA Pancreas"
base.DATASET_NAME = "TCIA Pancreatic Cancer"
base.TASK_DESCRIPTION = "Binary classification"
base.PREEXTRACT_HINT = "python src/preextract_pancreas.py"
base.HU_WINDOW_DESC = "[-150, 250]"
base.POS_LABEL = "Tumor"
base.NEG_LABEL = "Non-tumor"
base.INPUT_PATCH_SIZE = (96, 96, 96)

# Keep methodology consistent with previous organs:
# - train BOTH models
# - use the same core training setup defined in main_liver.py
base.MODELS = [
    {"name": "ResNet3D", "arch": "resnet3d", "gradcam_layer": "encoder.b4"},
    {"name": "VGG3D", "arch": "vgg3d", "gradcam_layer": "block3"},
]


if __name__ == "__main__":
    base.main()
