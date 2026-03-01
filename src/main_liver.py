"""
LiTS Liver Tumor Classification - Multi-Model 3-Fold Cross-Validation Pipeline

Uses pre-extracted .npy patches for fast training.
Run preextract_liver.py FIRST to create the patch files.

Pipeline:
  1. Read manifest (all patches + patient IDs)
  2. Split patients into 3 folds (deterministic, no leakage)
  3. For each model (ResNet3D, VGG3D):
     a. For each fold: train on 2 folds, test on 1 (with val carved from train)
     b. Generate Grad-CAM figures after fold 0
     c. Aggregate results: mean +/- std across folds
     d. Generate per-model plots (ROC, training curves)
  4. Cross-model comparison (ROC overlay + results table)

Estimated time: ~6-12 hours on RTX 4050 (6GB VRAM)
"""

import os
import csv
import time
import random

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for overnight runs)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import set_seed, save_checkpoint, load_checkpoint, print_model_info
from fast_dataset import FastPatchDataset
from architecture import get_model
from training import Trainer
from evaluator import MetricsCalculator
from gradcam import generate_gradcam_figures


# ======================================================================
# CONFIGURATION
# ======================================================================
SEED = 42
BATCH_SIZE = 4
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
MIN_EPOCHS = 25         # Don't early-stop before this epoch
NUM_WORKERS = 2
K_FOLDS = 3
VAL_FRACTION = 0.15
ACCUM_STEPS = 4         # Gradient accumulation (effective batch = BATCH_SIZE * 4 = 16)
WARMUP_EPOCHS = 5       # Linear LR warmup epochs
USE_FOCAL_LOSS = True   # Focal Loss instead of CrossEntropyLoss
FOCAL_GAMMA = 2.0       # Focal Loss focusing parameter

PATCHES_DIR = r"D:\Research Paper Work\Multi Organ Cancer Detector\data\liver_patches"
RESULTS_DIR = r"D:\Research Paper Work\Multi Organ Cancer Detector\results\liver"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# Label names for display and Grad-CAM
POS_LABEL = "Tumor"
NEG_LABEL = "Non-tumor"
ORGAN_NAME = "LiTS Liver Tumor"

# Models to train and compare
# Each entry: (display_name, architecture_key, gradcam_target_layer_path)
MODELS = [
    {"name": "ResNet3D", "arch": "resnet3d", "gradcam_layer": "encoder.b4"},
    {"name": "VGG3D",    "arch": "vgg3d",    "gradcam_layer": "block3"},
]


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def read_manifest(patches_dir):
    """Read manifest.csv from patches directory"""
    manifest_path = os.path.join(patches_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"\n  ERROR: {manifest_path} not found!")
        print(f"  Run preextract_liver.py first to create patch files.")
        print(f"  Usage:  python preextract_liver.py\n")
        return None

    samples = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "filename": row["filename"],
                "label": int(row["label"]),
                "seriesuid": row["seriesuid"],
            })
    return samples


def make_loader(dataset, batch_size, shuffle, num_workers, sampler=None):
    """Create a DataLoader with proper settings"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def tta_predict(model, dataloader, device, use_amp):
    """
    Test-Time Augmentation: average softmax probabilities over geometric
    transforms to get more robust predictions.

    Uses the same transforms available during training (flips + 90-deg rotations).
    For cubic patches, rotations work in all 3 planes.

    Returns:
        labels:     np.array of true labels
        avg_proba:  np.array of averaged positive-class probabilities
        avg_preds:  np.array of predicted classes (threshold=0.5)
    """
    model.eval()

    # Geometric augmentation functions on (B, C, D, H, W) tensors
    augmentations = [
        lambda x: x,                          # Original
        lambda x: torch.flip(x, [2]),          # Flip Z
        lambda x: torch.flip(x, [3]),          # Flip Y
        lambda x: torch.flip(x, [4]),          # Flip X
        lambda x: torch.rot90(x, 1, [3, 4]),   # Rotate 90 deg axial
        lambda x: torch.rot90(x, 1, [2, 4]),   # Rotate 90 deg coronal
        lambda x: torch.rot90(x, 1, [2, 3]),   # Rotate 90 deg sagittal
    ]

    all_labels = None
    proba_sum = None

    for aug_fn in augmentations:
        batch_proba = []
        batch_labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x_aug = aug_fn(x).to(device)

                if use_amp and device == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(x_aug)
                else:
                    logits = model(x_aug)

                proba = F.softmax(logits, dim=1)[:, 1]
                batch_proba.extend(proba.cpu().numpy())
                batch_labels.extend(y.numpy())

        proba_arr = np.array(batch_proba)

        if all_labels is None:
            all_labels = np.array(batch_labels)
            proba_sum = proba_arr
        else:
            proba_sum += proba_arr

    avg_proba = proba_sum / len(augmentations)
    avg_preds = (avg_proba > 0.5).astype(int)

    return all_labels, avg_proba, avg_preds


# ======================================================================
# SINGLE-MODEL K-FOLD CV
# ======================================================================

def train_model_cv(model_config, manifest, folds, device, pipeline_start,
                   all_model_results):
    """
    Run full K-fold cross-validation for a single model architecture.
    """
    model_name = model_config["name"]
    arch_key = model_config["arch"]

    model_ckpt_dir = os.path.join(CHECKPOINT_DIR, arch_key)
    model_plot_dir = os.path.join(PLOT_DIR, arch_key)
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(model_plot_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"#  MODEL: {model_name} ({arch_key})")
    print("#" * 80)

    # Storage for aggregate results
    all_fold_metrics = []
    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_train_accs = []
    all_fold_val_accs = []
    all_fold_epochs = []
    fold_times = []

    # For aggregate ROC (each sample tested exactly once across all folds)
    agg_test_labels = []
    agg_test_proba = []
    agg_test_preds = []

    # Per-fold ROC data for plotting
    fold_roc_data = []

    model_start = time.time()

    for fold_idx in range(K_FOLDS):
        fold_start = time.time()

        print("\n" + "=" * 80)
        print(f"  {model_name} - FOLD {fold_idx + 1} / {K_FOLDS}")
        print("=" * 80)

        # ── Split patients for this fold ─────────────────────────────
        test_uids = folds[fold_idx]

        # Remaining patients (for train + val)
        remaining_uids = []
        for j in range(K_FOLDS):
            if j != fold_idx:
                remaining_uids.extend(list(folds[j]))

        # Split remaining into train and val (by patient)
        rng_fold = random.Random(SEED + fold_idx)
        rng_fold.shuffle(remaining_uids)
        val_count = max(1, int(len(remaining_uids) * VAL_FRACTION))
        val_uids = set(remaining_uids[:val_count])
        train_uids = set(remaining_uids[val_count:])

        print(f"\n  Split: {len(train_uids)} train patients, "
              f"{len(val_uids)} val patients, "
              f"{len(test_uids)} test patients")

        # ── Filter manifest by patient sets ──────────────────────────
        train_all = [s for s in manifest if s["seriesuid"] in train_uids]
        val_all = [s for s in manifest if s["seriesuid"] in val_uids]
        test_all = [s for s in manifest if s["seriesuid"] in test_uids]

        n_pos_train = sum(1 for s in train_all if s["label"] == 1)
        n_neg_train = len(train_all) - n_pos_train

        # ── Create datasets ──────────────────────────────────────────
        print(f"  Datasets:")
        train_dataset = FastPatchDataset(
            PATCHES_DIR, augment=True, samples=train_all, name="train"
        )
        val_dataset = FastPatchDataset(
            PATCHES_DIR, augment=False, samples=val_all, name="val"
        )
        test_dataset = FastPatchDataset(
            PATCHES_DIR, augment=False, samples=test_all, name="test"
        )

        # Weighted sampler: oversample positives so each batch is ~50/50
        sample_weights = [
            n_neg_train / max(n_pos_train, 1) if s["label"] == 1 else 1.0
            for s in train_all
        ]
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_all), replacement=True
        )

        train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, sampler=train_sampler)
        val_loader = make_loader(val_dataset, BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS)
        test_loader = make_loader(test_dataset, BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS)

        # ── Fresh model and trainer ──────────────────────────────────
        model = get_model(arch_key, num_classes=2)
        if fold_idx == 0:
            print_model_info(model)

        trainer = Trainer(
            model, device=device, epochs=EPOCHS,
            lr=LR, weight_decay=WEIGHT_DECAY,
            class_weights=None,  # Sampler handles class balance
            loss_type="focal" if USE_FOCAL_LOSS else "cross_entropy",
            focal_gamma=FOCAL_GAMMA,
            accum_steps=ACCUM_STEPS,
            warmup_epochs=WARMUP_EPOCHS,
        )

        # ── Training loop ────────────────────────────────────────────
        print(f"\n  Training (max {EPOCHS} epochs, patience={PATIENCE})...\n")

        fold_ckpt_dir = os.path.join(model_ckpt_dir, f"fold_{fold_idx}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(EPOCHS):
            epoch_start = time.time()

            train_loss, train_acc = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.validate(val_loader)

            epoch_time = time.time() - epoch_start
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            lr_now = trainer.optimizer.param_groups[0]["lr"]

            # Print every 5 epochs, first epoch, and last epoch
            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed_fold = time.time() - fold_start
                elapsed_total = time.time() - pipeline_start
                print(f"    Ep {epoch+1:3d}/{EPOCHS} | "
                      f"TrL:{train_loss:.4f} TrA:{train_acc:.3f} | "
                      f"VaL:{val_loss:.4f} VaA:{val_acc:.3f} | "
                      f"LR:{lr_now:.1e} | {epoch_time:.0f}s/ep | "
                      f"fold:{elapsed_fold/60:.0f}m total:{elapsed_total/60:.0f}m")

            # Best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    model, trainer.optimizer, epoch, val_loss,
                    os.path.join(fold_ckpt_dir, "best_model.pth")
                )
            else:
                patience_counter += 1

            # Periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model, trainer.optimizer, epoch, val_loss,
                    os.path.join(fold_ckpt_dir, f"epoch_{epoch+1}.pth")
                )

            # Early stopping (only after MIN_EPOCHS)
            if patience_counter >= PATIENCE and (epoch + 1) >= MIN_EPOCHS:
                print(f"\n    Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

        actual_epochs = len(train_losses)
        fold_train_time = time.time() - fold_start

        print(f"\n  {model_name} Fold {fold_idx+1} training complete: "
              f"{actual_epochs} epochs in {fold_train_time/60:.1f} min "
              f"(best val loss: {best_val_loss:.4f})")

        # Time estimate for remaining folds
        fold_times.append(fold_train_time)
        remaining_folds = K_FOLDS - fold_idx - 1
        if remaining_folds > 0:
            avg_fold_time = np.mean(fold_times)
            est_remaining = avg_fold_time * remaining_folds
            print(f"  Estimated time remaining for {model_name}: "
                  f"~{est_remaining/60:.0f} min")

        # Store training history
        all_fold_train_losses.append(train_losses)
        all_fold_val_losses.append(val_losses)
        all_fold_train_accs.append(train_accs)
        all_fold_val_accs.append(val_accs)
        all_fold_epochs.append(actual_epochs)

        # ── Evaluate on test fold (with TTA) ─────────────────────────
        print(f"\n  Evaluating {model_name} fold {fold_idx+1} on test set "
              f"({len(test_dataset)} samples) with TTA...")

        best_path = os.path.join(fold_ckpt_dir, "best_model.pth")
        load_checkpoint(model, path=best_path)
        model.to(device)

        fold_labels, fold_proba, fold_preds = tta_predict(
            model, test_loader, device, use_amp=trainer.use_amp
        )

        # Per-fold metrics
        metrics = MetricsCalculator.calculate_metrics(
            fold_labels, fold_proba, fold_preds
        )
        MetricsCalculator.print_metrics(metrics,
                                        f"{model_name} FOLD {fold_idx+1} TEST")
        all_fold_metrics.append(metrics)

        # Store for aggregate ROC
        agg_test_labels.extend(fold_labels)
        agg_test_proba.extend(fold_proba)
        agg_test_preds.extend(fold_preds)

        # Per-fold ROC curve data
        fpr, tpr, _ = roc_curve(fold_labels, fold_proba)
        fold_roc_data.append((fpr, tpr, metrics["auc_roc"]))

        # ── Grad-CAM after fold 0 ───────────────────────────────────
        if fold_idx == 0:
            print(f"\n  Generating Grad-CAM visualizations for {model_name}...")
            gradcam_dir = os.path.join(PLOT_DIR, "gradcam", arch_key)
            try:
                generate_gradcam_figures(
                    model, arch_key, test_loader, device, gradcam_dir,
                    num_examples=3, use_amp=trainer.use_amp,
                    pos_label=POS_LABEL, neg_label=NEG_LABEL,
                )
            except Exception as e:
                print(f"  WARNING: Grad-CAM failed: {e}")
                print(f"  (Continuing without Grad-CAM figures)")

        # Clean up GPU memory for next fold
        del model, trainer
        torch.cuda.empty_cache()

    model_time = time.time() - model_start

    # ── Per-model aggregate results ──────────────────────────────────
    metric_keys = ["sensitivity", "specificity", "precision",
                   "accuracy", "auc_roc", "f1"]

    print(f"\n{'=' * 80}")
    print(f"  {model_name} AGGREGATE RESULTS ({K_FOLDS}-FOLD CV)")
    print(f"{'=' * 80}")

    # Per-fold table
    header = f"  {'Metric':<15}"
    for i in range(K_FOLDS):
        header += f" {'Fold '+str(i+1):>8}"
    header += f" {'Mean+/-Std':>14}"
    print(f"\n{header}")
    print(f"  {'-' * (15 + 9 * K_FOLDS + 15)}")

    summary = {}
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[key] = {"mean": mean_val, "std": std_val, "values": values}

        row = f"  {key:<15}"
        for v in values:
            row += f" {v:>8.4f}"
        row += f" {mean_val:.4f}+/-{std_val:.4f}"
        print(row)

    # Aggregate metrics (all test predictions combined)
    agg_labels = np.array(agg_test_labels)
    agg_proba = np.array(agg_test_proba)
    agg_preds = np.array(agg_test_preds)

    agg_metrics = MetricsCalculator.calculate_metrics(
        agg_labels, agg_proba, agg_preds
    )

    print(f"\n  Aggregate (all folds combined, {len(agg_labels)} samples):")
    for key in metric_keys:
        print(f"    {key:<15} {agg_metrics[key]:.4f}")

    # ── Per-model ROC curve ──────────────────────────────────────────
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for i, (fpr, tpr, auc_val) in enumerate(fold_roc_data):
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"Fold {i+1} (AUC = {auc_val:.4f})")

    agg_fpr, agg_tpr, _ = roc_curve(agg_labels, agg_proba)
    agg_auc = auc(agg_fpr, agg_tpr)
    ax.plot(agg_fpr, agg_tpr, color="black", linewidth=2.5, linestyle="--",
            label=f"Aggregate (AUC = {agg_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"{model_name} ROC - {ORGAN_NAME} ({K_FOLDS}-Fold CV)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    roc_path = os.path.join(model_plot_dir, "roc_curve_kfold.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ROC curve saved: {roc_path}")

    # ── Per-model training curves ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(K_FOLDS):
        axes[0].plot(all_fold_train_losses[i], color=colors[i], alpha=0.4,
                     linewidth=1, label=f"Fold {i+1} Train")
        axes[0].plot(all_fold_val_losses[i], color=colors[i], linewidth=2,
                     linestyle="--", label=f"Fold {i+1} Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Training & Validation Loss")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    for i in range(K_FOLDS):
        axes[1].plot(all_fold_train_accs[i], color=colors[i], alpha=0.4,
                     linewidth=1, label=f"Fold {i+1} Train")
        axes[1].plot(all_fold_val_accs[i], color=colors[i], linewidth=2,
                     linestyle="--", label=f"Fold {i+1} Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Training & Validation Accuracy")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    curves_path = os.path.join(model_plot_dir, "training_curves_kfold.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {curves_path}")

    # Return results for cross-model comparison
    total_params = sum(
        p.numel() for p in get_model(arch_key, num_classes=2).parameters()
    )

    return {
        "name": model_name,
        "arch": arch_key,
        "summary": summary,
        "agg_metrics": agg_metrics,
        "agg_labels": agg_labels,
        "agg_proba": agg_proba,
        "fold_roc_data": fold_roc_data,
        "fold_epochs": all_fold_epochs,
        "fold_times": fold_times,
        "total_time": model_time,
        "total_params": total_params,
        "roc_path": roc_path,
        "curves_path": curves_path,
    }


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    set_seed(SEED)
    pipeline_start = time.time()

    print("\n" + "=" * 80)
    print(f"  {ORGAN_NAME.upper()} CLASSIFICATION")
    print("  MULTI-MODEL {}-FOLD CROSS-VALIDATION PIPELINE".format(K_FOLDS))
    print("=" * 80 + "\n")

    # ==================================================================
    # STEP 1: READ MANIFEST
    # ==================================================================
    print("STEP 1: READING MANIFEST")
    print("-" * 80)

    manifest = read_manifest(PATCHES_DIR)
    if manifest is None:
        return

    n_pos = sum(1 for s in manifest if s["label"] == 1)
    n_neg = len(manifest) - n_pos
    unique_uids = sorted(set(s["seriesuid"] for s in manifest))

    print(f"  Total patches:   {len(manifest)} "
          f"({n_pos} tumor, {n_neg} non-tumor)")
    print(f"  Unique patients: {len(unique_uids)}")
    print()

    # ==================================================================
    # STEP 2: SET UP FOLDS (shared across all models)
    # ==================================================================
    print("STEP 2: SETTING UP {}-FOLD CROSS-VALIDATION".format(K_FOLDS))
    print("-" * 80)

    # Shuffle patients deterministically
    rng = random.Random(SEED)
    shuffled_uids = unique_uids.copy()
    rng.shuffle(shuffled_uids)

    # Split into K folds (by patient -- no data leakage)
    fold_size = len(shuffled_uids) // K_FOLDS
    folds = []
    for i in range(K_FOLDS):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < K_FOLDS - 1 else len(shuffled_uids)
        folds.append(set(shuffled_uids[start_idx:end_idx]))

    print(f"  Patient-level splitting (prevents data leakage):\n")
    for i, fold_uids in enumerate(folds):
        fold_patches = [s for s in manifest if s["seriesuid"] in fold_uids]
        fold_pos = sum(1 for s in fold_patches if s["label"] == 1)
        fold_neg = len(fold_patches) - fold_pos
        print(f"    Fold {i+1}: {len(fold_uids):3d} patients, "
              f"{len(fold_patches):4d} patches "
              f"({fold_pos} tumor, {fold_neg} non-tumor)")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:    {gpu_name}")
        print(f"  VRAM:   {vram:.1f} GB")

    print(f"\n  Models to train: {', '.join(m['name'] for m in MODELS)}")
    print()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ==================================================================
    # STEP 3: TRAIN EACH MODEL
    # ==================================================================
    all_model_results = {}

    for model_idx, model_config in enumerate(MODELS):
        print(f"\n{'*' * 80}")
        print(f"  TRAINING MODEL {model_idx+1}/{len(MODELS)}: "
              f"{model_config['name']}")
        elapsed = time.time() - pipeline_start
        print(f"  Pipeline elapsed: {elapsed/60:.0f} min")
        print(f"{'*' * 80}")

        result = train_model_cv(
            model_config, manifest, folds, device, pipeline_start,
            all_model_results
        )
        all_model_results[model_config["name"]] = result

    # ==================================================================
    # STEP 4: CROSS-MODEL COMPARISON
    # ==================================================================
    pipeline_time = time.time() - pipeline_start

    print("\n" + "=" * 80)
    print("  STEP 4: CROSS-MODEL COMPARISON")
    print("=" * 80)

    metric_keys = ["sensitivity", "specificity", "precision",
                   "accuracy", "auc_roc", "f1"]

    # Comparison table
    print(f"\n  {'Model':<15} {'Params':<10}", end="")
    for key in metric_keys:
        print(f" {key:<15}", end="")
    print(f" {'Time':<10}")
    print(f"  {'-' * (15 + 10 + 15 * len(metric_keys) + 10)}")

    for name, res in all_model_results.items():
        params_str = f"{res['total_params']/1e6:.2f}M"
        time_str = f"{res['total_time']/60:.0f}m"
        print(f"  {name:<15} {params_str:<10}", end="")
        for key in metric_keys:
            m = res["summary"][key]["mean"]
            s = res["summary"][key]["std"]
            print(f" {m:.3f}+/-{s:.3f} ", end="")
        print(f" {time_str:<10}")

    # ── Cross-model comparison ROC plot ──────────────────────────────
    model_colors = {"ResNet3D": "#2196F3", "VGG3D": "#FF5722"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for name, res in all_model_results.items():
        agg_fpr, agg_tpr, _ = roc_curve(res["agg_labels"], res["agg_proba"])
        agg_auc = auc(agg_fpr, agg_tpr)
        color = model_colors.get(name, "#333333")
        ax.plot(agg_fpr, agg_tpr, color=color, linewidth=2.5,
                label=f"{name} (AUC = {agg_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"Model Comparison - {ORGAN_NAME} ({K_FOLDS}-Fold CV)",
                 fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    comparison_roc_path = os.path.join(PLOT_DIR, "comparison_roc.png")
    plt.tight_layout()
    plt.savefig(comparison_roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison ROC saved: {comparison_roc_path}")

    # ── Save results summary ─────────────────────────────────────────
    results_path = os.path.join(PLOT_DIR, "results_summary.txt")
    with open(results_path, "w") as f:
        f.write(f"{ORGAN_NAME} Classification - Multi-Model "
                f"{K_FOLDS}-Fold CV Results\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total pipeline time: {pipeline_time/60:.1f} min "
                f"({pipeline_time/3600:.1f} hours)\n\n")

        for name, res in all_model_results.items():
            f.write(f"\n{'=' * 70}\n")
            f.write(f"MODEL: {name} ({res['arch']})\n")
            f.write(f"Parameters: {res['total_params']:,}\n")
            f.write(f"Training time: {res['total_time']/60:.1f} min\n")
            f.write(f"{'=' * 70}\n\n")

            f.write(f"{'Metric':<15}")
            for i in range(K_FOLDS):
                f.write(f" {'Fold '+str(i+1):>8}")
            f.write(f" {'Mean':>8} {'Std':>8}\n")
            f.write(f"{'-' * (15 + 9 * K_FOLDS + 18)}\n")

            for key in metric_keys:
                s = res["summary"][key]
                f.write(f"{key:<15}")
                for v in s["values"]:
                    f.write(f" {v:>8.4f}")
                f.write(f" {s['mean']:>8.4f} {s['std']:>8.4f}\n")

            f.write(f"\nAggregate ({len(res['agg_labels'])} test samples):\n")
            for key in metric_keys:
                f.write(f"  {key:<15} {res['agg_metrics'][key]:.4f}\n")

            f.write(f"\nPer-fold training:\n")
            for i in range(K_FOLDS):
                f.write(f"  Fold {i+1}: {res['fold_epochs'][i]} epochs, "
                        f"{res['fold_times'][i]/60:.1f} min\n")

        # Cross-model comparison table
        f.write(f"\n\n{'=' * 70}\n")
        f.write(f"CROSS-MODEL COMPARISON (Mean +/- Std)\n")
        f.write(f"{'=' * 70}\n\n")

        f.write(f"{'Model':<15} {'Params':<12}")
        for key in metric_keys:
            f.write(f" {key:<17}")
        f.write("\n")
        f.write(f"{'-' * (15 + 12 + 17 * len(metric_keys))}\n")

        for name, res in all_model_results.items():
            params_str = f"{res['total_params']/1e6:.2f}M"
            f.write(f"{name:<15} {params_str:<12}")
            for key in metric_keys:
                m = res["summary"][key]["mean"]
                s = res["summary"][key]["std"]
                f.write(f" {m:.4f}+/-{s:.4f} ")
            f.write("\n")

    print(f"  Results summary saved: {results_path}")

    # ==================================================================
    # STEP 5: RESEARCH SUMMARY
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("  RESEARCH SUMMARY (copy these into your paper)")
    print("=" * 80)

    print(f"\n  METHODOLOGY:")
    print(f"    Organ:           Liver")
    print(f"    Dataset:         LiTS (Liver Tumor Segmentation Challenge)")
    print(f"    Task:            Binary classification ({POS_LABEL} vs {NEG_LABEL})")
    print(f"    Validation:      {K_FOLDS}-fold cross-validation")
    print(f"    Split strategy:  Patient-level (prevents data leakage)")
    print(f"    Total patches:   {len(manifest)} "
          f"({n_pos} tumor, {n_neg} non-tumor)")
    print(f"    Unique patients: {len(unique_uids)}")
    print(f"    Patch size:      64x64x64 at 1mm isotropic")
    print(f"    HU window:       [-200, 300]")
    print(f"    Class balance:   WeightedRandomSampler (50/50 in each batch)")
    print(f"    Val fraction:    {VAL_FRACTION:.0%} of non-test patients")
    loss_name = f"Focal Loss (gamma={FOCAL_GAMMA})" if USE_FOCAL_LOSS else "CrossEntropyLoss"
    print(f"    Loss function:   {loss_name}")
    print(f"    Augmentation:    3D flips (all axes), 90-deg rotation (all planes), "
          f"Gaussian noise, intensity shift/scale")
    print(f"    Test-time aug:   7 geometric transforms averaged")

    for name, res in all_model_results.items():
        print(f"\n  MODEL: {name}")
        print(f"    Architecture:    {name}")
        print(f"    Parameters:      {res['total_params']/1e6:.2f}M "
              f"({res['total_params']:,})")
        print(f"    Input shape:     (1, 64, 64, 64) at 1mm isotropic")
        print(f"    Optimizer:       AdamW (lr={LR}, wd={WEIGHT_DECAY})")
        print(f"    Scheduler:       LinearWarmup({WARMUP_EPOCHS}ep) + CosineAnnealingLR")
        print(f"    Grad accumulation: {ACCUM_STEPS} steps (effective batch={BATCH_SIZE*ACCUM_STEPS})")
        print(f"    Early stopping:  patience={PATIENCE}, min_epochs={MIN_EPOCHS}")

        print(f"\n    RESULTS (mean +/- std across {K_FOLDS} folds):")
        print(f"    {'─' * 35}")
        for key in metric_keys:
            s = res["summary"][key]
            print(f"    {key:<15} {s['mean']:.4f} +/- {s['std']:.4f}")
        print(f"    {'─' * 35}")

        print(f"\n    TIMING:")
        for i in range(K_FOLDS):
            print(f"    Fold {i+1}: {res['fold_epochs'][i]:2d} epochs, "
                  f"{res['fold_times'][i]/60:.1f} min")
        print(f"    Total: {res['total_time']/60:.1f} min")

    print(f"\n  TOTAL PIPELINE TIME: {pipeline_time/60:.1f} min "
          f"({pipeline_time/3600:.1f} hours)")

    print(f"\n  ARTIFACTS:")
    for name, res in all_model_results.items():
        print(f"    {name}:")
        for i in range(K_FOLDS):
            ckpt = os.path.join(CHECKPOINT_DIR, res['arch'],
                                f"fold_{i}", "best_model.pth")
            print(f"      Fold {i+1} model: {ckpt}")
        print(f"      ROC curve:       {res['roc_path']}")
        print(f"      Training curves: {res['curves_path']}")
        gradcam_dir = os.path.join(PLOT_DIR, "gradcam", res['arch'])
        print(f"      Grad-CAM:        {gradcam_dir}")
    print(f"    Comparison ROC:    {comparison_roc_path}")
    print(f"    Results text:      {results_path}")

    print(f"\n{'=' * 80}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
