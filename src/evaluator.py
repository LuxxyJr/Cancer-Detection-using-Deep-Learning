"""
Evaluation Metrics - Calculate and visualize classification performance
Handles edge cases (single-class predictions, empty arrays)
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt


class MetricsCalculator:
    """Calculate and display evaluation metrics for binary classification"""

    @staticmethod
    def calculate_metrics(y_true, y_pred_proba, y_pred_class):
        """
        Calculate all important metrics.

        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Predicted probabilities for positive class [0-1]
            y_pred_class: Predicted class (0 or 1)

        Returns:
            Dictionary with all metrics
        """
        # Convert tensors to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred_proba, torch.Tensor):
            y_pred_proba = y_pred_proba.cpu().numpy()
        if isinstance(y_pred_class, torch.Tensor):
            y_pred_class = y_pred_class.cpu().numpy()

        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        y_pred_class = np.asarray(y_pred_class)

        # Confusion matrix with edge case handling
        labels = [0, 1]
        cm = confusion_matrix(y_true, y_pred_class, labels=labels)
        tn, fp, fn, tp = cm.ravel()

        # Core metrics (safe division)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = accuracy_score(y_true, y_pred_class)
        f1 = f1_score(y_true, y_pred_class, zero_division=0)

        # AUC-ROC (requires both classes present in y_true)
        unique_labels = np.unique(y_true)
        if len(unique_labels) >= 2:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        else:
            auc_roc = 0.0  # Cannot compute AUC with single class

        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,
            "f1": f1,
            "auc_roc": auc_roc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    @staticmethod
    def print_metrics(metrics, dataset_name=""):
        """Print formatted metrics table"""
        print("\n" + "=" * 60)
        print(f"  EVALUATION METRICS - {dataset_name}")
        print("=" * 60)
        print(f"  Sensitivity (Recall):  {metrics['sensitivity']:.4f}")
        print(f"  Specificity:           {metrics['specificity']:.4f}")
        print(f"  Precision:             {metrics['precision']:.4f}")
        print(f"  Accuracy:              {metrics['accuracy']:.4f}")
        print(f"  F1 Score:              {metrics['f1']:.4f}")
        print(f"  AUC-ROC:               {metrics['auc_roc']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    True Positives:  {metrics['tp']:4d}  (Caught nodules)")
        print(f"    True Negatives:  {metrics['tn']:4d}  (Correct normals)")
        print(f"    False Positives: {metrics['fp']:4d}  (False alarms)")
        print(f"    False Negatives: {metrics['fn']:4d}  (Missed nodules)")
        print("=" * 60 + "\n")

    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, save_path="roc_curve.png"):
        """Plot and save ROC curve"""
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            print(f"  Warning: Cannot plot ROC curve (only class {unique_labels} present)")
            return

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve - Nodule Classification", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  ROC curve saved: {save_path}")

    @staticmethod
    def compare_models(results_dict):
        """Compare multiple models side by side"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"{'Model':<30} {'Sensitivity':<15} {'Specificity':<15} {'AUC-ROC':<15}")
        print("-" * 80)
        for name, m in results_dict.items():
            print(f"{name:<30} {m['sensitivity']:<15.4f} "
                  f"{m['specificity']:<15.4f} {m['auc_roc']:<15.4f}")
        print("=" * 80 + "\n")
