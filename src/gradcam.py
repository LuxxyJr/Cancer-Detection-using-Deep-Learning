"""
Grad-CAM for 3D medical image classification.

Generates class activation heatmaps showing which regions of the CT patch
the model focuses on when making its prediction. Produces publication-ready
overlay figures for TP, TN, FP, FN examples.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class GradCAM3D:
    """
    Grad-CAM for 3D CNNs.

    Registers forward/backward hooks on a target convolutional layer,
    computes gradient-weighted activation maps, and resizes to input dims.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained nn.Module (must be in eval mode)
            target_layer: nn.Module -- the conv layer to hook
                          (e.g. model.encoder.b4 for ResNet,
                           model.block3 for VGG)
        """
        self.model = model
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a single input.

        Args:
            input_tensor: (1, 1, D, H, W) tensor on correct device
            target_class: Class index to explain (default: predicted class)

        Returns:
            cam: numpy array (D, H, W) in [0, 1], same spatial size as input
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Grad-CAM computation
        # gradients shape: (1, C, d, h, w)
        # activations shape: (1, C, d, h, w)
        gradients = self._gradients[0]     # (C, d, h, w)
        activations = self._activations[0]  # (C, d, h, w)

        # Global average pool gradients over spatial dims -> channel weights
        weights = gradients.mean(dim=(1, 2, 3))  # (C,)

        # Weighted sum of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (keep only positive influence)
        cam = F.relu(cam)

        # Resize to input spatial dims
        input_shape = input_tensor.shape[2:]  # (D, H, W)
        cam = cam.unsqueeze(0).unsqueeze(0)   # (1, 1, d, h, w)
        cam = F.interpolate(cam, size=input_shape, mode="trilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Clean up hooks to avoid memory leaks"""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _get_target_layer(model, model_name):
    """
    Get the appropriate target layer for Grad-CAM based on model type.

    Args:
        model: nn.Module
        model_name: "resnet3d" or "vgg3d"

    Returns:
        nn.Module -- the target layer
    """
    if model_name == "resnet3d":
        return model.encoder.b4
    elif model_name == "vgg3d":
        return model.block3
    else:
        raise ValueError(f"Unknown model name for Grad-CAM: {model_name}")


def generate_gradcam_figures(model, model_name, dataloader, device, save_dir,
                             num_examples=3, use_amp=True,
                             pos_label="Nodule", neg_label="Non-nodule"):
    """
    Generate Grad-CAM overlay figures for publication.

    Collects TP, TN, FP, FN examples from the dataloader and produces
    side-by-side figures: [CT slice | Grad-CAM heatmap | Overlay].

    Args:
        model: Trained nn.Module (will be set to eval mode)
        model_name: "resnet3d" or "vgg3d" (determines target layer)
        dataloader: Test DataLoader
        device: "cuda" or "cpu"
        save_dir: Directory to save PNG files
        num_examples: Number of examples per category (TP/TN/FP/FN)
        use_amp: Whether to use mixed precision for inference
        pos_label: Display name for positive class (default "Nodule")
        neg_label: Display name for negative class (default "Non-nodule")
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    target_layer = _get_target_layer(model, model_name)
    gradcam = GradCAM3D(model, target_layer)

    # Collect examples by category
    categories = {"TP": [], "TN": [], "FP": [], "FN": []}

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)

            if use_amp and device == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(x_batch)
            else:
                logits = model(x_batch)

            proba = F.softmax(logits, dim=1)[:, 1]
            preds = (proba > 0.5).long()

            for i in range(x_batch.size(0)):
                true_label = y_batch[i].item()
                pred_label = preds[i].item()

                if true_label == 1 and pred_label == 1:
                    cat = "TP"
                elif true_label == 0 and pred_label == 0:
                    cat = "TN"
                elif true_label == 0 and pred_label == 1:
                    cat = "FP"
                else:
                    cat = "FN"

                if len(categories[cat]) < num_examples:
                    categories[cat].append(
                        (x_batch[i:i+1].clone(), true_label, pred_label,
                         proba[i].item())
                    )

            # Check if we have enough examples
            if all(len(v) >= num_examples for v in categories.values()):
                break

    # Generate figures for each category
    for cat_name, examples in categories.items():
        if not examples:
            print(f"    {cat_name}: No examples found (skipping)")
            continue

        for ex_idx, (x_single, true_label, pred_label, prob) in enumerate(examples):
            # Need gradients for Grad-CAM, so enable grad temporarily
            x_single = x_single.detach().requires_grad_(False)
            x_input = x_single.clone().detach().to(device).requires_grad_(True)

            # Generate heatmap (predicted class)
            cam = gradcam.generate(x_input, target_class=pred_label)

            # Get middle axial slice
            ct_vol = x_single[0, 0].cpu().numpy()  # (D, H, W)
            mid_slice = ct_vol.shape[0] // 2
            ct_slice = ct_vol[mid_slice]            # (H, W)
            cam_slice = cam[mid_slice]              # (H, W)

            # Create figure: [CT | Heatmap | Overlay]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # CT slice
            axes[0].imshow(ct_slice, cmap="gray", aspect="auto")
            axes[0].set_title("CT Slice (z={})".format(mid_slice), fontsize=11)
            axes[0].axis("off")

            # Grad-CAM heatmap
            im = axes[1].imshow(cam_slice, cmap="jet", aspect="auto",
                                norm=Normalize(0, 1))
            axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
            axes[1].axis("off")

            # Overlay
            axes[2].imshow(ct_slice, cmap="gray", aspect="auto")
            axes[2].imshow(cam_slice, cmap="jet", alpha=0.4, aspect="auto",
                           norm=Normalize(0, 1))
            axes[2].set_title("Overlay", fontsize=11)
            axes[2].axis("off")

            # Suptitle with prediction info
            label_str = pos_label if true_label == 1 else neg_label
            pred_str = pos_label if pred_label == 1 else neg_label
            fig.suptitle(
                f"{cat_name} | True: {label_str} | Pred: {pred_str} "
                f"(p={prob:.3f})",
                fontsize=13, fontweight="bold"
            )

            fname = f"{cat_name.lower()}_{ex_idx+1}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=150,
                        bbox_inches="tight")
            plt.close()

        print(f"    {cat_name}: {len(examples)} figure(s) saved")

    gradcam.remove_hooks()
    print(f"  Grad-CAM figures saved to: {save_dir}")
