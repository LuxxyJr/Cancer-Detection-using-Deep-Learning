"""
Training with validation, LR scheduling, gradient clipping, gradient accumulation,
focal loss, and conditional AMP.
Designed for RTX 4050 (6GB VRAM) -- uses mixed precision when CUDA is available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma=0 reduces to weighted cross-entropy
    - Higher gamma downweights easy examples, focusing on hard ones
    - alpha provides per-class weighting (like CE class weights)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # Probability of the correct class
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce_loss).mean()


class Trainer:

    def __init__(self, model, device="cuda", epochs=50, lr=1e-3, weight_decay=1e-4,
                 class_weights=None, loss_type="cross_entropy", focal_gamma=2.0,
                 accum_steps=1, warmup_epochs=0):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.accum_steps = accum_steps

        # Loss function
        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Ensure criterion is on correct device (for FocalLoss alpha buffer)
        if isinstance(self.criterion, nn.Module):
            self.criterion = self.criterion.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # LR schedule: optional linear warmup + cosine annealing
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(epochs - warmup_epochs, 1)
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )

        # Mixed precision only on CUDA
        self.use_amp = (device == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def train_epoch(self, loader):
        """Train for one epoch with gradient accumulation. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for step, (x, y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    loss = self.criterion(logits, y) / self.accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == len(loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y) / self.accum_steps

                loss.backward()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps  # Undo scaling for logging
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        self.scheduler.step()

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, loader):
        """Evaluate on validation/test set. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
