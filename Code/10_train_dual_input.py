"""
10_train_dual_input.py
──────────────────────
Dual-input CNN + MLP architecture for facial pain detection.

Architecture
────────────
  ┌─────────────────────────┐     ┌────────────────────────────┐
  │  Full Face (224×224)    │     │  Mouth Crop (224×224)      │
  │  ResNet-50 encoder      │     │  ResNet-18 encoder         │
  │  → 2048-d features      │     │  (lower 45 % of face)      │
  └──────────┬──────────────┘     │  → 512-d features          │
             │                    └──────────┬─────────────────┘
             └──────────┐  ┌────────────────┘
                        ▼  ▼
              concat → 2560-d vector
                        │
              Linear(2560 → 512) → ReLU → Dropout(0.5)
              Linear(512  → 128) → ReLU → Dropout(0.3)
              Linear(128  →   4)              [MLP head]

Why this design
───────────────
  • Satisfies the course's CNN + MLP architecture requirement explicitly.
  • Mouth-crop branch forces the model to extract pain-relevant Action Units
    (AU20, AU17, AU25) from the lip/chin region.
  • Two-branch fusion is interpretable: Grad-CAM on each branch can be
    compared side-by-side to show what each encoder attends to.
  • No extra landmark libraries — the mouth crop uses a simple heuristic
    (rows 55 %–100 % of the already-aligned face image), which is reliable
    because AffectNet images are tightly face-cropped.

Usage
─────
  # Phase 1 only (freeze both backbones, train MLP head — fast)
  python3 10_train_dual_input.py --epochs 15 --unfreeze-after 999

  # Full training: head first, then joint fine-tune
  python3 10_train_dual_input.py --epochs 30 --unfreeze-after 15

  # Custom data dir
  python3 10_train_dual_input.py --data-dir data/raw --epochs 30

Outputs
───────
  models/dual_input_best.pth          ← best checkpoint (by val accuracy)
  models/dual_input_last.pth          ← final epoch checkpoint
  results/dual_input_history.csv      ← per-epoch metrics
  plots/dual_input_training_curves.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score

# CODE_DIR = the Code/ folder — same base as all other training scripts
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from config import (
    NUM_CLASSES, IMG_SIZE, DEVICE, MODELS_DIR, RESULTS_DIR,
    PLOTS_DIR, CLASS_NAMES
)

# Paths relative to Code/ so model files land in Code/models/ alongside
# vgg16_best.pth, resnet50_best.pth, etc. produced by scripts 02–05.
MODELS_PATH = CODE_DIR / MODELS_DIR
RESULTS_PATH = CODE_DIR / RESULTS_DIR
PLOTS_PATH = CODE_DIR / PLOTS_DIR
for d in (MODELS_PATH, RESULTS_PATH, PLOTS_PATH):
    d.mkdir(parents=True, exist_ok=True)

# ── Mouth-crop hyper-parameters ─────────────────────────────────────────────
# After the face is detected and resized to IMG_SIZE × IMG_SIZE:
#   • rows 0 … (MOUTH_START_FRAC × IMG_SIZE)  = upper face (eyes/brows)
#   • rows (MOUTH_START_FRAC × IMG_SIZE) … end = nose / mouth / chin
MOUTH_START_FRAC = 0.55   # start of crop — just below the nose bridge
MOUTH_END_FRAC   = 1.00   # crop to bottom of face

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Transforms
# ═══════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_FULL_FACE_TRAIN = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_FULL_FACE_VAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# Mouth crop: resize the 45 % strip to full IMG_SIZE so ResNet-18 sees enough
# detail.  We apply the SAME horizontal flip as the full-face branch so
# left/right symmetry is preserved.

class MouthCropTransform:
    """
    Crop the lower ``end_frac - start_frac`` fraction of a PIL image
    (the mouth / chin region) and resize to ``size × size``.

    Applied AFTER the full-face transform's random flip by wrapping a
    shared flag — but since PIL images aren't flipped yet at this stage we
    simply apply the same RandomHorizontalFlip independently (statistically
    equivalent at scale).
    """
    def __init__(self, size: int, start_frac: float, end_frac: float,
                 augment: bool = False):
        self.size = size
        self.start_frac = start_frac
        self.end_frac = end_frac
        base = [
            transforms.Resize((size, size)),
        ]
        if augment:
            base += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.25, contrast=0.2),
            ]
        base += [
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
        self.pipeline = transforms.Compose(base)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        w, h = pil_img.size
        top    = int(self.start_frac * h)
        bottom = int(self.end_frac   * h)
        mouth_crop = pil_img.crop((0, top, w, bottom))
        return self.pipeline(mouth_crop)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Dataset — wraps ImageFolder and returns (full_face, mouth_crop, label)
# ═══════════════════════════════════════════════════════════════════════════

class DualInputDataset(Dataset):
    """
    Wraps ``torchvision.datasets.ImageFolder``.

    For each sample returns:
        full_tensor  : full face transformed to IMG_SIZE × IMG_SIZE (3, H, W)
        mouth_tensor : lower-face crop resized to IMG_SIZE × IMG_SIZE (3, H, W)
        label        : integer class index
    """

    def __init__(self, root: str, full_tf, mouth_tf):
        self._base = datasets.ImageFolder(root)
        self.full_tf  = full_tf
        self.mouth_tf = mouth_tf

    def __len__(self) -> int:
        return len(self._base)

    @property
    def classes(self):
        return self._base.classes

    def __getitem__(self, idx: int):
        path, label = self._base.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.full_tf(img), self.mouth_tf(img), label


def build_loaders(data_dir: str, batch_size: int = 32):
    mouth_train_tf = MouthCropTransform(
        IMG_SIZE, MOUTH_START_FRAC, MOUTH_END_FRAC, augment=True)
    mouth_val_tf = MouthCropTransform(
        IMG_SIZE, MOUTH_START_FRAC, MOUTH_END_FRAC, augment=False)

    train_ds = DualInputDataset(
        os.path.join(data_dir, "train"),
        _FULL_FACE_TRAIN, mouth_train_tf
    )
    val_ds = DualInputDataset(
        os.path.join(data_dir, "test"),
        _FULL_FACE_VAL, mouth_val_tf
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=max(4, os.cpu_count() // 2), pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(4, os.cpu_count() // 2), pin_memory=True, persistent_workers=True)

    print(f"[INFO] Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"[INFO] Classes: {train_ds.classes}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Dual-Input Model
# ═══════════════════════════════════════════════════════════════════════════

class DualInputPainNet(nn.Module):
    """
    Two-branch CNN + MLP fusion network.

    Branch A — Full-face encoder  : ResNet-50  → 2048-d
    Branch B — Mouth-crop encoder : ResNet-18  →  512-d
    MLP head                      : 2560 → 512 → 128 → num_classes

    Parameters
    ──────────
    num_classes   : number of output classes (default 4)
    freeze_at_init: freeze both backbones on construction (True for phase 1)
    """

    FULL_FACE_DIM = 2048
    MOUTH_DIM     = 512
    FUSED_DIM     = FULL_FACE_DIM + MOUTH_DIM  # 2560

    def __init__(self, num_classes: int = 4, freeze_at_init: bool = True):
        super().__init__()

        # ── Branch A : ResNet-50 (full face) ────────────────────────────
        resnet50 = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final FC layer; keep the global avg-pool
        self.full_face_encoder = nn.Sequential(
            *list(resnet50.children())[:-1]   # (B, 2048, 1, 1)
        )

        # ── Branch B : ResNet-18 (mouth crop) ───────────────────────────
        resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.mouth_encoder = nn.Sequential(
            *list(resnet18.children())[:-1]   # (B, 512, 1, 1)
        )

        # ── MLP fusion head ─────────────────────────────────────────────
        self.mlp_head = nn.Sequential(
            nn.Linear(self.FUSED_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        if freeze_at_init:
            self._set_backbone_grad(False)

    # ── helpers ─────────────────────────────────────────────────────────

    def _set_backbone_grad(self, requires_grad: bool):
        for p in self.full_face_encoder.parameters():
            p.requires_grad = requires_grad
        for p in self.mouth_encoder.parameters():
            p.requires_grad = requires_grad

    def freeze_backbones(self):
        self._set_backbone_grad(False)

    def unfreeze_backbones(self):
        self._set_backbone_grad(True)

    # ── forward ─────────────────────────────────────────────────────────

    def forward(self, full_face: torch.Tensor,
                mouth_crop: torch.Tensor) -> torch.Tensor:
        """
        Args:
            full_face  : (B, 3, H, W) full-face tensor
            mouth_crop : (B, 3, H, W) mouth-region crop tensor

        Returns:
            logits     : (B, num_classes)
        """
        feat_full  = self.full_face_encoder(full_face).flatten(1)   # (B, 2048)
        feat_mouth = self.mouth_encoder(mouth_crop).flatten(1)       # (B,  512)
        fused = torch.cat([feat_full, feat_mouth], dim=1)            # (B, 2560)
        return self.mlp_head(fused)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Training helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for full, mouth, labels in loader:
        full, mouth, labels = full.to(device), mouth.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(full, mouth)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * full.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += full.size(0)
    return total_loss / total, correct / total


# @torch.no_grad()
# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss, correct, total = 0.0, 0, 0
#     for full, mouth, labels in loader:
#         full, mouth, labels = full.to(device), mouth.to(device), labels.to(device)
#         logits = model(full, mouth)
#         total_loss += criterion(logits, labels).item() * full.size(0)
#         correct    += (logits.argmax(1) == labels).sum().item()
#         total      += full.size(0)
#     return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for full, mouth, labels in loader:
        full, mouth, labels = full.to(device), mouth.to(device), labels.to(device)

        logits = model(full, mouth)
        total_loss += criterion(logits, labels).item() * full.size(0)

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += full.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="macro")

    return total_loss / total, acc, f1


def plot_history(history: list[dict], out_path: str):
    epochs = [r["epoch"] for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [r["tr_acc"] for r in history],  label="Train")
    ax1.plot(epochs, [r["vl_acc"] for r in history],  label="Val", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Dual-Input Model — Accuracy"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, [r["tr_loss"] for r in history], label="Train")
    ax2.plot(epochs, [r["vl_loss"] for r in history], label="Val",   linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Dual-Input Model — Loss"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the dual-input (full-face + mouth-crop) CNN+MLP model.")
    p.add_argument("--data-dir",
                   default=str(CODE_DIR / "data" / "processed"),
                   help="Root with train/ and test/ sub-folders (pain classes)")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr-head",       type=float, default=3e-4,
                   help="LR for MLP head while backbones are frozen")
    p.add_argument("--lr-backbone",   type=float, default=1e-5,
                   help="LR for backbones after unfreezing")
    p.add_argument("--unfreeze-after",type=int,   default=15,
                   help="Epoch at which both backbones are unfrozen (0 = never frozen)")
    p.add_argument("--resume",        default=None,
                   help="Path to a checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    print(f"[INFO] Data   : {args.data_dir}")

    model = DualInputPainNet(
        num_classes=NUM_CLASSES,
        freeze_at_init=(args.unfreeze_after > 0)
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[INFO] Resumed from {args.resume}")

    trainable_head  = list(model.mlp_head.parameters())
    backbone_params = (list(model.full_face_encoder.parameters()) +
                       list(model.mouth_encoder.parameters()))

    print(f"[INFO] MLP head params   : {sum(p.numel() for p in trainable_head):,}")
    print(f"[INFO] Backbone params   : {sum(p.numel() for p in backbone_params):,}")

    optimizer = optim.AdamW(trainable_head, lr=args.lr_head, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = build_loaders(args.data_dir, args.batch_size)

    best_val_f1 = 0.0
    history      = []

    for epoch in range(1, args.epochs + 1):

        # ── progressive unfreezing ─────────────────────────────────────
        if args.unfreeze_after > 0 and epoch == args.unfreeze_after + 1:
            print(f"\n[INFO] Epoch {epoch}: unfreezing both backbones "
                  f"(lr={args.lr_backbone})")
            model.unfreeze_backbones()
            optimizer.add_param_group(
                {"params": backbone_params, "lr": args.lr_backbone}
            )

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1 = evaluate(
            model, val_loader,   criterion, device)
        scheduler.step()

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                   vl_loss=vl_loss, vl_acc=vl_acc, vl_f1=vl_f1)
        history.append(row)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"│ train {tr_acc:.4f} / {tr_loss:.4f}  "
              f"│ val   {vl_acc:.4f} / {vl_loss:.4f}  "
              f"| f1 macro val {vl_f1:.4f}"
              f"│ lr {lr_now:.2e}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(),
                       str(MODELS_PATH / "dual_input_best.pth"))
            print(f"  #> Best val f1: {best_val_f1:.4f}")

    torch.save(model.state_dict(), str(MODELS_PATH / "dual_input_last.pth"))

    csv_path = RESULTS_PATH / "dual_input_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"[SAVED] {csv_path}")

    plot_path = str(PLOTS_PATH / "dual_input_training_curves.png")
    plot_history(history, plot_path)

    print(f"\n{'='*55}")
    print(f"  Final best val f1 macro : {best_val_f1:.4f}")
    print(f"  Weights → models/dual_input_best.pth")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
