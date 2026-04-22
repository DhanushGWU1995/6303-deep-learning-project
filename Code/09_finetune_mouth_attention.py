"""
09_finetune_mouth_attention.py
──────────────────────────────
Fine-tune the best pre-trained model (VGG-16 or ResNet-50) with a lightweight
Convolutional Block Attention Module (CBAM) whose spatial head is initialised
with a soft lower-face bias.  The backbone is **frozen**; only the attention
module + classifier head are updated.

This is the recommended path when you want to redirect Grad-CAM heat maps
toward the mouth region WITHOUT a full retrain from scratch.

Usage
─────
On the EC2 GPU instance (kernel 6.8.0-1031-aws):

    # Fine-tune VGG-16 (default, best accuracy)
    python3 09_finetune_mouth_attention.py --model vgg16 --epochs 20

    # Fine-tune ResNet-50
    python3 09_finetune_mouth_attention.py --model resnet50 --epochs 20

    # Only use mouth-emphasised images already in data/processed_mouth/
    python3 09_finetune_mouth_attention.py --model vgg16 --data-dir data/processed_mouth

Outputs
───────
    models/<arch>_mouth_attention_best.pth   ← best checkpoint
    models/<arch>_mouth_attention_last.pth   ← final epoch checkpoint
    results/<arch>_mouth_attention_history.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score

# ── Code/ directory — same base as scripts 02–05 ────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
MODELS_DIR = CODE_DIR / "models"
RESULTS_DIR = CODE_DIR / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NUM_CLASSES, IMG_SIZE, CLASS_NAMES


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Spatial Attention Module with lower-face prior
# ═══════════════════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    """
    Lightweight spatial attention (3×3 conv over avg+max pooled feature maps).
    The bias of the final sigmoid-activated conv is initialised so that the
    lower half of the spatial map receives higher initial attention — guiding
    the network toward mouth/chin Action Units from the start of fine-tuning.
    """
    def __init__(self, kernel_size: int = 7, img_rows: int = 7,
                 mouth_boost: float = 0.5, upper_dampen: float = -0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=True)

        # Initialise bias to a lower-face prior (sigmoid(x) ≈ 0.62 for x=0.5)
        with torch.no_grad():
            rows = torch.arange(img_rows, dtype=torch.float32)
            midpoint = img_rows * 0.50
            steepness = 8.0 / img_rows
            sig = torch.sigmoid(steepness * (rows - midpoint))
            prior = upper_dampen + (mouth_boost - upper_dampen) * sig  # (H,)
            # Set the conv bias (shape 1) to the mean prior — per-spatial via
            # weight initialisation below
            self.conv.bias.data.fill_(prior.mean().item())

            # Skew the kernel weights so top rows get less weight
            prior_2d = prior.view(1, 1, -1, 1).expand(1, 2, img_rows, img_rows)
            # Scale existing Kaiming weights by the spatial prior
            self.conv.weight.data *= prior_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(combined))
        return x * attn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class CBAM(nn.Module):
    """Full CBAM: channel attention → spatial attention."""
    def __init__(self, in_channels: int, spatial_rows: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention(img_rows=spatial_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Model builders with CBAM inserted before classifier
# ═══════════════════════════════════════════════════════════════════════════

class VGG16WithAttention(nn.Module):
    def __init__(self, num_classes: int, pretrained_path: str | None = None):
        super().__init__()
        # base = models.vgg16(weights=None)
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = base.features          # outputs (B, 512, 7, 7) for 224px
        self.cbam = CBAM(512, spatial_rows=7)
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        if pretrained_path:
            self._load_weights(pretrained_path)

    # def _load_weights(self, path: str):
    #     state = torch.load(path, map_location="cpu")
    #     # Load matching keys only (backbone); ignore head size mismatches
    #     own = self.state_dict()
    #     filtered = {k: v for k, v in state.items()
    #                 if k in own and own[k].shape == v.shape}
    #     own.update(filtered)
    #     self.load_state_dict(own)
    #     print(f"[INFO] Loaded {len(filtered)}/{len(own)} keys from {path}")

    def _load_weights(self, path: str):
        state = torch.load(path, map_location="cpu")

        # handle checkpoint dict
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        # load partially (important)
        missing, unexpected = self.load_state_dict(state, strict=False)

        print(f"[INFO] Loaded weights from {path}")
        print(f"       Missing keys   : {len(missing)}")
        print(f"       Unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet50WithAttention(nn.Module):
    def __init__(self, num_classes: int, pretrained_path: str | None = None):
        super().__init__()
        # base = models.resnet50(weights=None)
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Keep everything except the final FC layer
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # (B,2048,7,7)
        self.cbam = CBAM(2048, spatial_rows=7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        if pretrained_path:
            self._load_weights(pretrained_path)

    # def _load_weights(self, path: str):
    #     state = torch.load(path, map_location="cpu")
    #     own = self.state_dict()
    #     filtered = {k: v for k, v in state.items()
    #                 if k in own and own[k].shape == v.shape}
    #     own.update(filtered)
    #     self.load_state_dict(own)
    #     print(f"[INFO] Loaded {len(filtered)}/{len(own)} keys from {path}")

    def _load_weights(self, path: str):
        state = torch.load(path, map_location="cpu")

        # handle checkpoint dict
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        # load partially (important)
        missing, unexpected = self.load_state_dict(state, strict=False)

        print(f"[INFO] Loaded weights from {path}")
        print(f"       Missing keys   : {len(missing)}")
        print(f"       Unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Data loaders
# ═══════════════════════════════════════════════════════════════════════════

def build_loaders(data_dir: str, batch_size: int = 32):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), val_tf)
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
    #                           num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
    #                         num_workers=4, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=max(4, os.cpu_count() // 2),
                              pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(4, os.cpu_count() // 2),
                            pin_memory=True,
                            persistent_workers=True)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(4, os.cpu_count() // 2),
    #                          pin_memory=True,
    #                          persistent_workers=True)


    print(f"[INFO] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"[INFO] Classes: {train_ds.classes}")
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


# @torch.no_grad()
# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss, correct, total = 0.0, 0, 0
#     for imgs, labels in loader:
#         imgs, labels = imgs.to(device), labels.to(device)
#         out = model(imgs)
#         total_loss += criterion(out, labels).item() * imgs.size(0)
#         correct += (out.argmax(1) == labels).sum().item()
#         total += imgs.size(0)
#     return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)

        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="macro")

    return total_loss / total, acc, f1


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vgg16", "resnet50"], default="vgg16")
    p.add_argument("--data-dir", default=str(CODE_DIR / "data" / "processed"),
                   help="Root directory with train/ and test/ sub-folders (pain classes)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for attention + classifier (backbone frozen)")
    p.add_argument("--unfreeze-after", type=int, default=10,
                   help="Epoch at which to unfreeze backbone for joint fine-tuning")
    p.add_argument("--lr-backbone", type=float, default=1e-5,
                   help="Learning rate applied to backbone after unfreezing")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── resolve pretrained weights ──────────────────────────────────────────
    arch_map = {
        "vgg16": ("vgg16_best.pth", VGG16WithAttention),
        "resnet50": ("resnet50_best.pth", ResNet50WithAttention),
    }
    weight_name, ModelClass = arch_map[args.model]
    pretrained_path = MODELS_DIR / weight_name
    if not pretrained_path.exists():
        print(f"[WARN] No existing weights at {pretrained_path}; "
              "starting from ImageNet random init.")
        pretrained_path = None

    model = ModelClass(NUM_CLASSES, pretrained_path=str(pretrained_path)
                       if pretrained_path else None).to(device)

    # ── freeze backbone, only train CBAM + classifier ───────────────────────
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if name.startswith("cbam") or name.startswith("classifier"):
            head_params.append(param)
        else:
            param.requires_grad = False
            backbone_params.append(param)

    print(f"[INFO] Trainable params (CBAM + head): "
          f"{sum(p.numel() for p in head_params):,}")

    optimizer = optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = build_loaders(args.data_dir, args.batch_size)

    best_val_f1 = 0.0
    history = []
    save_stem = f"{args.model}_mouth_attention"

    for epoch in range(1, args.epochs + 1):

        # ── progressive unfreezing ─────────────────────────────────────────
        if epoch == args.unfreeze_after + 1:
            print(f"\n[INFO] Epoch {epoch}: unfreezing backbone "
                  f"(lr={args.lr_backbone})")
            for p in backbone_params:
                p.requires_grad = True
            optimizer.add_param_group(
                {"params": backbone_params, "lr": args.lr_backbone}
            )

        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                   vl_loss=vl_loss, vl_acc=vl_acc)
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train {tr_acc:.4f} | Val {vl_acc:.4f} | "
              f"Val F1 {vl_f1:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(),
                       MODELS_DIR / f"{save_stem}_best.pth")
            print(f"  ✓ New best val f1: {best_val_f1:.4f}")

    torch.save(model.state_dict(), MODELS_DIR / f"{save_stem}_last.pth")

    csv_path = RESULTS_DIR / f"{save_stem}_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"\n[DONE] Best val f1: {best_val_f1:.4f}")
    print(f"       Weights → models/{save_stem}_best.pth")
    print(f"       History → {csv_path}")


if __name__ == "__main__":
    main()
