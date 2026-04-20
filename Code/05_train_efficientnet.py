"""
05_train_efficientnet.py
------------------------
Transfer learning with EfficientNet-B3 pretrained on ImageNet.
Expected to be the best-performing model in the comparison.
Two-phase training: head → full fine-tune.
"""

import os
from collections import Counter
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from config import (
    PROCESSED_DATA_DIR, IMG_SIZE, NUM_CLASSES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, DROPOUT_RATE, DEVICE,
    MODEL_PATHS, PLOTS_DIR
)
from utils import run_training_loop, evaluate_model

os.makedirs("models", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_efficientnet(num_classes: int = NUM_CLASSES, dropout: float = DROPOUT_RATE):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.SiLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    print(f"[Device] {DEVICE}")
    train_tf, val_tf = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(PROCESSED_DATA_DIR, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(PROCESSED_DATA_DIR, "val"),   val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(PROCESSED_DATA_DIR, "test"),  val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Compute class weights to handle imbalance (severe >> mild/moderate)
    targets = [label for _, label in train_ds.samples]
    counts  = Counter(targets)
    total   = len(targets)
    weights = torch.FloatTensor([
        total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)
    ]).to(DEVICE)
    print(f"[Info] Class weights: {weights.cpu().numpy().round(3)}")

    model = build_efficientnet().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    head_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    print("\n[Phase 1] Training EfficientNet-B3 head only")
    run_training_loop(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, DEVICE, "efficientnet_phase1",
        MODEL_PATHS["efficientnet"], NUM_EPOCHS // 2, EARLY_STOPPING_PATIENCE
    )

    print("\n[Phase 2] Fine-tuning all EfficientNet-B3 layers (lower LR)")
    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.05,
                                    weight_decay=WEIGHT_DECAY)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    best_acc = run_training_loop(
        model, train_loader, val_loader, criterion, optimizer2,
        scheduler2, DEVICE, "efficientnet",
        MODEL_PATHS["efficientnet"], NUM_EPOCHS, EARLY_STOPPING_PATIENCE
    )
    print(f"\n[Done] Best Val Acc: {best_acc:.2f}%")

    model.load_state_dict(torch.load(MODEL_PATHS["efficientnet"], map_location=DEVICE))
    evaluate_model(model, test_loader, DEVICE, "efficientnet")


if __name__ == "__main__":
    main()