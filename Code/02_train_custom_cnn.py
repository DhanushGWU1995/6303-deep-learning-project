"""
02_train_custom_cnn.py
----------------------
Baseline custom CNN trained from scratch on the pain dataset.
Architecture: 4 convolutional blocks + global average pooling + MLP classifier.
"""

import os
from collections import Counter
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    PROCESSED_DATA_DIR, IMG_SIZE, NUM_CLASSES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, DEVICE, MODEL_PATHS, PLOTS_DIR
)
from utils import run_training_loop, evaluate_model
from train_custom_cnn import CustomCNN

os.makedirs("models", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers= max(4, os.cpu_count() // 2), pin_memory= True,
              persistent_workers= True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers= max(4, os.cpu_count() // 2), pin_memory= True,
              persistent_workers= True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers= max(4, os.cpu_count() // 2), pin_memory= True,
              persistent_workers= True)

    # Compute class weights to handle imbalance (severe >> mild/moderate)
    targets = [label for _, label in train_ds.samples]
    counts  = Counter(targets)
    total   = len(targets)
    weights = torch.FloatTensor([
        total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)
    ]).to(DEVICE)
    print(f"[Info] Class weights: {weights.cpu().numpy().round(3)}")

    model = CustomCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    print("\n[Training] Custom CNN")
    best_acc = run_training_loop(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, DEVICE, "custom_cnn",
        MODEL_PATHS["custom_cnn"], NUM_EPOCHS, EARLY_STOPPING_PATIENCE
    )
    print(f"\n[Done] Best Val F1: {best_acc:.2f}%")

    model.load_state_dict(torch.load(MODEL_PATHS["custom_cnn"], map_location=DEVICE))
    evaluate_model(model, test_loader, DEVICE, "custom_cnn")


if __name__ == "__main__":
    main()
