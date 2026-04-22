import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

import seaborn as sns
from config import CLASSES, PAIN_LABELS, PLOTS_DIR, RESULTS_DIR

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# TRAINING PLOTS (FIXED: 3 subplots)
# =========================================================
def save_training_plots(train_losses, val_losses,
                       train_accs, val_accs,
                       train_f1s, val_f1s,
                       model_name):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # ✅ FIXED

    # LOSS
    axes[0].plot(train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(val_losses, label="Val Loss", color="tomato")
    axes[0].set_title(f"{model_name} — Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # ACCURACY
    axes[1].plot(train_accs, label="Train Acc", color="steelblue")
    axes[1].plot(val_accs, label="Val Acc", color="tomato")
    axes[1].set_title(f"{model_name} — Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    # F1 SCORE
    axes[2].plot(train_f1s, label="Train F1", color="steelblue")
    axes[2].plot(val_f1s, label="Val F1", color="tomato")
    axes[2].set_title(f"{model_name} — F1 Score per Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score (%)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[Saved] {path}")


# =========================================================
# CONFUSION MATRIX
# =========================================================
def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)

    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"[Saved] {path}")


# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(model, loader, device, model_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds) * 100
    f1   = f1_score(all_labels, all_preds, average="macro") * 100
    prec = precision_score(all_labels, all_preds, average="macro") * 100
    rec  = recall_score(all_labels, all_preds, average="macro") * 100

    report = classification_report(all_labels, all_preds,
                                   target_names=CLASSES, digits=4)

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"\nClassification Report:\n{report}")
    print(f"{'='*50}\n")

    save_confusion_matrix(all_labels, all_preds, model_name)

    result_path = os.path.join(RESULTS_DIR, f"{model_name}_results.txt")
    with open(result_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy  : {acc:.2f}%\n")
        f.write(f"F1 Score  : {f1:.2f}%\n")
        f.write(f"Precision : {prec:.2f}%\n")
        f.write(f"Recall    : {rec:.2f}%\n\n")
        f.write(report)

    return {
        "model": model_name,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }


# =========================================================
# TRAIN ONE EPOCH
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# =========================================================
# VALIDATION (WITH F1)
# =========================================================
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct, total = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = (correct / total) * 100

    val_f1 = f1_score(all_labels, all_preds, average="macro") * 100

    return val_loss, val_acc, val_f1


# =========================================================
# TRAINING LOOP (F1-BASED SAVING)
# =========================================================
def run_training_loop(model, train_loader, val_loader, criterion, optimizer,
                      scheduler, device, model_name, save_path,
                      num_epochs, patience):

    best_val_f1 = 0.0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(1, num_epochs + 1):

        # TRAIN
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # TRAIN F1 (extra pass, acceptable)
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        tr_f1 = f1_score(all_labels, all_preds, average="macro") * 100

        # VALIDATION
        vl_loss, vl_acc, vl_f1 = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        # STORE METRICS
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)
        train_f1s.append(tr_f1)
        val_f1s.append(vl_f1)

        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.2f}% | F1: {vl_f1:.2f}%")

        # SAVE BEST MODEL (F1)
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"  >> Best model saved (val F1: {best_val_f1:.2f}%)")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  >> Early stopping at epoch {epoch}")
                break

    save_training_plots(
        train_losses, val_losses,
        train_accs, val_accs,
        train_f1s, val_f1s,
        model_name
    )

    return best_val_f1