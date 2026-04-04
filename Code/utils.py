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


def save_training_plots(train_losses, val_losses, train_accs, val_accs, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(val_losses, label="Val Loss", color="tomato")
    axes[0].set_title(f"{model_name} — Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accs, label="Train Acc", color="steelblue")
    axes[1].plot(val_accs, label="Val Acc", color="tomato")
    axes[1].set_title(f"{model_name} — Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


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

    return {"model": model_name, "accuracy": acc, "f1": f1,
            "precision": prec, "recall": rec}


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
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def run_training_loop(model, train_loader, val_loader, criterion, optimizer,
                      scheduler, device, model_name, save_path, num_epochs,
                      patience):
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        print(f"Epoch [{epoch:3d}/{num_epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.2f}%")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"  >> Best model saved (val acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  >> Early stopping at epoch {epoch}")
                break

    save_training_plots(train_losses, val_losses, train_accs, val_accs, model_name)
    return best_val_acc
