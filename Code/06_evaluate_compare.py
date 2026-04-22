"""
06_evaluate_compare.py
----------------------
Loads all four trained models, evaluates them on the test set,
and produces a side-by-side comparison bar chart.
"""

import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from config import (
    PROCESSED_DATA_DIR, IMG_SIZE, NUM_CLASSES,
    BATCH_SIZE, DEVICE, MODEL_PATHS, PLOTS_DIR, RESULTS_DIR, DROPOUT_RATE
)
from utils import evaluate_model
from train_custom_cnn import CustomCNN


def load_all_models():
    custom_cnn = CustomCNN(NUM_CLASSES).to(DEVICE)
    custom_cnn.load_state_dict(
        torch.load(MODEL_PATHS["custom_cnn"], map_location=DEVICE)
    )

    vgg16 = models.vgg16(weights=None)
    vgg16.classifier[6] = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(4096, 256), nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(256, NUM_CLASSES),
    )
    vgg16.load_state_dict(torch.load(MODEL_PATHS["vgg16"], map_location=DEVICE))
    vgg16 = vgg16.to(DEVICE)

    resnet50 = models.resnet50(weights=None)
    resnet50.fc = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(2048, 512), nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(512, NUM_CLASSES),
    )
    resnet50.load_state_dict(torch.load(MODEL_PATHS["resnet50"], map_location=DEVICE))
    resnet50 = resnet50.to(DEVICE)

    efficientnet = models.efficientnet_b3(weights=None)
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(1536, 512), nn.SiLU(inplace=True),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(512, NUM_CLASSES),
    )
    efficientnet.load_state_dict(
        torch.load(MODEL_PATHS["efficientnet"], map_location=DEVICE)
    )
    efficientnet = efficientnet.to(DEVICE)

    return {
        "Custom CNN":    custom_cnn,
        "VGG-16":        vgg16,
        "ResNet-50":     resnet50,
        "EfficientNet-B3": efficientnet,
    }


def plot_comparison(results):
    model_names = [r["model"] for r in results]
    metrics = ["accuracy", "f1", "precision", "recall"]
    colors  = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    x = range(len(model_names))
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=False)

    for ax, metric, color in zip(axes, metrics, colors):
        values = [r[metric] for r in results]
        bars = ax.bar(x, values, color=color, edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle("Facial Pain Detection — Model Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(PROCESSED_DATA_DIR, "test"), val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    models_dict = load_all_models()
    results = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, test_loader, DEVICE, name.replace(" ", "_"))
        results.append(metrics)

    plot_comparison(results)

    summary_path = os.path.join(RESULTS_DIR, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {summary_path}")

    print("\n=== FINAL COMPARISON TABLE ===")
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(f"{r['model']:<20} {r['accuracy']:>9.2f}% {r['f1']:>9.2f}% "
              f"{r['precision']:>9.2f}% {r['recall']:>9.2f}%")


if __name__ == "__main__":
    main()
