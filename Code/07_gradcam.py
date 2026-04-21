"""
07_gradcam.py
-------------
CLI wrapper for Grad-CAM visualization.
Core implementation lives in gradcam.py (importable by app.py).

Usage:
  python 07_gradcam.py --model efficientnet --image path/to/face.jpg
  python 07_gradcam.py --model resnet50     --image path/to/face.jpg --save out.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from torchvision import transforms
from PIL import Image

from config import IMG_SIZE, DEVICE, MODEL_PATHS, PLOTS_DIR, PAIN_LABELS
from gradcam import GradCAM, get_target_layer, load_model_for_gradcam, overlay_heatmap

os.makedirs(PLOTS_DIR, exist_ok=True)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_gradcam(model_name: str, image_path: str, save_path: str = None):
    pil_img   = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    original  = np.array(pil_img)
    tensor    = PREPROCESS(pil_img)

    model        = load_model_for_gradcam(model_name, MODEL_PATHS[model_name])
    target_layer = get_target_layer(model_name, model)
    gradcam      = GradCAM(model, target_layer)

    cam, pred_class, confidence = gradcam.generate(tensor)
    overlay = overlay_heatmap(original, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original);         axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet");  axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(overlay);          axes[2].set_title(
        f"Overlay — {PAIN_LABELS[pred_class]} ({confidence * 100:.1f}%)"
    )
    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"Grad-CAM: {model_name.upper()} | Predicted: {PAIN_LABELS[pred_class]}",
        fontsize=13
    )
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(
            PLOTS_DIR, f"gradcam_{model_name}_{os.path.basename(image_path)}"
        )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")
    print(f"  Prediction : {PAIN_LABELS[pred_class]}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    return cam, pred_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM for facial pain detection")
    parser.add_argument("--model", default="efficientnet",
                        choices=["custom_cnn", "vgg16", "resnet50", "efficientnet"])
    parser.add_argument("--image", required=True, help="Path to input face image")
    parser.add_argument("--save",  default=None,  help="Path to save output image")
    args = parser.parse_args()
    run_gradcam(args.model, args.image, args.save)
