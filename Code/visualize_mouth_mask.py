"""
visualize_mouth_mask.py
───────────────────────
Preview the mouth-emphasis spatial mask on a sample image before committing to
a full preprocessing rerun.  Saves a side-by-side PNG so you can verify the
lower-face amplification looks correct.

Usage:
    python3 visualize_mouth_mask.py --image path/to/face.jpg
    python3 visualize_mouth_mask.py          # auto-picks first test image
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "preprocessing",
    pathlib.Path(__file__).parent / "01_data_preprocessing.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
apply_mouth_emphasis = _mod.apply_mouth_emphasis


def _auto_find_image() -> str | None:
    patterns = [
        str(PROJECT_ROOT / "data" / "raw" / "test" / "*" / "*.jpg"),
        str(PROJECT_ROOT / "data" / "raw" / "test" / "*" / "*.png"),
        str(PROJECT_ROOT / "data" / "raw" / "train" / "*" / "*.jpg"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def _load_rgb(path: str, size: int = 224) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((size, size)))


def show_mask_effect(image_path: str, out_path: str = "mouth_mask_preview.png"):
    original = _load_rgb(image_path)
    emphasised = apply_mouth_emphasis(original.copy())

    # Compute the per-pixel weight map for display
    h = original.shape[0]
    rows = np.arange(h, dtype=np.float32)
    midpoint = 0.50 * h
    steepness = 10.0 / h
    sig = 1.0 / (1.0 + np.exp(-steepness * (rows - midpoint)))
    weights = 0.80 + (1.35 - 0.80) * sig          # (H,)
    weight_map = np.tile(weights[:, None], (1, original.shape[1]))  # (H,W)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Mouth-Region Emphasis Mask Preview", fontsize=14, fontweight="bold")

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(weight_map, cmap="RdYlGn", vmin=0.75, vmax=1.40)
    axes[1].set_title("Spatial Weight Map\n(green=boosted, red=dampened)", fontsize=11)
    axes[1].set_xlabel("← upper face (0.80×)     lower face (1.35×) →",
                        fontsize=9, labelpad=6)
    axes[1].tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(emphasised)
    axes[2].set_title("After Mouth Emphasis", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None,
                   help="Path to a face image.  Auto-picks from data/ if omitted.")
    p.add_argument("--out", default="mouth_mask_preview.png")
    args = p.parse_args()

    image_path = args.image or _auto_find_image()
    if image_path is None:
        print("[ERROR] No image found.  Pass --image <path>.")
        sys.exit(1)

    print(f"[INFO] Using image: {image_path}")
    show_mask_effect(image_path, args.out)


if __name__ == "__main__":
    main()
