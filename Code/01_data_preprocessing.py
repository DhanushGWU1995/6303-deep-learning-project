"""
01_data_preprocessing.py
------------------------
Loads raw facial images, detects and crops faces, resizes, augments,
and organizes them into train / val / test splits.

Supported dataset layouts
--------------------------
Option A — UNBC-McMaster (after requesting access):
  data/raw/
    Subject001/
      Subject001_frame001.png
      ...
    PSPI scores provided in a CSV (subject, frame, pspi_score)

Option B — Pre-organised pain folder (CK+, custom):
  data/raw/
    no_pain/
    mild/
    moderate/
    severe/

Option C — FER2013 / Balanced RGB FER (Kaggle: dollyprajapati182/balanced-image-fer-dataset-7575-rgb
                                          OR:  sayakbera/fer-2013-7-emotions-uniform-dataset):
  data/raw/
    train/
      angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
    test/
      angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
    validation/   (optional, pooled with the above)

  Emotion → Pain mapping applied automatically:
    neutral, happy  → no_pain
    sad             → mild
    fear            → moderate
    angry, disgust, surprise → severe

  Preferred: the 75x75 RGB dataset — native 3-channel input works
  directly with ImageNet-pretrained VGG16 / ResNet50 / EfficientNet.

The script auto-detects which layout is present.
"""

import os
import shutil
import random
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    IMG_SIZE, CLASSES, TRAIN_RATIO, VAL_RATIO,
    PSPI_THRESHOLDS
)

random.seed(42)
np.random.seed(42)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Emotion labels → pain severity mapping (case-insensitive)
# Covers: sayakbera uniform dataset, dollyprajapati RGB dataset, and plain FER2013
FER_TO_PAIN = {
    # No Pain — relaxed / positive expressions
    "neutral":      "no_pain",
    "happy":        "no_pain",
    "happiness":    "no_pain",

    # Mild Pain — subtle distress
    "sad":          "mild",
    "sadness":      "mild",

    # Moderate Pain — visible tension
    "fear":         "moderate",
    "fearful":      "moderate",

    # Severe Pain — intense facial action
    "angry":        "severe",
    "anger":        "severe",
    "disgust":      "severe",
    "disgusted":    "severe",
    "surprise":     "severe",
    "surprised":    "severe",
}

FER_EMOTION_LABELS = set(FER_TO_PAIN.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pspi_to_class(score: float) -> str:
    score = float(score)
    if score == 0:
        return "no_pain"
    elif score <= 3:
        return "mild"
    elif score <= 7:
        return "moderate"
    else:
        return "severe"


def detect_and_crop_face(image_path: str, target_size: int = IMG_SIZE):
    """
    Read an image (RGB or grayscale), detect the face region, crop and resize.
    Optimised for the 75x75 RGB dataset; gracefully handles grayscale too.
    Always returns a 3-channel (RGB) PIL image at target_size x target_size.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)   # always load as BGR
    if img is None:
        return None

    # Ensure 3-channel BGR regardless of source format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Face detection on grayscale version
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    if len(faces) == 0:
        # No face detected — use the full image (common in 75x75 crops)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        x, y, w, h = faces[0]
        face = img[y:y + h, x:x + w]
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb).resize((target_size, target_size))


# ---------------------------------------------------------------------------
# Layout detectors
# ---------------------------------------------------------------------------

def _is_fer2013_layout(raw_dir: str) -> bool:
    """True if raw_dir contains train/ and test/ subfolders with emotion names."""
    for split in ("train", "test"):
        split_path = os.path.join(raw_dir, split)
        if not os.path.isdir(split_path):
            return False
        subdirs = {d.lower() for d in os.listdir(split_path)
                   if os.path.isdir(os.path.join(split_path, d))}
        if not subdirs.intersection(FER_EMOTION_LABELS):
            return False
    return True


def _is_pain_organised(raw_dir: str) -> bool:
    """True if raw_dir contains at least one pain-class subfolder."""
    return any(
        os.path.isdir(os.path.join(raw_dir, cls)) for cls in CLASSES
    )


# ---------------------------------------------------------------------------
# Sample collectors
# ---------------------------------------------------------------------------

def collect_samples_from_fer2013(raw_dir: str):
    """
    Walk FER2013 train/ test/ validation/ splits, map emotion labels to pain
    classes, then return a flat list of (image_path, pain_class) tuples.
    All splits are pooled and re-split by this script for consistent ratios.
    Emotion matching is case-insensitive to handle 'Anger', 'Happiness', etc.
    """
    samples = []
    for fer_split in ("train", "test", "validation"):
        split_dir = os.path.join(raw_dir, fer_split)
        if not os.path.isdir(split_dir):
            continue
        for emotion_dir in os.listdir(split_dir):
            emotion = emotion_dir.lower().strip()
            pain_class = FER_TO_PAIN.get(emotion)
            if pain_class is None:
                print(f"  [Skip] Unknown emotion folder: {emotion_dir}")
                continue
            full_dir = os.path.join(split_dir, emotion_dir)
            for fname in os.listdir(full_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    samples.append((os.path.join(full_dir, fname), pain_class))
    return samples


def collect_samples_from_organised(raw_dir: str):
    """Collect from pre-organised pain-class subfolders."""
    samples = []
    for cls in CLASSES:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(cls_dir, fname), cls))
    return samples


def collect_samples_from_unbc(raw_dir: str, pspi_csv: str):
    """Collect from UNBC-McMaster subject folders using a PSPI CSV."""
    samples = []
    pspi_map = {}
    with open(pspi_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pspi_map[(row["subject"], row["frame"])] = float(row["pspi"])

    for subject in os.listdir(raw_dir):
        subj_dir = os.path.join(raw_dir, subject)
        if not os.path.isdir(subj_dir):
            continue
        for fname in os.listdir(subj_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            frame_id = os.path.splitext(fname)[0]
            key = (subject, frame_id)
            if key in pspi_map:
                cls = pspi_to_class(pspi_map[key])
                samples.append((os.path.join(subj_dir, fname), cls))
    return samples


# ---------------------------------------------------------------------------
# Split and save
# ---------------------------------------------------------------------------

def split_samples(samples):
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return (samples[:n_train],
            samples[n_train:n_train + n_val],
            samples[n_train + n_val:])


def save_split(split_samples, split_name: str):
    saved, skipped = 0, 0
    for img_path, cls in split_samples:
        dest_dir = os.path.join(PROCESSED_DATA_DIR, split_name, cls)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        pil_img = detect_and_crop_face(img_path)
        if pil_img is not None:
            pil_img.save(dest_path)
            saved += 1
        else:
            skipped += 1
    print(f"  Saved: {saved} | Skipped (unreadable): {skipped}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pspi_csv = os.path.join(RAW_DATA_DIR, "pspi_scores.csv")

    if _is_fer2013_layout(RAW_DATA_DIR):
        print("[Info] Detected FER2013 layout (train/test + emotion subfolders).")
        print("[Info] Applying emotion → pain mapping:")
        for emotion, pain in FER_TO_PAIN.items():
            print(f"         {emotion:12s} → {pain}")
        samples = collect_samples_from_fer2013(RAW_DATA_DIR)

    elif _is_pain_organised(RAW_DATA_DIR):
        print("[Info] Detected pre-organised pain folder structure.")
        samples = collect_samples_from_organised(RAW_DATA_DIR)

    elif os.path.exists(pspi_csv):
        print("[Info] Detected UNBC-McMaster structure with PSPI CSV.")
        samples = collect_samples_from_unbc(RAW_DATA_DIR, pspi_csv)

    else:
        raise FileNotFoundError(
            "\n[Error] Cannot find dataset. Expected one of:\n"
            "  (A) data/raw/train/<emotion>/ and data/raw/test/<emotion>/  ← FER2013\n"
            "  (B) data/raw/<no_pain|mild|moderate|severe>/                ← Pre-organised\n"
            "  (C) data/raw/pspi_scores.csv + subject folders              ← UNBC-McMaster\n"
        )

    print(f"\n[Info] Total samples collected: {len(samples)}")
    print("[Info] Class distribution:")
    for cls in CLASSES:
        count = sum(1 for _, c in samples if c == cls)
        bar = "█" * (count // 100)
        print(f"  {cls:12s}: {count:5d}  {bar}")

    train_s, val_s, test_s = split_samples(samples)
    print(f"\n[Info] Split → Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)

    for split_name, split in [("train", train_s), ("val", val_s), ("test", test_s)]:
        print(f"[Info] Processing '{split_name}' split…")
        save_split(split, split_name)

    print("\n[Done] Preprocessing complete. Data saved to:", PROCESSED_DATA_DIR)
    print("\n[Info] Final processed distribution:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}/")
        for cls in CLASSES:
            path = os.path.join(PROCESSED_DATA_DIR, split_name, cls)
            count = len(os.listdir(path)) if os.path.isdir(path) else 0
            print(f"    {cls:12s}: {count}")


if __name__ == "__main__":
    main()
