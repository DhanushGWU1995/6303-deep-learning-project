"""
Shared face preprocessing helpers used by dataset prep and inference.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from config import IMG_SIZE

FRONTAL_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
PROFILE_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)


def apply_mouth_emphasis(rgb_array: np.ndarray, mouth_boost: float = 1.35,
                         upper_dampen: float = 0.80,
                         mouth_start_frac: float = 0.50) -> np.ndarray:
    """
    Apply a smooth lower-face brightness prior.

    This is disabled by default in the project because it is not part of the
    paper's preprocessing pipeline and can teach the model to rely on an
    artificial cue rather than genuine pain-related facial action units.
    """
    h, _ = rgb_array.shape[:2]
    rows = np.arange(h, dtype=np.float32)
    midpoint = mouth_start_frac * h
    steepness = 10.0 / max(h, 1)
    sigmoid = 1.0 / (1.0 + np.exp(-steepness * (rows - midpoint)))
    row_weights = upper_dampen + (mouth_boost - upper_dampen) * sigmoid
    emphasised = np.clip(
        rgb_array.astype(np.float32) * row_weights[:, np.newaxis, np.newaxis],
        0,
        255,
    )
    return emphasised.astype(np.uint8)


def _largest_face(faces) -> tuple[int, int, int, int] | None:
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return int(x), int(y), int(w), int(h)


def detect_face_bbox_from_bgr(bgr_img: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Detect the largest face in an image.

    The paper reports using both frontal and side Haar cascades, so we apply
    the frontal detector first and then fall back to the profile cascade on the
    original and mirrored image.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    frontal = FRONTAL_FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    bbox = _largest_face(frontal)
    if bbox is not None:
        return bbox

    profile = PROFILE_FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    bbox = _largest_face(profile)
    if bbox is not None:
        return bbox

    flipped = cv2.flip(gray, 1)
    profile_flipped = PROFILE_FACE_CASCADE.detectMultiScale(
        flipped, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    bbox = _largest_face(profile_flipped)
    if bbox is None:
        return None

    x, y, w, h = bbox
    return gray.shape[1] - x - w, y, w, h


def preprocess_face_pil(pil_img: Image.Image, target_size: int = IMG_SIZE,
                        mouth_emphasis: bool = False,
                        return_bbox: bool = False):
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bbox = detect_face_bbox_from_bgr(bgr)

    if bbox is None:
        face_rgb = rgb
    else:
        x, y, w, h = bbox
        face_rgb = rgb[y:y + h, x:x + w]

    face_pil = Image.fromarray(face_rgb)
    if target_size is not None:
        face_pil = face_pil.resize((target_size, target_size))

    if mouth_emphasis:
        face_pil = Image.fromarray(apply_mouth_emphasis(np.array(face_pil)))

    if return_bbox:
        return face_pil, bbox
    return face_pil


def preprocess_face_image_path(image_path: str, target_size: int = IMG_SIZE,
                               mouth_emphasis: bool = False):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_face_pil(
        Image.fromarray(rgb),
        target_size=target_size,
        mouth_emphasis=mouth_emphasis,
    )
