import os
import torch

DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

CLASSES = ["mild", "moderate", "no_pain", "severe"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

PAIN_LABELS = {
    0: "Mild Pain",
    1: "Moderate Pain",
    2: "No Pain",
    3: "Severe Pain"
}

PAIN_COLORS = {
    0: "#f1c40f",
    1: "#e67e22",
    2: "#2ecc71",
    3: "#e74c3c"
}

PSPI_THRESHOLDS = {
    "no_pain":  (0, 0),
    "mild":     (1, 3),
    "moderate": (4, 7),
    "severe":   (8, 15)
}

IMG_SIZE = 224
CHANNELS = 3

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.5

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "custom_cnn":    os.path.join(MODELS_DIR, "custom_cnn_best.pth"),
    "vgg16":         os.path.join(MODELS_DIR, "vgg16_best.pth"),
    "resnet50":      os.path.join(MODELS_DIR, "resnet50_best.pth"),
    "efficientnet":  os.path.join(MODELS_DIR, "efficientnet_best.pth"),
    # Mouth-attention fine-tuned variants (09_finetune_mouth_attention.py)
    "vgg16_mouth":   os.path.join(MODELS_DIR, "vgg16_mouth_attention_best.pth"),
    "resnet50_mouth": os.path.join(MODELS_DIR, "resnet50_mouth_attention_best.pth"),
    # Dual-input CNN+MLP (10_train_dual_input.py)
    "dual_input":    os.path.join(MODELS_DIR, "dual_input_best.pth"),
}

CLASS_NAMES = CLASSES
