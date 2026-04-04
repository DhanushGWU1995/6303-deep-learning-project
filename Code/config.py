import os
import torch

DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

CLASSES = ["no_pain", "mild", "moderate", "severe"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

PAIN_LABELS = {
    0: "No Pain",
    1: "Mild Pain",
    2: "Moderate Pain",
    3: "Severe Pain"
}

PAIN_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
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
    "custom_cnn":  os.path.join(MODELS_DIR, "custom_cnn_best.pth"),
    "vgg16":       os.path.join(MODELS_DIR, "vgg16_best.pth"),
    "resnet50":    os.path.join(MODELS_DIR, "resnet50_best.pth"),
    "efficientnet": os.path.join(MODELS_DIR, "efficientnet_best.pth"),
}
