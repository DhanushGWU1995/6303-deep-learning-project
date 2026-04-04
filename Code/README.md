# Code — Facial Pain Detection

## Order of Execution

Run the scripts in this exact order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_data_preprocessing.py` | Face detection, crop, resize, train/val/test split |
| 2 | `02_train_custom_cnn.py` | Train baseline CNN from scratch |
| 3 | `03_train_vgg16.py` | Fine-tune VGG-16 (transfer learning) |
| 4 | `04_train_resnet50.py` | Fine-tune ResNet-50 (transfer learning) |
| 5 | `05_train_efficientnet.py` | Fine-tune EfficientNet-B3 (transfer learning) |
| 6 | `06_evaluate_compare.py` | Compare all models, plot metrics |
| 7 | `07_gradcam.py` | Generate Grad-CAM heatmaps |
| 8 | `app.py` | Launch Streamlit webcam demo |

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Setup

### Option A — Pre-organised (easiest)
Organize your dataset as:
```
data/raw/
  no_pain/    ← PSPI score = 0
  mild/       ← PSPI score 1-3
  moderate/   ← PSPI score 4-7
  severe/     ← PSPI score 8+
```

### Option B — UNBC-McMaster
After requesting access from https://www.pitt.edu/~emotion/um-spread.htm:
```
data/raw/
  Subject001/
    Subject001_frame001.png
    ...
  pspi_scores.csv    ← columns: subject, frame, pspi
```

### Option C — CK+ (Kaggle, quick start)
Download from Kaggle, map emotion labels to pain levels:
- Angry, Disgust, Fear → mild/moderate pain proxy
- Neutral, Happy → no_pain

## Running the Demo App

```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.
Select a model from the sidebar, enable webcam or upload an image, and view the pain prediction with Grad-CAM overlay.

## Running Grad-CAM on a specific image

```bash
python 07_gradcam.py --model efficientnet --image path/to/face.jpg
```

## Output Files

| Location | Contents |
|----------|----------|
| `models/` | Saved `.pth` model weights |
| `plots/` | Training curves, confusion matrices, Grad-CAM images, comparison chart |
| `results/` | Per-model metrics text files + `comparison_summary.json` |
