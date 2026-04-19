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

### Recommended — AffectNet Relabeled Balanced (Genuine RGB color images)

Download from Kaggle:
```
https://www.kaggle.com/datasets/viktormodroczky/facial-affect-data-relabeled
```

After extracting, move the folders so your structure looks like this:
```
data/raw/
  train/
    anger/   contempt/   disgust/   fear/
    happy/   neutral/    sad/       surprise/
  test/
    anger/   contempt/   disgust/   fear/
    happy/   neutral/    sad/       surprise/
```

Quick setup commands (run from `Code/`):
```bash
kaggle datasets download -d viktormodroczky/facial-affect-data-relabeled
unzip facial-affect-data-relabeled.zip
mv data_relabeled_balanced_1x/train data/raw/train
mv data_relabeled_balanced_1x/test  data/raw/test
```

Use `data_relabeled_balanced_1x` for standard size, `2x` or `3x` for larger training sets.

**Emotion → Pain mapping applied automatically:**
| Emotion | Pain Level |
|---|---|
| neutral, happy | No Pain |
| sad, contempt | Mild |
| fear | Moderate |
| angry, disgust, surprise | Severe |

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
