# Facial Pain Detection — Final Project

**Course:** DATS 6303 – Deep Learning
**University:** The George Washington University
**Instructor:** Dr. Amir Jafari

## Problem Statement

Accurate and real-time pain assessment is critical in clinical settings, especially for patients who cannot verbally communicate — such as post-surgery patients, infants, individuals with dementia, or those with communication disorders. This project develops a deep learning system that detects and classifies pain intensity levels (No Pain, Mild, Moderate, Severe) from facial images using computer vision and transfer learning, with Grad-CAM explainability.

## Repository Structure

```
Facial-pain-detection-deep-learning/
├── Code/                              # All source code
│   ├── config.py                      # Central configuration (paths, hyperparams)
│   ├── utils.py                       # Training/evaluation utilities
│   ├── gradcam.py                     # Grad-CAM + DualInputGradCAM classes
│   ├── 01_data_preprocessing.py       # Download, face-crop, emotion→pain mapping
│   ├── 02_train_custom_cnn.py         # Baseline CNN (from scratch)
│   ├── 03_train_vgg16.py              # VGG-16 transfer learning
│   ├── 04_train_resnet50.py           # ResNet-50 transfer learning
│   ├── 05_train_efficientnet.py       # EfficientNet-B3 transfer learning
│   ├── 06_evaluate_compare.py         # Multi-model comparison
│   ├── 07_gradcam.py                  # Grad-CAM visualizations (CLI)
│   ├── 08_upload_to_hf.py             # Upload weights + app to Hugging Face
│   ├── 09_finetune_mouth_attention.py # VGG-16 / ResNet-50 + CBAM mouth attention
│   ├── 10_train_dual_input.py         # Dual-Input CNN+MLP (full face + mouth crop)
│   ├── app.py                         # Streamlit webcam demo app
│   ├── requirements.txt               # Python dependencies
│   ├── data/
│   │   ├── raw/                       # Raw download (emotion class folders)
│   │   └── processed/                 # Pain-mapped output (used by all trainers)
│   └── models/                        # Saved .pth weight files (all scripts)
├── Group-Proposal/
│   └── Group-Proposal.pdf
├── Final-Group-Project-Report/
│   └── Final-Report.pdf
├── Final-Group-Presentation/
│   └── Presentation.pdf
├── DEPLOYMENT.md                      # Full EC2 run guide
└── README.md                          # This file
```

## Data Flow

The dataset is the [AffectNet Relabeled Balanced](https://www.kaggle.com/datasets/viktormodroczky/facial-affect-data-relabeled) collection (Hugging Face). Preprocessing converts emotion labels into pain severity classes:

```
Code/data/raw/train/          ← raw download (emotion folders)
  anger/ contempt/ disgust/ fear/ happy/ neutral/ sad/ surprise/
          ↓  01_data_preprocessing.py  ↓
Code/data/processed/train/    ← used by ALL training scripts (02 – 10)
  no_pain/ mild/ moderate/ severe/
```

> **Note:** `data/raw/` is only ever read by the preprocessing script.
> Every training script reads from `data/processed/` (via `PROCESSED_DATA_DIR` in `config.py`).

### Pain Classification via Emotion Mapping

| Emotion | → Pain Class | Clinical Basis |
|---|---|---|
| Neutral, Happy | No Pain | Relaxed facial musculature |
| Sad, Contempt | Mild | Subtle brow tension, AU4 partial activation |
| Fear | Moderate | AU4+20 visible tension |
| Angry, Disgust, Surprise | Severe | High-intensity AU grimace pattern |

## Deep Learning Approach

### Networks Compared

| Model | Type | Parameters |
|-------|------|-----------|
| Custom CNN | From scratch | ~2M |
| VGG-16 | Transfer learning (ImageNet) | ~138M |
| ResNet-50 | Transfer learning (ImageNet) | ~25M |
| EfficientNet-B3 | Transfer learning (ImageNet) | ~12M |
| VGG-16 + CBAM Mouth Attention | Fine-tuned + attention module | ~138M |
| ResNet-50 + CBAM Mouth Attention | Fine-tuned + attention module | ~25M |
| **Dual-Input CNN+MLP** | ResNet-50 (full face) + ResNet-18 (mouth crop) + MLP fusion | ~36M |

### Mouth-Region Focus

Three complementary strategies are used to bias the network toward the lower face, which carries the strongest pain signal:

1. **Sigmoid spatial mask** (`01_data_preprocessing.py`) — boosts lower face 1.35×, dampens upper face 0.80× before saving processed images.
2. **CBAM attention fine-tuning** (`09_finetune_mouth_attention.py`) — adds a Convolutional Block Attention Module with a lower-face spatial prior on top of VGG-16 / ResNet-50.
3. **Dual-input architecture** (`10_train_dual_input.py`) — explicit second branch: ResNet-18 receives a cropped mouth region (rows 55 – 100 % of face height) and produces a 512-d embedding merged with the full-face 2048-d embedding via an MLP head.

### Course Topics Covered
- **Computer Vision** — pretrained networks (VGG, ResNet, EfficientNet)
- **Interpretability & Explainability** — Grad-CAM and Dual-Branch Grad-CAM (Class Activation Maps)
- **Transfer Learning & Fine-tuning** — frozen backbone phases, differential LR
- **Attention Mechanisms** — CBAM channel + spatial attention

### Dataset
- **Primary:** [AffectNet Relabeled Balanced](https://www.kaggle.com/datasets/viktormodroczky/facial-affect-data-relabeled) — genuine RGB color facial images based on AffectNet (real-world photographs, 96×96)
- **Emotion → Pain mapping** applied via FACS Action Unit overlap (validated in pain research literature)
- **Size:** ~36,500 images balanced across 4 pain classes

## Model Weights (all saved to `Code/models/`)

| Script | Output weight file |
|--------|--------------------|
| 02 | `Code/models/custom_cnn_best.pth` |
| 03 | `Code/models/vgg16_best.pth` |
| 04 | `Code/models/resnet50_best.pth` |
| 05 | `Code/models/efficientnet_best.pth` |
| 09 (vgg16) | `Code/models/vgg16_mouth_attention_best.pth` |
| 09 (resnet50) | `Code/models/resnet50_mouth_attention_best.pth` |
| 10 | `Code/models/dual_input_best.pth` |

## Demo Application

Built with **Streamlit** — enables real-time pain detection via webcam:
- Live face detection and cropping
- Pain level prediction with confidence score
- Grad-CAM heatmap showing which facial regions (brow furrow, eye squeeze, mouth) drove the prediction
- Dual-input model shows two side-by-side branch heatmaps (full face + mouth crop)
- Downloadable JSON report

## Quick Start

```bash
cd Code
pip install -r requirements.txt

# 1. Preprocess — reads data/raw/ (emotion folders), writes data/processed/ (pain folders)
python3 01_data_preprocessing.py

# 2. Train base models (run sequentially, all read from data/processed/)
python3 02_train_custom_cnn.py
python3 03_train_vgg16.py
python3 04_train_resnet50.py
python3 05_train_efficientnet.py

# 3. Train advanced models
python3 10_train_dual_input.py --epochs 30 --unfreeze-after 15
python3 09_finetune_mouth_attention.py --model vgg16 --epochs 20
python3 09_finetune_mouth_attention.py --model resnet50 --epochs 20

# 4. Compare all models
python3 06_evaluate_compare.py

# 5. Generate Grad-CAM visualizations
python3 07_gradcam.py --model vgg16
python3 07_gradcam.py --model dual_input

# 6. Launch demo app
streamlit run app.py
```

> All scripts must be run from inside the `Code/` directory so relative paths resolve correctly.

## Evaluation Metrics
- Accuracy, Macro F1-Score, Precision, Recall
- Confusion Matrix per model
- Grad-CAM qualitative interpretability
