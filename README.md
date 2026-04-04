# Facial Pain Detection — Final Project

**Course:** DATS 6303 – Deep Learning
**University:** The George Washington University
**Instructor:** Dr. Amir Jafari

## Problem Statement

Accurate and real-time pain assessment is critical in clinical settings, especially for patients who cannot verbally communicate — such as post-surgery patients, infants, individuals with dementia, or those with communication disorders. This project develops a deep learning system that detects and classifies pain intensity levels (No Pain, Mild, Moderate, Severe) from facial images using computer vision and transfer learning, with Grad-CAM explainability.

## Repository Structure

```
Final-Project-GroupX/
├── Code/                              # All source code
│   ├── config.py                      # Central configuration
│   ├── utils.py                       # Training/evaluation utilities
│   ├── 01_data_preprocessing.py       # Data loading, face crop, split
│   ├── 02_train_custom_cnn.py         # Baseline CNN (from scratch)
│   ├── 03_train_vgg16.py              # VGG-16 transfer learning
│   ├── 04_train_resnet50.py           # ResNet-50 transfer learning
│   ├── 05_train_efficientnet.py       # EfficientNet-B3 transfer learning
│   ├── 06_evaluate_compare.py         # Multi-model comparison
│   ├── 07_gradcam.py                  # Grad-CAM explainability
│   ├── app.py                         # Streamlit webcam demo app
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # Code execution instructions
├── Group-Proposal/
│   └── Group-Proposal.pdf
├── Final-Group-Project-Report/
│   └── Final-Report.pdf
├── Final-Group-Presentation/
│   └── Presentation.pdf
└── README.md                          # This file
```

## Deep Learning Approach

### Networks Compared
| Model | Type | Parameters |
|-------|------|-----------|
| Custom CNN | From scratch | ~2M |
| VGG-16 | Transfer learning (ImageNet) | ~138M |
| ResNet-50 | Transfer learning (ImageNet) | ~25M |
| EfficientNet-B3 | Transfer learning (ImageNet) | ~12M |

### Course Topics Covered
- **Computer Vision** — pretrained networks (VGG, ResNet, EfficientNet)
- **Interpretability & Explainability** — Grad-CAM (Class Activation Maps)

### Dataset
- **Primary:** UNBC-McMaster Shoulder Pain Expression Archive
- **Fallback:** CK+ dataset (Kaggle) with pain-proxy labeling
- https://www.kaggle.com/datasets/dollyprajapati182/balanced-image-fer-dataset-7575-rgb/data

### Pain Classification
| Class | PSPI Score |
|-------|-----------|
| No Pain | 0 |
| Mild | 1–3 |
| Moderate | 4–7 |
| Severe | 8–15 |

## Demo Application

Built with **Streamlit** — enables real-time pain detection via webcam:
- Live face detection and cropping
- Pain level prediction with confidence score
- Grad-CAM heatmap showing which facial regions (brow furrow, eye squeeze, mouth) drove the prediction
- Downloadable JSON report

## Quick Start

```bash
cd Code
pip install -r requirements.txt

# 1. Preprocess data
python 01_data_preprocessing.py

# 2. Train models (run sequentially)
python 02_train_custom_cnn.py
python 03_train_vgg16.py
python 04_train_resnet50.py
python 05_train_efficientnet.py

# 3. Compare all models
python 06_evaluate_compare.py

# 4. Generate Grad-CAM
python 07_gradcam.py --model efficientnet --image sample_face.jpg

# 5. Launch demo app
streamlit run app.py
```

## Evaluation Metrics
- Accuracy, Macro F1-Score, Precision, Recall
- Confusion Matrix per model
- Grad-CAM qualitative interpretability
