# Deployment & Run Instructions
## Real-Time Facial Pain Detection — DATS 6303 Final Project

---

## Prerequisites

- EC2 instance with NVIDIA GPU (tested on A10G)
- **Kernel:** must be `6.8.0-1031-aws` (not 6.17 — incompatible with NVIDIA driver 560)
- Python 3.10+
- CUDA 12.1 + PyTorch with GPU support

### Verify kernel and GPU

```bash
uname -r          # must show 6.8.0-1031-aws
nvidia-smi        # must show GPU and driver 560
python3 -c "import torch; print(torch.cuda.is_available())"  # must print True
```

---

## Step 0 — Clone & Install

```bash
git clone https://github.com/DhanushGWU1995/Facial-pain-detection-deep-learning.git
cd Facial-pain-detection-deep-learning

pip install -r Code/requirements.txt
pip install huggingface_hub      # required for HF upload step only
```

---

## Step 1 — Data Preprocessing

Downloads the AffectNet Relabeled Balanced dataset (`viktormodroczky/facial-affect-data-relabeled` from Hugging Face),
detects and crops faces, applies the **mouth-region emphasis mask** (lower face boosted 1.35×, upper face dampened 0.80×),
and maps emotion labels to pain severity classes.

### Data Flow

```
Code/data/raw/train/          ← raw download: 8 emotion class folders
  anger/ contempt/ disgust/ fear/ happy/ neutral/ sad/ surprise/
          ↓  01_data_preprocessing.py  ↓
Code/data/processed/train/    ← pain-mapped output used by ALL training scripts
  no_pain/ mild/ moderate/ severe/
```

> `data/raw/` is read **only** by the preprocessing script.
> Every training script (02 through 10) reads from `data/processed/`.

### Run from inside the `Code/` directory

```bash
cd Facial-pain-detection-deep-learning/Code
python3 01_data_preprocessing.py
```

**Expected output:**
- Raw images saved to `data/raw/train/` and `data/raw/test/` (emotion folders)
- Processed images (pain classes) saved to `data/processed/train/` and `data/processed/test/`
- ~36,500 images | 4 class folders: `no_pain`, `mild`, `moderate`, `severe`
- Estimated time: ~10 minutes

**Optional — preview the mouth mask on a sample image:**

```bash
python3 visualize_mouth_mask.py --out /tmp/mouth_preview.png
```

### Pain Class Mapping

| Emotion | Pain Class | Clinical Basis |
|---------|-----------|----------------|
| neutral, happy | No Pain | Relaxed facial musculature |
| sad, contempt | Mild | Subtle brow tension (AU4 partial) |
| fear | Moderate | AU4+20 visible tension |
| angry, disgust, surprise | Severe | High-intensity AU grimace pattern |

---

## Step 2 — Train All Models

> **All training scripts must be run from inside `Code/`.** They detect their own location and save weights to `Code/models/`, which is the same folder for every script.

Use `tmux` to keep sessions alive if SSH drops:

```bash
tmux new -s training
```

### Baseline Custom CNN

```bash
python3 02_train_custom_cnn.py
```

### VGG-16 (Transfer Learning — ImageNet)

```bash
python3 03_train_vgg16.py
```

### ResNet-50 (Transfer Learning — ImageNet)

```bash
python3 04_train_resnet50.py
```

### EfficientNet-B3 (Transfer Learning — ImageNet)

```bash
python3 05_train_efficientnet.py
```

### Dual-Input CNN+MLP (Full Face + Mouth Crop) — NEW

Two-branch architecture: ResNet-50 (full face, 2048-d) + ResNet-18 (mouth crop, 512-d)
fused through an MLP head. Backbone frozen for first 15 epochs, then jointly fine-tuned.

```bash
python3 10_train_dual_input.py --epochs 30 --unfreeze-after 15
```

### VGG-16 + CBAM Mouth-Attention Fine-tune — NEW

Adds a Convolutional Block Attention Module (CBAM) with a lower-face spatial prior on top
of the pre-trained VGG-16 backbone.

```bash
python3 09_finetune_mouth_attention.py --model vgg16 --epochs 20
```

### ResNet-50 + CBAM Mouth-Attention Fine-tune — NEW

```bash
python3 09_finetune_mouth_attention.py --model resnet50 --epochs 20
```

### Model Weight Output (all land in `Code/models/`)

| Script | Output weight file |
|--------|--------------------|
| 02 | `Code/models/custom_cnn_best.pth` |
| 03 | `Code/models/vgg16_best.pth` |
| 04 | `Code/models/resnet50_best.pth` |
| 05 | `Code/models/efficientnet_best.pth` |
| 10 | `Code/models/dual_input_best.pth` |
| 09 (vgg16) | `Code/models/vgg16_mouth_attention_best.pth` |
| 09 (resnet50) | `Code/models/resnet50_mouth_attention_best.pth` |

> **Important:** Scripts 09 and 10 were updated (commit: fix path bug) to resolve to `Code/models/` — the same directory as scripts 02–05. If you have an older clone, run `git pull origin main` before training.

---

## Step 3 — Evaluate All Models

Generates confusion matrices, per-class F1 scores, and a side-by-side comparison table
for all trained models. Saves plots to `plots/` and metrics to `results/`.

```bash
python3 06_evaluate_compare.py
```

---

## Step 4 — Grad-CAM Visualizations

Generates Grad-CAM heatmaps showing which facial regions each model attends to.

```bash
# Standard models
python3 07_gradcam.py --model vgg16
python3 07_gradcam.py --model resnet50

# Mouth-focused models (run after Step 2 fine-tuning completes)
python3 07_gradcam.py --model vgg16_mouth
python3 07_gradcam.py --model resnet50_mouth

# Dual-input model (shows two branch heatmaps: full face + mouth crop)
python3 07_gradcam.py --model dual_input
```

---

## Step 5 — Upload to Hugging Face Hub

Uploads all model weights to the HF model repo and deploys the Streamlit app to HF Spaces.

```bash
python3 08_upload_to_hf.py --token YOUR_HF_TOKEN
```

- **Model repo:** `DhanushGWU1995/facial-pain-detection-models`
- **HF Space:** `DhanushGWU1995/facial-pain-detection`

---

## Step 6 — Run the Streamlit Demo App

```bash
cd Facial-pain-detection-deep-learning/Code
streamlit run app.py --server.port 8501
```

Open `http://<EC2-PUBLIC-IP>:8501` in your browser.

The app supports:
- **Webcam** live capture
- **Image upload**
- **6 model options** in the sidebar (Dual-Input CNN+MLP is the default)
- **Grad-CAM heatmaps** — dual-input model shows two branch heatmaps side-by-side (full face + mouth crop)

---

## Model Architecture Summary

| Model | Architecture | Params | Val Accuracy |
|-------|-------------|--------|-------------|
| Custom CNN | From scratch | ~2M | ~39% |
| VGG-16 | ImageNet transfer | ~138M | **79.65%** |
| ResNet-50 | ImageNet transfer | ~25M | 79.19% |
| EfficientNet-B3 | ImageNet transfer | ~12M | 78.72% |
| VGG-16 + Mouth Attention | CBAM fine-tune | ~138M | TBD |
| ResNet-50 + Mouth Attention | CBAM fine-tune | ~25M | TBD |
| **Dual-Input CNN+MLP** | ResNet-50 + ResNet-18 + MLP | ~36M | TBD |

---

## Estimated GPU Time (NVIDIA A10G)

| Step | Estimated Time |
|------|---------------|
| Preprocessing | ~10 min |
| Custom CNN | ~20 min |
| VGG-16 / ResNet-50 / EfficientNet | ~45 min each |
| Dual-Input (30 epochs) | ~50 min |
| CBAM fine-tunes ×2 (20 epochs each) | ~25 min each |
| **Total** | **~5 hours** |

---

## tmux Cheatsheet

```bash
tmux new -s training          # start new session
# Ctrl+B then D               # detach (session keeps running)
tmux attach -t training       # reconnect to session
tmux ls                       # list all sessions
```

---

## Dataset

**AffectNet Relabeled Balanced**
- Source: `viktormodroczky/facial-affect-data-relabeled` (Hugging Face)
- Format: Genuine RGB color photographs, 96×96 px
- Size: ~36,500 images across 4 pain classes (balanced)
