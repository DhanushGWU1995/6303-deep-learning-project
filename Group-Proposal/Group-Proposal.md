# Group Proposal
## DATS 6303 — Deep Learning
## The George Washington University

---

**Project Title:** Real-Time Facial Pain Detection Using Deep Learning and Transfer Learning

**Course Instructor:** Dr. Amir Jafari

**Submission Date:** [Insert Date]

**Group Number:** [Insert Group Number]

**Group Members:** [Insert Names]

---

## 1. Problem Statement and Motivation

Pain assessment is a fundamental challenge in clinical medicine. Accurate measurement of pain is
essential for effective treatment, yet current methods rely heavily on patient self-report — a
method that fails when patients are unable to communicate verbally. This includes post-surgical
patients, infants, elderly individuals with dementia, patients in intensive care units, and those
with neurological or developmental conditions.

The human face is a powerful and natural channel for pain expression. The Facial Action Coding
System (FACS), developed by Ekman and Friesen, describes specific muscle movements — called
Action Units (AUs) — that reliably indicate pain. For example, brow lowering (AU4), cheek raising
(AU6), and nose wrinkling (AU9) are consistently activated during painful experiences.

We propose a deep learning system that automatically detects and classifies pain intensity levels
from facial images using computer vision and convolutional neural networks. This system could be
deployed as an IoT-style camera device in hospital rooms, recovery wards, or eldercare facilities
to provide continuous, non-invasive pain monitoring without requiring patient input.

---

## 2. Dataset

**Primary Dataset: UNBC-McMaster Shoulder Pain Expression Archive**

- **Source:** University of Pittsburgh (access request required at
  https://www.pitt.edu/~emotion/um-spread.htm)
- **Size:** 48,398 FACS-coded frames from 25 subjects (200 video sequences)
- **Labels:** Frame-level PSPI (Prkachin and Solomon Pain Intensity) scores ranging from 0 to 15,
  along with individual Action Unit intensities
- **Pain Classes:** We map PSPI scores to four severity levels:
  - No Pain: PSPI = 0
  - Mild Pain: PSPI = 1–3
  - Moderate Pain: PSPI = 4–7
  - Severe Pain: PSPI = 8–15

**Fallback Dataset: CK+ (Extended Cohn-Kanade)**

- **Source:** Available on Kaggle
- **Size:** 981 image sequences from 123 subjects, 8 emotion categories
- **Usage:** Pain-adjacent emotions (Angry, Disgust, Fear) serve as proxies for pain states during
  initial prototyping

Both datasets are large enough to train deep neural networks, and we will apply data augmentation
(horizontal flipping, rotation, color jitter) to further increase effective training set size.

---

## 3. Deep Learning Network

We will train and compare **four deep learning architectures** to identify the best-performing
model for this task:

**a) Custom CNN (Baseline)**
A convolutional neural network trained from scratch. Four convolutional blocks with batch
normalization, ReLU activations, and max pooling, followed by a fully-connected MLP classifier.
This establishes a performance baseline without transfer learning.

**b) VGG-16 (Transfer Learning)**
The VGG-16 architecture pretrained on ImageNet. We will freeze the convolutional feature extractor
and replace the final classification head with a custom fully-connected layer for 4-class pain
output. A two-phase training strategy (head only → full fine-tune) will be used.

**c) ResNet-50 (Transfer Learning)**
ResNet-50 pretrained on ImageNet. The residual connections make this architecture more efficient
and less prone to vanishing gradients than VGG-16. Same two-phase training approach.

**d) EfficientNet-B3 (Transfer Learning)**
EfficientNet-B3 uses compound scaling to balance depth, width, and resolution. It achieves
superior accuracy with fewer parameters, making it our primary candidate for the final deployed
model.

All pretrained models were originally trained on ImageNet (1.2M images, 1,000 classes), providing
rich visual feature representations that transfer well to facial analysis tasks.

---

## 4. Framework

**Framework: PyTorch (torch, torchvision)**

PyTorch is selected for the following reasons:
- Native support for all four architectures via torchvision.models with pretrained weights
- Dynamic computation graphs that simplify debugging and experimentation
- Clean integration with Grad-CAM via PyTorch hooks
- Strong community support and extensive documentation for transfer learning

**Demo Application: Streamlit**

The final demo application will be built with Streamlit, which provides:
- Built-in `st.camera_input()` component for webcam capture
- Rapid web UI development in pure Python
- Easy deployment and sharing

---

## 5. Reference Materials

1. Zhang, K. et al. (2016). UNBC-McMaster Shoulder Pain Expression Archive Database.
   IEEE FG 2011.
2. He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
3. Simonyan, K. & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale
   Image Recognition. ICLR 2015.
4. Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks. ICML 2019.
5. Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization. ICCV 2017.
6. Prkachin, K.M. & Solomon, P.E. (2008). The structure, reliability and validity of pain
   expression: Evidence from patients with shoulder pain.
7. Papers With Code — Facial Action Unit Detection:
   https://paperswithcode.com/task/facial-action-unit-detection

---

## 6. Performance Metrics

We will evaluate all four models using:

- **Accuracy** — percentage of correctly classified pain levels
- **Macro F1-Score** — weighted average of F1 across all four pain classes
  (important given potential class imbalance)
- **Precision and Recall** — per-class analysis to identify where models fail
- **Confusion Matrix** — visual analysis of class-level misclassifications
- **Grad-CAM Qualitative Analysis** — visual validation that the model focuses on
  pain-relevant facial regions (brow, eyes, mouth) rather than artifacts

---

## 7. Project Schedule

| Week | Tasks |
|------|-------|
| Week 1 | Dataset access/download, EDA, class distribution analysis |
| Week 2 | Data preprocessing pipeline: face detection, cropping, augmentation, splits |
| Week 3 | Train Custom CNN baseline; establish performance floor |
| Week 4 | Train VGG-16 and ResNet-50 with transfer learning |
| Week 5 | Train EfficientNet-B3; implement Grad-CAM explainability |
| Week 6 | Model comparison, evaluation, generate all plots and tables |
| Week 7 | Build and refine Streamlit demo app (webcam + report) |
| Week 8 | Write final report, prepare presentation, finalize GitHub repo |

---

## 8. Individual Contributions Plan

| Member | Responsibility |
|--------|---------------|
| Member 1 | Data preprocessing, face detection pipeline, dataset analysis |
| Member 2 | Custom CNN architecture and baseline training |
| Member 3 | VGG-16 and ResNet-50 transfer learning |
| Member 4 | EfficientNet-B3 training and Grad-CAM implementation |
| Member 5 | Streamlit app, evaluation comparison, report writing |

*(Adjust based on actual group size)*
