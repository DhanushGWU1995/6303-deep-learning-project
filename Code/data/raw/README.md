# Dataset Placeholder

  Raw dataset images live here. This folder is excluded from version control because the dataset is too large for GitHub.

  ## Recommended Dataset (RGB — Better Quality)

  **Balanced RGB FER Dataset (75×75)**
  https://www.kaggle.com/datasets/dollyprajapati182/balanced-image-fer-dataset-7575-rgb

  > Native 3-channel RGB input — works directly with ImageNet-pretrained models (VGG16, ResNet50, EfficientNet).

  ## Alternative Dataset (Grayscale)

  **Uniform FER2013 (48×48)**
  https://www.kaggle.com/datasets/sayakbera/fer-2013-7-emotions-uniform-dataset

  ---

  ## Download & Setup

  1. Download and unzip your chosen dataset from Kaggle
  2. Place contents so the structure looks like:

  ```
  Code/data/raw/
    train/
      angry/     disgust/    fear/
      happy/     neutral/    sad/     surprise/
    test/
      (same 7 folders)
    validation/       ← optional, will be pooled automatically
      (same 7 folders)
  ```

  3. Run from the Code/ directory:
     ```bash
     python 01_data_preprocessing.py
     ```

  The script will auto-detect the FER layout and apply this mapping:

  | Emotion | → Pain Level |
  |---------|-------------|
  | neutral, happy | no_pain |
  | sad | mild |
  | fear | moderate |
  | angry, disgust, surprise | severe |
  