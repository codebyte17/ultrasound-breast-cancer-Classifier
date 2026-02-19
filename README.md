# Ultrasound Breast Cancer Classification (Focus on Malignant Sensitivity)

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Related Work](#2-related-work)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architectures](#model-architectures)
   - [Training Strategy](#training-strategy)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Results and Discussion](#6-results-and-discussion)
7. [Repository Structure](#7-repository-structure)
8. [How to Run the Project](#8-how-to-run-the-project)
9. [Limitations and Future Work](#9-limitations-and-future-work)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Project Overview

Breast cancer is one of the most common life-threatening diseases among women worldwide. Early diagnosis significantly improves treatment outcomes. Ultrasound imaging is a non-invasive and cost-effective modality and is especially suitable for dense breast tissue.

This project implements an automated deep learningâ€“based system to classify breast ultrasound images into three categories: normal, benign, and malignant. Multiple convolutional neural network architectures and transfer learning approaches are explored and compared to analyze performance, robustness, and clinical relevance.

---

## 2. Related Work

Recent research in medical image analysis has demonstrated the effectiveness of convolutional neural networks for ultrasound image classification. Transfer learning using ImageNet-pretrained models such as ResNet and AlexNet is commonly applied to overcome limited dataset size and improve feature representation.

The Breast Ultrasound Images Dataset (BUSI) has been widely used in prior studies for automated breast lesion classification and provides both image-level labels and lesion masks.

---

## 3. Dataset

The project uses the Breast Ultrasound Images Dataset (BUSI), which consists of ultrasound images grouped into three classes:
- `               Normal               ,                  Benign         ,                    Malignant     `
- <img width="268" height="170" alt="normal (1)" src="https://github.com/user-attachments/assets/d32be2ba-d8ae-4443-8bd4-ee6098db2ab0" />  <img width="268" height="170" alt="benign (2)" src="https://github.com/user-attachments/assets/bb1c71f6-f862-48fc-b864-d24f383d0508" />  <img width="268" height="170" alt="malignant (1)" src="https://github.com/user-attachments/assets/b81889cf-1e30-473a-99aa-44c8faa309db" />
**Dataset Source:** https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset

**Dataset Statistics:**

| Class      | Number of Images | Percentage |
|------------|----------------|-----------|
| Normal     | 133            | 17%       |
| Benign     | 437            | 56%       |
| Malignant  | 210            | 27%       |
| Total      | 780            | 100%      |

- Image resolution: 500 Ã— 500 pixels


For benign and malignant cases, corresponding ground-truth lesion masks are available. These masks are used in one experimental setup to generate overlay images that emphasize lesion regions.

---

## 4. Methodology

### Data Preprocessing

Data preprocessing is handled through scripts located in the `tools/` directory.

The preprocessing pipeline includes:
- Automated dataset download
- Image resizing to 224 Ã— 224
- RGB conversion
- Label extraction from directory structure
- Generation of an overlay dataset using lesion masks:
  - Background regions are darkened (80% reduction) for benign and malignant images
  - Normal images remain unchanged

### Model Architectures

Five model configurations were implemented and evaluated:

| Experiment | Model | Data | Key Features |
|------------|-------|------|--------------|
| 1 | Custom CNN | Original | Class weights, balanced sampler, 75 epochs |
| 2 | ResNet50 | Original | Transfer learning, frozen backbone |
| 3 | Custom CNN | Overlay (masked) | Hyperparameter-optimized, 75 epochs |
| 4 | Custom CNN | Original | Adam optimizer, no class weights, 35 epochs |
| 5 | AlexNet | Original | Transfer learning, frozen features, 20 epochs |

**Custom CNN Architecture:**
- 4 convolutional blocks (32 â†’ 64 â†’ 128 â†’ 256 filters)
- BatchNorm + ReLU after each convolution
- MaxPooling after each block
- AdaptiveAvgPool2d to 7Ã—7
- Classifier: 3 linear layers with Dropout(0.4)

  <img width="1795" height="935" alt="image" src="https://github.com/user-attachments/assets/335eb78b-3183-4d8b-86df-feb4ea69c16a" />


### Training Strategy

All models are trained using supervised learning with cross-entropy loss.

Training configuration includes:
- **Optimizers**: SGD (with momentum) and AdamW
- **Learning rate scheduling**: Cosine annealing with warm-up (5 epochs)
- **Batch size**: 32 (variable in HPO experiments)
- **Class imbalance handling**: WeightedRandomSampler + weighted CrossEntropyLoss
- **Hardware**: GPU acceleration when available

<img width="1523" height="757" alt="image" src="https://github.com/user-attachments/assets/983bc9cd-33ec-4be4-b173-92e0bda1efab" />

---

## 5. Evaluation Metrics

Model performance is evaluated using the following metrics:

- Precision
- Recall (Sensitivity)
- F1-Score
- Accuracy
- Confusion Matrix

Malignant detection is additionally treated as a binary classification task with threshold tuning to achieve â‰¥95% recall, reflecting clinical screening priorities.

---

## 6. Results and Discussion

All models are evaluated on a held-out test set using consistent evaluation metrics.

### Key Results

| Model | Malignant Recall | Malignant Precision | Accuracy | Balanced Accuracy |
|-------|-----------------|---------------------|----------|-------------------|
| Custom CNN + Overlay (Exp 3) | **94%** | 84% | 94% | 95% |
| Custom CNN (Exp 1) | **94%** | 84% | 87% | 95% |
| Custom CNN + Adam (Exp 4) | 91% | 88% | 82% | 88% |
| AlexNet (Exp 5) | 88% | 85% | 79% | 86% |
| ResNet50 (Exp 2) | 64% | 70% | 74% | 72% |

**Best Model Performance (Custom CNN + Overlay):**
- Malignant: 94% recall, 84% precision, F1-score 0.89
- Benign: 97% precision, 92% recall
- Normal: 100% precision, 100% recall

Visual results include:
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrices

<img width="685" height="507" alt="image" src="https://github.com/user-attachments/assets/adeeb6f0-c2a7-43d8-96e7-be88041021b8" />
<img width="603" height="152" alt="image" src="https://github.com/user-attachments/assets/f5f42832-166d-424d-b984-ebc59aa9268b" />
<img width="1271" height="537" alt="image" src="https://github.com/user-attachments/assets/f70b35da-8b8f-48a4-b040-b7a97f82f1d5" />


---

## 7. Repository Structure
```
├── src/                  # Core datasets, training engine, and utilities
│   ├── datasets/         # Dataset classes and data loaders
│   ├── engine/           # Training and evaluation loops
│   └── utils/            # Device management utilities
├── tools/                 # Data download and preprocessing scripts
│   ├── data_downloader.py
│   ├── data_preprocessing.py
│   └── data_split.py
├── notebooks/             # Experimental notebooks and analysis
│   ├── Custom Model - Masked data- Experiment 3.ipynb
│   ├── Custom Model with Original Data - Experiment 1.ipynb
│   ├── Transfer Learning-AlexNet - Experiment 5.ipynb
│   ├── Transfer Learning-RESNET50- Experiment 2.ipynb
│   └── final plots of results.ipynb
├── configs/               # YAML configuration files
│   ├── cnn_model_config.yml
│   ├── data_config.yaml
│   └── training_config.yaml
├── run.py                 # Main training entry point
├── requirements.txt       # Project dependencies
└── README.md              # This file
```



---

## 8. How to Run the Project

Always run the following commands from the project root directory.

```bash
python -m tools.data_downloader  #Download the dataset from kaggle and save to folder
python -m tools.data_preprocessing  # Perform the required data preprocessing (Masked image x Orignal Images)
python run.py  # Run the final Model training with your specified setting in .yml file.
```
---

## 9. Limitations and Future Work

The project is limited by dataset size and lack of external validation. Classification is performed independently from segmentation.

Future work may include attention-based models, vision transformers, multi-task learning, and clinical decision threshold calibration.


## 10. Acknowledgements

This project was developed as part of an academic coursework submission. The authors acknowledge the creators of the BUSI dataset and the open-source deep learning community.