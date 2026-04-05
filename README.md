# 🧠 Brain Tumor MRI Classification — IEEE Paper Implementation

> **Lab Assignment | Deep Learning | Transfer Learning | Pre-trained Models**

---

## 📄 Research Paper

| Field | Details |
|---|---|
| **Title** | Enhancing Brain Tumor Classification by a Comprehensive Study on Transfer Learning Techniques and Model Efficiency Using MRI Datasets |
| **Authors** | N. Shamshad et al. |
| **Journal** | IEEE Access, Volume 12, 2024 |
| **DOI** | [10.1109/ACCESS.2024.3430109](https://ieeexplore.ieee.org/document/10600700) |

---

## 👤 Student Details

| Field | Details |
|---|---|
| **Student Name** | Kulshree Sanjay Nakshane |
| **Student ID** | 202301040299 |
| **Course** | Deep Learning |
| **Date** | 05/04/2026 |
| **Group Members** | Preeti Koli, Vaishnavi Thorave, Sakshi Bhingarkar |

---

## 👥 Group Members

| Name | Student ID | Role |
|---|---|---|
| Preeti Koli | 202301040213 | Model Implementation & Training |
| Sakshi Bhingarkar | 202301040260 | Dataset Preparation & Preprocessing |
| Vaishnavi Thorave | 202301040261 | Model Evaluation & Visualization |
| Kulshree Nakshane | 202301040299 | Research Paper Study & Comparison |

---

## 🎯 Objective

This assignment reproduces the methodology of the above IEEE Access paper using Transfer Learning with pre-trained CNN models for automated **Brain Tumor MRI Classification**. The goal is to:

1. Study a published IEEE research paper utilizing pre-trained deep learning models
2. Reproduce the model implementation using the same dataset and methodology
3. Fine-tune pre-trained models and optimize hyperparameters
4. Evaluate and compare performance with the original paper's reported results

---

## 🗂️ Dataset

| Field | Details |
|---|---|
| **Name** | Brain Tumor MRI Dataset |
| **Source** | Kaggle |
| **Link** | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| **Total Images** | ~7,023 MRI Images |
| **Classes** | 4 — Glioma, Meningioma, No Tumor, Pituitary |
| **Format** | JPG, Grayscale MRI Scans |
| **Split** | 80% Train / 20% Validation / Separate Test Set |

### Class Distribution

| Class | Train | Test |
|---|---|---|
| Glioma | ~1,321 | ~300 |
| Meningioma | ~1,339 | ~306 |
| No Tumor | ~1,595 | ~405 |
| Pituitary | ~1,457 | ~300 |

---

## 🤖 Models Implemented

| Model | Base Architecture | Parameters | Strategy |
|---|---|---|---|
| **VGG16** | VGG-16 (ImageNet) | ~138M | Transfer Learning + Fine-tuning |
| **ResNet50** | ResNet-50 (ImageNet) | ~25M | Transfer Learning + Fine-tuning |
| **MobileNetV2** | MobileNetV2 (ImageNet) | ~3.4M | Transfer Learning + Fine-tuning |

### Training Strategy (2-Phase Fine-tuning as per paper)
```
Phase 1: Freeze entire base model → Train only custom classification head
         Optimizer: Adam | LR: 1e-3 | Epochs: 20

Phase 2: Unfreeze top layers → Fine-tune with lower LR
         Optimizer: Adam | LR: 1e-4 | Epochs: 15
```

### Custom Classification Head (5 layers added per paper)
```
Base Model (frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) → BatchNormalization → Dropout(0.5)
    ↓
Dense(256, ReLU) → Dropout(0.3)
    ↓
Dense(4, Softmax)   ← Output: 4 classes
```

---

## 📊 Results

### Our Implementation vs IEEE Paper

| Model | Paper Accuracy | Our Accuracy |
|---|---|---|
| VGG-16 | 97.0% |93.69% |
| ResNet-50 | 95.0% | 68.06% |
| MobileNetV2 | 95.0% | 80.62%|

### Metrics Evaluated
- Accuracy
- Precision (per class + weighted)
- Recall (per class + weighted)
- F1-Score (per class + weighted)
- Confusion Matrix
- ROC Curve + AUC Score

---

## 📁 Repository Structure

```
📦 Brain-Tumor-MRI-Classification/
├── 📓 IEEE_BrainTumor_Classification.ipynb   ← Main Colab Notebook
├── 📄 README.md                               ← This file
├── 📂 outputs/
│   ├── 🖼️ sample_images.png
│   ├── 🖼️ class_distribution.png
│   ├── 🖼️ augmented_samples.png
│   ├── 🖼️ feature_maps/
│   │   ├── fmap_block1_conv2.png
│   │   ├── fmap_block2_conv2.png
│   │   ├── fmap_block3_conv3.png
│   │   └── fmap_block4_conv3.png
│   ├── 🖼️ all_training_history.png
│   ├── 🖼️ confusion_matrices.png
│   ├── 🖼️ per_class_metrics.png
│   ├── 🖼️ roc_curves.png
│   ├── 🖼️ paper_vs_ours_comparison.png
│   └── 🖼️ sample_predictions.png
```

---

## 🔧 How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. Set Runtime to GPU
```
Runtime → Change Runtime Type → T4 GPU
```

### 3. Configure Kaggle API
- Go to **kaggle.com → Settings → API → Create New Token**
- Download `kaggle.json`
- Paste credentials in **Task 1 - Step 1** of the notebook:
```python
kaggle_json_content = '{"username":"your_username","key":"your_api_key"}'
```

### 4. Run All Cells
```
Runtime → Run All  (Ctrl + F9)
```

---

## 📦 Dependencies

```python
tensorflow >= 2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
kaggle
```

All dependencies are installed automatically in the first cell of the notebook.

---

## 🔬 Methodology Overview

```
1. Dataset Download (Kaggle API)
        ↓
2. Preprocessing
   • Resize → 224×224
   • Normalize → [0, 1]
   • Augmentation (rotation, zoom, flip, shear)
        ↓
3. Model Building
   • Load pre-trained model (ImageNet weights)
   • Freeze base layers
   • Add custom classification head
        ↓
4. Phase 1 Training (Frozen base, LR=1e-3)
        ↓
5. Phase 2 Fine-tuning (Unfrozen top layers, LR=1e-4)
        ↓
6. Evaluation
   • Accuracy / Precision / Recall / F1
   • Confusion Matrix
   • ROC Curves + AUC
        ↓
7. Comparison with IEEE Paper Results
```

---

## 📈 Key Visualizations

| Visualization | Description |
|---|---|
| Sample MRI Images | 2 samples per class (4 classes) |
| Class Distribution | Bar chart + Pie chart for train/test sets |
| Augmented Samples | 9 augmented versions of one MRI scan |
| Feature Maps | 4 VGG16 layers — block1 to block4 |
| Training History | Accuracy + Loss curves for all 3 models |
| Confusion Matrices | Side-by-side for VGG16, ResNet50, MobileNetV2 |
| Per-Class Metrics | Precision, Recall, F1 grouped bar chart |
| ROC Curves | Per-class ROC + AUC for all 3 models |
| Comparison Chart | Our results vs 6 models from IEEE paper |
| Sample Predictions | Test images with true/predicted labels |

---

## ⚠️ Potential Weaknesses & Improvements

1. **Class Imbalance** — Meningioma class is underrepresented; SMOTE or class weights could help
2. **Grad-CAM** — Adding gradient class activation maps would improve model explainability
3. **Ensemble Learning** — Combining VGG16 + ResNet50 predictions could improve accuracy
4. **Hyperparameter Tuning** — Keras Tuner for automated LR/dropout optimization
5. **More Models** — Implementing VGG19, InceptionV3, DenseNet121 (as in the full paper)

---

## 📚 References

[1] N. Shamshad et al.,  
"Enhancing Brain Tumor Classification by a Comprehensive Study on Transfer Learning Techniques and Model Efficiency Using MRI Datasets,"  
*IEEE Access*, vol. 12, pp. 100407–100418, 2024.  
DOI: https://doi.org/10.1109/ACCESS.2024.3430109

---

##  Submission Checklist

- [x] Research paper details and summary (IEEE Access 2024)  
- [x] Code file (Local Notebook / .ipynb)  
- [x] Dataset link and description 
- [x] Visualizations:
  - Feature maps  
  - Training curves  
  - Confusion matrix  
  - ROC curve  
- [x] Screenshots of model performance metrics  
- [x] README file  
- [x] Comparison with research paper results  

---

##  Declaration

We, **Preeti Koli, Sakshi Bhingarkar, Vaishnavi Thorave, and Kulshree Nakshane**,  
confirm that the work submitted in this assignment is our own and has been  
completed following academic integrity guidelines.

---
