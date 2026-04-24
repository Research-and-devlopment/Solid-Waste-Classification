# 🗑️ Hybrid Deep Learning for Municipal Solid Waste Classification

> **"A Hybrid Deep Learning and Enhanced Nature-Inspired Classification Networks for an Effective Classification of Municipal Solid Wastes"**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This repository provides the official end-to-end training code for a **hybrid deep learning framework** that classifies municipal solid waste into **organic** and **recyclable** categories using:

- **Attention-Evoked Dilated Convolutional Networks (AE-DC)** — multi-scale spatial feature extraction
- **Residual Gated Recurrent Units (Res-GRU)** — spatio-temporal feature refinement
- **Extreme Learning Machine (ELM)** — fast feedforward classification
- **Artificial Rain Water Drop Optimization (ARWDO)** — automated hyperparameter tuning

**Achieved Performance (Kaggle Solid Waste Dataset):**

| Metric | Score |
|---|---|
| Accuracy | 98.53% |
| Precision | 98.3% |
| Recall | 97.6% |
| F1-Score | 98.0% |
| AUC | 0.997 |

---

## 🗂️ Project Structure

```
waste_classification/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths
│
├── data/
│   └── dataset.py               # Dataset loader + augmentation
│
├── models/
│   ├── ae_dc_block.py           # Attention-Evoked Dilated Conv Block
│   ├── residual_gru.py          # Residual GRU networks
│   ├── elm_classifier.py        # Extreme Learning Machine classifier
│   └── hybrid_model.py          # Full hybrid model assembly
│
├── utils/
│   ├── arwdo_optimizer.py       # Artificial Rain Water Drop Optimizer
│   ├── metrics.py               # Evaluation metrics
│   ├── visualization.py         # Plots: confusion matrix, ROC, loss curves
│   └── early_stopping.py        # Early stopping callback
│
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/waste-classification-hybrid-dl.git
cd waste-classification-hybrid-dl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

### Option 1: Kaggle Waste Classification Dataset (Primary)
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d techsash/waste-classification-data
unzip waste-classification-data.zip -d data/raw/
```

### Option 2: TrashNet
```bash
# Download from: https://github.com/garythung/trashnet
```

### Option 3: TACO Dataset
```bash
# Download from: http://tacodataset.org/
```

**Expected directory structure after download:**
```
data/raw/
├── TRAIN/
│   ├── O/        # Organic waste images
│   └── R/        # Recyclable waste images
└── TEST/
    ├── O/
    └── R/
```

---

## 🚀 Training

### Quick Start
```bash
python scripts/train.py --config configs/config.yaml
```

### Custom Configuration
```bash
python scripts/train.py \
  --data_dir data/raw \
  --epochs 150 \
  --batch_size 32 \
  --lr 0.0001 \
  --optimizer arwdo \
  --save_dir results/
```

---

## 📊 Evaluation
```bash
python scripts/evaluate.py \
  --checkpoint results/best_model.h5 \
  --test_dir data/raw/TEST
```
---

## 📈 Results

The model achieves state-of-the-art performance compared to baselines:

| Model | Accuracy | F1-Score |
|---|---|---|
| CNN | 72.3% | 70.9% |
| CNN+LSTM | 77.5% | 75.4% |
| CNN+GRU | 78.9% | 75.8% |
| ResNet50+RCNN | 84.0% | 87.5% |
| CNN+Graph LSTM | 93.4% | 93.0% |
| **Proposed (Ours)** | **98.4%** | **98.0%** |

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{lavanya2025hybrid,
  title={A Hybrid Deep Learning and Enhanced Nature Inspired Classification Networks for an Effective Classification of Municipal Solid Wastes},
  author={Lavanya, K. and Thangamani, M.},
  journal={},
  year={2025}
}
```

---

## 📬 Contact

- **Lavanya K** — lavanyaaids@outlook.com
- **Dr. M. Thangamani** — manithangamani2@gmail.com
