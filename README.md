# 🔍 VisionInspect AI — Industrial Defect Detection With Limited Training Data

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

VisionInspect AI is a research-grade industrial visual inspection system that:

- **Learns normal product characteristics** from unlabeled or minimally labeled data
- **Detects anomalous images** using reconstruction-based and feature-embedding methods
- **Localizes defective regions** using anomaly heatmaps
- **Identifies unseen defect types** without requiring labeled defect examples
- **Quantifies prediction uncertainty** to route borderline cases to human review
- **Benchmarks against conventional supervised classifiers** under limited-data conditions

### Core Research Question
> Can anomaly detection and uncertainty-aware human review improve the detection of previously unseen manufacturing defects compared with conventional supervised classification when labeled defect data is limited?

---

## Project Structure

```
VisionInspect-AI/
├── data/                        # Datasets and preprocessing
│   ├── raw/                     # Raw MVTec AD / custom images
│   ├── processed/               # Preprocessed tensors
│   └── splits/                  # Train/val/test split configs
├── src/
│   ├── data/                    # Dataset loaders & augmentation
│   ├── models/                  # Anomaly detection architectures
│   │   ├── autoencoder.py       # Convolutional AE baseline
│   │   ├── padim.py             # PaDiM feature-embedding model
│   │   ├── patchcore.py         # PatchCore memory bank model
│   │   └── student_teacher.py   # Student-Teacher distillation
│   ├── uncertainty/             # Uncertainty quantification
│   │   ├── bayesian.py          # MC-Dropout uncertainty
│   │   └── conformal.py        # Conformal prediction intervals
│   ├── localization/            # Defect localization
│   │   ├── gradcam.py          # Gradient-based heatmaps
│   │   └── anomaly_map.py      # Feature-distance anomaly maps
│   ├── evaluation/              # Metrics & benchmarks
│   │   ├── metrics.py           # AUROC, PRO, F1, latency
│   │   └── benchmark.py         # Limited-data benchmark harness
│   ├── human_review/            # Human-in-the-loop interface
│   │   └── review_queue.py
│   └── utils/                   # Visualization, logging
├── experiments/                 # Experiment configs (YAML)
├── notebooks/                   # Jupyter exploration notebooks
├── results/                     # Saved metrics, heatmaps, reports
├── tests/                       # Unit and integration tests
├── requirements.txt
├── setup.py
└── train.py / evaluate.py / inspect.py
```

---

## Quick Start

### 1. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (cmd):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies inside the virtual environment
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Download MVTec AD Dataset
```bash
python scripts/download_mvtec.py --category bottle
```

### 3. Train an Anomaly Detection Model
```bash
python train.py --model patchcore --category bottle --config experiments/patchcore_mvtec.yaml
```

### 4. Evaluate & Generate Reports
```bash
python evaluate.py --model patchcore --category bottle --output results/
```

### 5. Run Inference on New Images
```bash
python inspect.py --image path/to/product.jpg --model patchcore --threshold 0.5
```

---

## Models Implemented

| Model | Type | Supervision | Key Strength |
|-------|------|-------------|--------------|
| **CAE** | Reconstruction | Unsupervised | Baseline, fast training |
| **PaDiM** | Feature Embedding | Unsupervised | Strong localization |
| **PatchCore** | Memory Bank | Unsupervised | State-of-the-art AUROC |
| **Student-Teacher** | Distillation | Unsupervised | Lightweight deployment |
| **MC-Dropout** | Bayesian | Semi-supervised | Uncertainty estimation |
| **Few-Shot Classifier** | Classification | Supervised (few-shot) | Comparison baseline |

---

## Evaluation Metrics

- **Image-level AUROC** — overall anomaly detection performance
- **Pixel-level PRO** (Per-Region Overlap) — localization accuracy
- **False Positive Rate** at 95% TPR
- **Inference Latency** (ms per image)
- **Few-shot performance curves** — AUROC vs. number of defect examples

---

## Research Contributions

1. **Limited-data benchmark**: Systematic evaluation with 1, 5, 10, 50 defect examples
2. **Uncertainty-gated human review**: Conformal prediction sets to route uncertain samples
3. **Cross-condition robustness**: Augmentation strategies for lighting & appearance variation
4. **Unified leaderboard**: Single framework comparing all models on MVTec AD

---

## License
MIT License — see [LICENSE](LICENSE)
