# 🔬 LUCIAN — Lung Carcinoma Histopathology Imaging & Analysis

![CI](https://github.com/flxhrdyn/LUCIAN/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

**LUCIAN** *(Lung Carcinoma Histopathology Imaging & Analysis)* is an AI-powered web application for classifying lung tissue types from histopathology slides using a fine-tuned **ConvNeXt-Base** model. Developed as an undergraduate thesis applying the **CRISP-DM** methodology, and extended post-thesis with production-grade code structure and deployment-ready architecture.

---

## 🎯 Problem Statement

Lung cancer is one of the leading causes of cancer-related mortality worldwide. Early and accurate classification of histopathological tissue types — Adenocarcinoma (LUAD), Squamous Cell Carcinoma (LUSC), and Benign tissue — is critical for treatment planning. Manual diagnosis by pathologists is time-consuming and prone to inter-observer variability. This project explores using a state-of-the-art CNN to assist in automated classification.

---

## 🧠 Model & Architecture

- **Base model:** [ConvNeXt-Base](https://arxiv.org/abs/2201.03545) pretrained on ImageNet-1K
- **Approach:** Transfer learning + fine-tuning
- **Added layers:** Global Average Pooling → Dense (256, ReLU) → Dropout (0.3) → Dense (3, Softmax)
- **Input size:** 224 × 224 px
- **Dataset:** [LC25000 Lung and Colon Histopathology Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) — 3,000 lung images (3 classes × 1,000 images)
- **Model hosted on:** [HuggingFace 🤗](https://huggingface.co/felixhrdyn/convnextv1-lung-cancer)

---

## 📊 Model Performance

Two data split experiments were conducted:

| Metric | Split 80:10:10 (Final) | Split 70:15:15 |
|--------|----------------------|----------------|
| Train Accuracy | 96.08% | — |
| Validation Accuracy | 96.67% | — |
| **Test Accuracy** | **93.67%** | **90.44%** |
| Precision (macro) | 93.63% | 90.47% |
| Recall (macro) | 93.67% | 90.44% |
| F1-Score (macro) | 93.64% | 90.39% |

### Per-Class Performance (80:10:10 Split)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Adenocarcinoma | 91.75% | 89.00% | 90.36% | 100 |
| Benign | 98.04% | 100.00% | 99.01% | 100 |
| Squamous Cell Carcinoma | 91.09% | 92.00% | 91.54% | 100 |

---

## 🔥 Grad-CAM Explainability

LUCIAN includes **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize which regions of the histopathology image most influenced the model's prediction. Warmer colors (red/yellow) indicate higher importance.

The implementation targets the `flatten` input tensor — the 7×7×1024 feature map immediately after `convnext_base` — which lies in the outer model's symbolic graph, resolving the Keras 3 `KeyError` that occurs when referencing tensors from a nested sub-model's isolated graph. The grad model is cached via `functools.lru_cache` to avoid repeated `tf.function` retracing.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Setup Environment

```bash
python -m venv myvenv
myvenv\Scripts\activate   # Windows
# source myvenv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. The model is loaded from the local `.keras` file on first run, or automatically downloaded from HuggingFace if not present. Download uses a **temp file + atomic rename** to prevent corrupt model files from failed downloads.

### Run Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
LUCIAN/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions — install deps + run pytest on push
├── .streamlit/
│   └── config.toml              # Theme & server configuration
├── src/
│   ├── __init__.py              # Package version & author
│   ├── config.py                # Centralized constants (URLs, labels, colors, paths)
│   ├── gradcam.py               # Grad-CAM computation & heatmap overlay
│   ├── model.py                 # load_model(), preprocess_image(), predict()
│   └── styles.py                # CSS, footer, sidebar components
├── pages/
│   ├── Home.py                  # Landing page with model status
│   ├── 1_Classification.py      # Image upload & real-time prediction
│   ├── 2_Model_Performance.py   # Training metrics, classification report, confusion matrix
│   └── 3_Lung_Cancer_Info.py    # Clinical background for each cancer type
├── tests/
│   ├── conftest.py              # Shared pytest fixtures (fake images, arrays)
│   ├── test_config.py           # Constants, paths, demo image validation
│   ├── test_model.py            # preprocess_image (happy path + error cases), predict
│   └── test_gradcam.py          # overlay_heatmap, compute_gradcam with tiny model
├── assets/
│   ├── model_performance_82split/   # Plots & report for 80:10:10 split
│   ├── model_performance_73split/   # Plots & report for 70:15:15 split
│   └── lung_cancer_image_demo/      # Sample images per class
├── notebooks/
│   ├── 01_training_80_10_10_split.ipynb   # Main training experiment (CRISP-DM)
│   └── 02_training_70_15_15_split.ipynb   # Baseline experiment
├── models/                      # Placeholder — model weights are gitignored
├── app.py                       # Streamlit entry point
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies (pytest)
└── readme.md
```

---

## 🖥️ App Features

- **Home** — Landing page with model status and navigation overview
- **Image Classification** — Upload histopathology images (JPG/PNG) for real-time prediction; displays per-class confidence bars, inference time, and **Grad-CAM heatmap**
- **Model Performance** — Training curves, classification report, confusion matrix, sample test predictions, and a comparison between the two split strategies
- **Lung Cancer Info** — Detailed clinical descriptions of each tissue type (LUAD, LUSC, Benign) with academic references

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow 2.19 / Keras |
| Model Architecture | ConvNeXt-Base |
| Explainability | Grad-CAM |
| Web Framework | Streamlit 1.45 |
| Model Hosting | HuggingFace Hub |
| Data Processing | NumPy, Pandas |
| Testing | pytest |
| CI | GitHub Actions |
| Methodology | CRISP-DM |

---

## ⚠️ Limitations

- **Small dataset** — trained on 3,000 images (1,000 per class); performance may degrade on out-of-distribution staining protocols or scanner types
- **Single institution** — LC25000 originates from one source; no external validation has been performed
- **Not clinically validated** — this tool is for research and educational purposes only and has not undergone clinical trials or regulatory review
- **3-class scope** — does not cover other lung cancer subtypes (e.g., large cell carcinoma, small cell carcinoma)

---

## 🔭 Future Work

- [ ] **Test-Time Augmentation (TTA)** — average predictions over flipped/rotated variants to improve robustness
- [ ] **ONNX export** — convert model for faster, framework-agnostic inference
- [ ] **Docker deployment** — containerize app for reproducible, one-command deployment
- [ ] **External dataset validation** — evaluate on TCGA or other public lung pathology datasets
- [ ] **Attention map alternatives** — explore GradCAM++, ScoreCAM for sharper localization

---

## 📚 References

- Liu, Z., et al. (2022). *A ConvNet for the 2020s*. CVPR. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Borkowski, A.A., et al. (2019). *Lung and Colon Cancer Histopathological Image Dataset (LC25000)*. [arXiv:1912.12378](https://arxiv.org/abs/1912.12378)
- Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Succony, L., et al. (2021). *Adenocarcinoma spectrum lesions of the lung*. Cancer Treatment Reviews, 99, 102237.
- Berezowska, S., et al. (2024). *Pulmonary squamous cell carcinoma*. Histopathology, 84(1), 32–49.

---

## 👤 Author

**Felix Hardyan**
- HuggingFace: [felixhrdyn](https://huggingface.co/felixhrdyn)

---

*Undergraduate Thesis Project — Computer Science, 2025*

**LUCIAN** *(Lung Carcinoma Histopathology Imaging & Analysis)* is an AI-powered web application for classifying lung tissue types from histopathology slides using a fine-tuned **ConvNeXt-Base** model. Developed as an undergraduate thesis applying the **CRISP-DM** methodology, and extended post-thesis with production-grade code structure and deployment-ready architecture.

---

## 🎯 Problem Statement

Lung cancer is one of the leading causes of cancer-related mortality worldwide. Early and accurate classification of histopathological tissue types — Adenocarcinoma (LUAD), Squamous Cell Carcinoma (LUSC), and Benign tissue — is critical for treatment planning. Manual diagnosis by pathologists is time-consuming and prone to inter-observer variability. This project explores using a state-of-the-art CNN to assist in automated classification.

---

## 🧠 Model & Architecture

- **Base model:** [ConvNeXt-Base](https://arxiv.org/abs/2201.03545) pretrained on ImageNet-1K
- **Approach:** Transfer learning + fine-tuning
- **Added layers:** Global Average Pooling → Dense (256, ReLU) → Dropout (0.3) → Dense (3, Softmax)
- **Input size:** 224 × 224 px
- **Dataset:** [LC25000 Lung and Colon Histopathology Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) — 3,000 lung images (3 classes × 1,000 images)
- **Model hosted on:** [HuggingFace 🤗](https://huggingface.co/felixhrdyn/convnextv1-lung-cancer)

---

## 📊 Model Performance

Two data split experiments were conducted:

| Metric | Split 80:10:10 (Final) | Split 70:15:15 |
|--------|----------------------|----------------|
| Train Accuracy | 96.08% | — |
| Validation Accuracy | 96.67% | — |
| **Test Accuracy** | **93.67%** | **90.44%** |
| Precision (macro) | 93.63% | 90.47% |
| Recall (macro) | 93.67% | 90.44% |
| F1-Score (macro) | 93.64% | 90.39% |

### Per-Class Performance (80:10:10 Split)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Adenocarcinoma | 91.75% | 89.00% | 90.36% | 100 |
| Benign | 98.04% | 100.00% | 99.01% | 100 |
| Squamous Cell Carcinoma | 91.09% | 92.00% | 91.54% | 100 |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Setup Environment

```bash
python -m venv myvenv
myvenv\Scripts\activate   # Windows
# source myvenv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. The model is loaded from the local file on first run, or downloaded from HuggingFace if not present.

---

## 📁 Project Structure

```
lung-cancer-classifier/
├── .gitignore
├── .streamlit/
│   └── config.toml              # Theme & server configuration
├── src/
│   ├── __init__.py
│   ├── config.py                # Centralized constants (URLs, labels, colors, paths)
│   └── model.py                 # load_model(), preprocess_image(), predict()
├── pages/
│   ├── 1_Classification.py      # Image upload & real-time prediction
│   ├── 2_Model_Performance.py   # Training metrics, classification report, confusion matrix
│   └── 3_Lung_Cancer_Info.py    # Clinical background for each cancer type
├── models/
│   └── .gitkeep                 # Placeholder — model weights are gitignored
├── assets/
│   ├── model_performance_82split/   # Plots & report for 80:10:10 split
│   ├── model_performance_73split/   # Plots & report for 70:15:15 split
│   ├── luad.jpg
│   ├── lusc.jpg
│   └── benign.jpg
├── notebooks/
│   ├── 01_training_80_10_10_split.ipynb   # Main training experiment (CRISP-DM)
│   └── 02_training_70_15_15_split.ipynb   # Baseline experiment
├── app.py                       # Entry point & home page
├── requirements.txt
└── readme.md
```

---

## 🖥️ App Features

- **Home** — Landing page with model status and navigation overview
- **Image Classification** — Upload histopathology images (JPG/PNG) for real-time prediction; displays per-class confidence bars and inference time
- **Model Performance** — Training curves, classification report, confusion matrix, sample test predictions, and a comparison between the two split strategies
- **Lung Cancer Info** — Detailed clinical descriptions of each tissue type (LUAD, LUSC, Benign) with academic references

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow 2.19 | Keras |
| Model Architecture | ConvNeXt-Base |
| Web Framework | Streamlit 1.45 |
| Model Hosting | HuggingFace Hub |
| Data Processing | NumPy, Pandas |
| Methodology | CRISP-DM |

---

## 📚 References

- Liu, Z., et al. (2022). *A ConvNet for the 2020s*. CVPR. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Borkowski, A.A., et al. (2019). *Lung and Colon Cancer Histopathological Image Dataset (LC25000)*. [arXiv:1912.12378](https://arxiv.org/abs/1912.12378)
- Succony, L., et al. (2021). *Adenocarcinoma spectrum lesions of the lung*. Cancer Treatment Reviews, 99, 102237.
- Berezowska, S., et al. (2024). *Pulmonary squamous cell carcinoma*. Histopathology, 84(1), 32–49.

---

## 👤 Author

**Felix Hardyan**
- HuggingFace: [felixhrdyn](https://huggingface.co/felixhrdyn)

---

*Undergraduate Thesis Project — Computer Science, 2025*