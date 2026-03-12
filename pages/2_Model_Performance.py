"""Model performance page — training curves, classification report, and confusion matrix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd
import streamlit as st

from src.config import ASSETS_73_SPLIT, ASSETS_82_SPLIT
from src.styles import apply_css, render_footer, render_sidebar_info


@st.cache_data(show_spinner=False)
def _img(path: str) -> bytes:
    # Cached so repeated expander opens don't re-read from disk.
    with open(path, "rb") as f:
        return f.read()

apply_css()
render_sidebar_info()

st.markdown('<div class="section-badge">EVALUATION</div>', unsafe_allow_html=True)
st.title("📊 Model Performance")
st.markdown(
    "**ConvNeXt-Base** fine-tuned on 3,000 lung histopathology images from the "
    "[LC25000 dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images). "
    "Two data split strategies were benchmarked."
)

st.markdown("")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Train Accuracy", "96.08%", help="Best epoch — training set")
col2.metric("Validation Accuracy", "96.67%", delta="+0.59% vs train")
col3.metric("Test Accuracy", "93.67%", delta="-3.00% vs val")
col4.metric("F1-Score (macro)", "93.64%", help="Macro-averaged across 3 classes")

st.divider()

# ── Architecture ──────────────────────────────────────────────────────────────
with st.expander("📌 Model Architecture"):
    st.markdown(
        """
        ConvNeXt is a modern CNN architecture that adopts design principles from Vision Transformers (ViT)
        while retaining the efficiency of standard convolutions.

        **Modifications applied for this task:**
        - Base: `ConvNeXt-Base` pretrained on ImageNet-1K
        - Head: Global Average Pooling → Dense (256, ReLU) → Dropout (0.3) → Dense (3, Softmax)
        - Training: Transfer learning + gradual fine-tuning
        - Optimizer: `AdamW` &nbsp;·&nbsp; Loss: `Categorical Cross-Entropy` &nbsp;·&nbsp; Epochs: `55` &nbsp;·&nbsp; Batch size: `32`
        """
    )
    st.image(
        _img(f"{ASSETS_82_SPLIT}/model_sum_82.png"),
        caption="Final ConvNeXt Model Architecture",
        width=700,
    )

# ── Training Curves ───────────────────────────────────────────────────────────
with st.expander("📈 Training Curves & Classification Report", expanded=True):
    st.image(
        _img(f"{ASSETS_82_SPLIT}/train_acc_loss_82.png"),
        caption="Training & Validation Accuracy / Loss over 55 Epochs",
        width=700,
    )

    st.markdown("#### Classification Report — Test Set (80:10:10 Split)")
    with open(f"{ASSETS_82_SPLIT}/class_report_82.json") as f:
        report = json.load(f).get("classification_report", {})
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
with st.expander("📊 Confusion Matrix"):
    st.markdown(
        "The model occasionally misclassifies **Adenocarcinoma** as **Squamous Cell Carcinoma** "
        "due to morphological similarity. **Benign tissue** achieves near-perfect recall **(100%)**."
    )
    st.image(
        _img(f"{ASSETS_82_SPLIT}/conf_matrix_82.png"),
        caption="Confusion Matrix — Test Set (80:10:10)",
        width=700,
    )

# ── Test Predictions ──────────────────────────────────────────────────────────
with st.expander("🔮 Sample Test Predictions"):
    st.markdown("Sample predictions on 300 held-out test images from the 80:10:10 split.")
    st.image(
        _img(f"{ASSETS_82_SPLIT}/test_predict_82.png"),
        caption="Model Predictions on Test Set",
        width=700,
    )

# ── Split Comparison ──────────────────────────────────────────────────────────
with st.expander("🔁 Data Split Strategy Comparison: 80:10:10 vs 70:15:15"):
    comp_data = {
        "Metric": [
            "Train Accuracy",
            "Validation Accuracy",
            "Test Accuracy",
            "Precision (macro)",
            "Recall (macro)",
            "F1-Score (macro)",
        ],
        "80:10:10 (Final)": ["96.08%", "96.67%", "93.67%", "93.63%", "93.67%", "93.64%"],
        "70:15:15 (Baseline)": ["94.95%", "94.00%", "90.44%", "90.47%", "90.44%", "90.39%"],
        "Improvement": ["+1.13%", "+2.67%", "+3.23%", "+3.16%", "+3.23%", "+3.25%"],
    }
    st.table(pd.DataFrame(comp_data).set_index("Metric"))
    st.info(
        "The **80:10:10 split** provides 14% more training data (2,400 vs 2,100 images), "
        "resulting in a consistent ~+3% improvement on evaluation metrics. "
        "For 70:15:15, the train/val entries come from the best checkpoint selected by val_accuracy."
    )

render_footer()
