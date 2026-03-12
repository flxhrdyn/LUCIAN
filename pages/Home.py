import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import MODEL_PATH
from src.model import load_model
from src.styles import apply_css, render_footer, render_sidebar_info

apply_css()
render_sidebar_info()

model_needs_loading = not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0
if model_needs_loading:
    with st.spinner("Downloading & loading ConvNeXt model..."):
        _, elapsed = load_model()
else:
    _, elapsed = load_model()

st.markdown(f"""
<div class="hero-banner">
    <div class="logo-container">
        <span class="logo-icon">🔬</span>
        <span class="logo-name">LUCIAN</span>
        <span class="logo-tagline">Lung Carcinoma Histopathology Imaging &amp; Analysis</span>
    </div>
    <div class="hero-subtitle">
        AI-assisted classification of lung tissue types from histopathology slides<br>
        using fine-tuned <strong>ConvNeXt-Base</strong> with transfer learning
    </div>
    <div>
        <span class="stat-pill">ConvNeXt-Base</span>
        <span class="stat-pill">93.67% Test Accuracy</span>
        <span class="stat-pill">3 Tissue Classes</span>
        <span class="stat-pill">LC25000 Dataset</span>
        <span class="stat-pill">Model ready in {elapsed:.2f}s</span>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <h3>Image Classification</h3>
        <p>Upload a histopathology image for real-time tissue classification
        with per-class confidence scores and inference timing.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3>Model Performance</h3>
        <p>Inspect training curves, classification report, confusion matrix,
        and a comparison of two data split strategies.</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧬</div>
        <h3>Clinical Background</h3>
        <p>Learn about Adenocarcinoma (LUAD), Squamous Cell Carcinoma (LUSC),
        and benign lung tissue with clinical references.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="badge-row">
    <span class="badge-label">Built with</span>
    <span class="tech-badge">TensorFlow 2.19</span>
    <span class="tech-badge">ConvNeXt-Base</span>
    <span class="tech-badge">Transfer Learning</span>
    <span class="tech-badge">Streamlit 1.45</span>
    <span class="tech-badge">HuggingFace Hub</span>
    <span class="tech-badge">LC25000 Dataset</span>
    <span class="tech-badge">CRISP-DM</span>
</div>
""", unsafe_allow_html=True)

render_footer()
