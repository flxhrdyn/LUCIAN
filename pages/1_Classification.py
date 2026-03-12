"""Classification page — image upload, real-time inference, and Grad-CAM display."""
import sys
from pathlib import Path

# Same sys.path fix as other page files — see pages/Home.py for explanation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from src.config import CLASS_LABELS_EN, CLASS_LABELS_ID, CLASS_COLORS, DEMO_IMAGES
from src.gradcam import compute_gradcam, overlay_heatmap
from src.model import load_model, predict, preprocess_image
from src.styles import apply_css, render_footer, render_sidebar_info

apply_css()
render_sidebar_info()

model, _ = load_model()


def _run_inference(source) -> None:
    """Preprocess *source*, run prediction + Grad-CAM, store results in session_state."""
    try:
        img, img_array = preprocess_image(source)
    except ValueError as exc:
        st.error(f"⚠️ {exc}")
        # Clear stale results so previous prediction is not shown alongside the error.
        for key in ("uploaded_image", "img_array", "probs", "predicted_idx", "inference_time", "gradcam_heatmap"):
            st.session_state.pop(key, None)
        return
    probs, inference_time = predict(model, img_array)
    predicted_idx = int(np.argmax(probs))
    heatmap = compute_gradcam(model, img_array, predicted_idx)
    st.session_state.update({
        "uploaded_image": img,
        "img_array": img_array,
        "probs": probs,
        "predicted_idx": predicted_idx,
        "inference_time": inference_time,
        "gradcam_heatmap": heatmap,
    })


st.markdown('<div class="section-badge">AI INFERENCE</div>', unsafe_allow_html=True)
st.title("🔬 Histopathology Image Classification")
st.markdown(
    "Upload a lung histopathology image **(JPG or PNG)** to classify the tissue type "
    "using the fine-tuned **ConvNeXt-Base** model."
)

# ── Try Demo buttons ──────────────────────────────────────────
st.markdown("""
<div style='font-size:0.8rem; color:#888; margin-bottom:4px'>
    💡 No image? Try a built-in sample:
</div>
""", unsafe_allow_html=True)
demo_cols = st.columns(3)
demo_labels = [
    ("🔴 Try LUAD Sample",   "LUAD Sample"),
    ("🟢 Try Benign Sample", "Benign Sample"),
    ("🟠 Try LUSC Sample",   "LUSC Sample"),
]
for col, (btn_label, key) in zip(demo_cols, demo_labels):
    if col.button(btn_label, use_container_width=True):
        with open(DEMO_IMAGES[key], "rb") as f:
            _run_inference(f)

st.markdown("")
col_upload, col_info = st.columns([2, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Drop your histopathology image here",
        type=["jpg", "jpeg", "png"],
        help="Supported: JPG, JPEG, PNG · Max 10MB",
    )

with col_info:
    st.markdown("""
    <div class="feature-card">
        <strong style="color:#fafafa">Classifiable tissue types:</strong>
        <ul style="margin-top:10px; padding-left:18px">
            <li><span style="color:#e74c3c">●</span> Adenocarcinoma (LUAD)</li>
            <li><span style="color:#2ecc71">●</span> Benign Lung Tissue</li>
            <li><span style="color:#e67e22">●</span> Squamous Cell Carcinoma (LUSC)</li>
        </ul>
        <p style="margin-top:14px; border-top:1px solid #2a2a3a; padding-top:12px">
            Input size: <strong style="color:#fafafa">224 × 224 px</strong> (auto-resized)<br>
            Architecture: <strong style="color:#fafafa">ConvNeXt-Base</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    _run_inference(uploaded_file)

# session_state persists results across reruns — without this, Streamlit would
# clear the prediction every time the user interacts with anything on the page.
if "predicted_idx" in st.session_state:
    idx = st.session_state["predicted_idx"]
    probs = st.session_state["probs"]
    img = st.session_state["uploaded_image"]
    color = CLASS_COLORS[idx]

    st.markdown("---")
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.image(img, caption="Input Histopathology Image", use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-card" style="border-left-color:{color}">
            <div class="prediction-label" style="color:{color}">{CLASS_LABELS_EN[idx]}</div>
            <div class="prediction-sub">{CLASS_LABELS_ID[idx]}</div>
            <div class="stat-row">
                <div class="stat-item">
                    <div class="s-label">Confidence</div>
                    <div class="s-value">{probs[idx]*100:.2f}%</div>
                </div>
                <div class="stat-item">
                    <div class="s-label">Inference Time</div>
                    <div class="s-value">{st.session_state['inference_time']:.4f}s</div>
                </div>
                <div class="stat-item">
                    <div class="s-label">Architecture</div>
                    <div class="s-value" style="font-size:1rem; padding-top:4px">ConvNeXt-B</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probability Distribution:**")
        for label, prob in zip(CLASS_LABELS_EN, probs):
            st.progress(float(prob), text=f"{label}: {prob*100:.2f}%")

        # below 70% confidence the prediction is too uncertain to be useful
        if probs[idx] < 0.70:
            st.warning(
                "⚠️ **Low confidence** — The model is less than 70% certain. "
                "This result may be unreliable. Please consult a pathologist for a definitive diagnosis.",
                icon="🔬",
            )

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    if "gradcam_heatmap" in st.session_state:
        with st.expander("🔥 Grad-CAM Explainability", expanded=False):
            st.markdown(
                "**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights "
                "the regions of the image that most influenced the model's prediction. "
                "Warmer colors (red/yellow) indicate higher importance."
            )
            overlay = overlay_heatmap(
                st.session_state["uploaded_image"],
                st.session_state["gradcam_heatmap"],
            )
            gcam_col1, gcam_col2 = st.columns(2, gap="medium")
            with gcam_col1:
                st.image(
                    st.session_state["uploaded_image"],
                    caption="Original Image",
                    use_container_width=True,
                )
            with gcam_col2:
                st.image(
                    overlay,
                    caption=f"Grad-CAM — {CLASS_LABELS_EN[idx]}",
                    use_container_width=True,
                )

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is intended for
        <strong>research and educational purposes only</strong>.
        It is not a substitute for professional medical diagnosis.
        Always consult a qualified pathologist or physician for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

render_footer()
