"""Clinical background page — descriptions and sample images for each tissue class."""
import base64
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import BASE_DIR
from src.styles import apply_css, render_footer, render_sidebar_info


@st.cache_data(show_spinner=False)
def _b64(path: str) -> str:
    # Images are embedded as base64 data URIs so they display inside raw HTML
    # blocks (st.markdown unsafe_allow_html) — st.image can't be used there.
    with open(path, "rb") as f:
        data = f.read()
    ext = Path(path).suffix.lstrip(".")
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"

apply_css()
render_sidebar_info()

ASSETS = BASE_DIR / "assets"

st.markdown('<div class="section-badge">CLINICAL BACKGROUND</div>', unsafe_allow_html=True)
st.title("🧬 Lung Cancer — Clinical Background")
st.markdown(
    """
    Lung cancer encompasses all malignancies originating in the lung, whether primary (arising
    from lung tissue itself) or metastatic. In clinical practice, the term *primary lung cancer*
    refers to malignant tumors arising from the bronchial epithelium (Joseph & Rotty, 2020).
    This classifier distinguishes **three tissue types** covered below.
    """
)

# ── LUAD ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cancer-card" style="border-left-color:#e74c3c;
     display:flex; gap:24px; align-items:center;">
    <div style="flex:3">
        <div class="cancer-title">1. 🔴 Adenocarcinoma (LUAD)</div>
        Lung Adenocarcinoma <strong>(LUAD)</strong> is the most common subtype of lung cancer,
        accounting for approximately <strong>50% of all lung cancer diagnoses</strong>.
        Its prevalence continues to rise, partly due to higher smoking rates among women and
        changes in cigarette composition over the past 50 years — causing deeper inhalation
        and greater carcinogen exposure in peripheral airways, where adenocarcinoma typically develops.
        Additional risk factors include radon gas, secondhand smoke, indoor pollutants,
        and environmental contamination <em>(Succony et al., 2021)</em>.
    </div>
    <div style="flex:1; text-align:center">
        <img src="{_b64(str(ASSETS / 'luad.jpg'))}"
             style="width:100%; border-radius:10px; object-fit:cover;" />
        <div style="font-size:0.75rem; color:#888; margin-top:6px">LUAD Histopathology Sample</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LUSC ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cancer-card" style="border-left-color:#e67e22;
     display:flex; gap:24px; align-items:center;">
    <div style="flex:3">
        <div class="cancer-title">2. 🟠 Squamous Cell Carcinoma (LUSC)</div>
        Lung Squamous Cell Carcinoma <strong>(LUSC)</strong> is the second most common NSCLC
        histological type, accounting for approximately <strong>20% of all lung cancer cases</strong>.
        It is strongly associated with smoking history. While traditionally known as a central tumor,
        peripheral LUSC now accounts for roughly one-third of all cases and is more prevalent
        in older patients and women. It may also be associated with fibrotic interstitial
        lung disease <em>(Berezowska et al., 2024)</em>.
    </div>
    <div style="flex:1; text-align:center">
        <img src="{_b64(str(ASSETS / 'lusc.jpg'))}"
             style="width:100%; border-radius:10px; object-fit:cover;" />
        <div style="font-size:0.75rem; color:#888; margin-top:6px">LUSC Histopathology Sample</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Benign ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cancer-card" style="border-left-color:#2ecc71;
     display:flex; gap:24px; align-items:center;">
    <div style="flex:3">
        <div class="cancer-title">3. 🟢 Benign Lung Tissue</div>
        Although LUAD and LUSC are among the most frequently encountered malignancies, it is equally
        important for pathologists to recognize <strong>benign lung tissue</strong> to avoid
        misclassifying morphologically similar non-cancerous structures as carcinoma.
        This model achieves near-perfect recall <strong>(100%)</strong> on this class,
        demonstrating reliable discrimination between benign and malignant tissue.
    </div>
    <div style="flex:1; text-align:center">
        <img src="{_b64(str(ASSETS / 'benign.jpg'))}"
             style="width:100%; border-radius:10px; object-fit:cover;" />
        <div style="font-size:0.75rem; color:#888; margin-top:6px">Benign Lung Tissue Sample</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── References ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### References")
st.markdown(
    """
    1. Joseph, J. & Rotty, L.W. (2020). Kanker paru: laporan kasus. *Medical Scope Journal*, 2(1).
    2. Succony, L., Rassl, D.M., Barker, A.P., McCaughan, F.M. & Rintoul, R.C. (2021).
       Adenocarcinoma spectrum lesions of the lung. *Cancer Treatment Reviews*, 99, 102237.
    3. Berezowska, S., Maillard, M., Keyter, M. & Bisig, B. (2024).
       Pulmonary squamous cell carcinoma and lymphoepithelial carcinoma. *Histopathology*, 84(1), 32–49.
    """
)

render_footer()
