"""CSS injection and shared UI components (sidebar info, footer) for all pages.

All CSS lives in the _CSS string so edits stay in one place instead of being
scattered across individual page files.
"""
import streamlit as st

_CSS = """
<style>
/* ── Layout ─────────────────────────────────────────────── */
.block-container { padding-top: 4rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(46, 204, 113, 0.2);
    background: linear-gradient(180deg, #0d1117 0%, #1a1a2e 50%, #16213e 100%);
}

/* ── Sidebar: brand above nav ────────────────────────────── */
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar brand header ────────────────────────────────── */
.sidebar-brand {
    padding: 20px 16px 14px;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 4px;
}
.sidebar-brand-icon { font-size: 1.8rem; display: block; margin-bottom: 2px; }
.sidebar-brand-name {
    font-size: 1.45rem; font-weight: 900; letter-spacing: 0.18em;
    background: linear-gradient(135deg, #2ecc71 0%, #1abc9c 60%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: block; line-height: 1.1;
}
.sidebar-brand-tagline {
    font-size: 0.58rem; color: rgba(46,204,113,0.6);
    letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 600; margin-top: 5px; display: block; line-height: 1.6;
}
.sidebar-meta {
    font-size: 0.76rem; color: #888; line-height: 2;
    padding: 10px 16px 4px;
    border-top: 1px solid #1e1e2e;
    margin-top: 6px;
}

/* ── Streamlit metric card override ─────────────────────── */
div[data-testid="metric-container"] {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #262730;
}

/* ── Logo ───────────────────────────────────────────────── */
.logo-container { text-align: center; margin-bottom: 20px; }
.logo-icon { font-size: 3.8rem; display: block; margin-bottom: 2px; }
.logo-name {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #2ecc71 0%, #1abc9c 40%, #27ae60 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.18em;
    line-height: 1;
    display: block;
}
.logo-tagline {
    font-size: 0.72rem;
    color: rgba(46, 204, 113, 0.7);
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 10px;
    display: block;
}

/* ── Hero Banner ─────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid rgba(46, 204, 113, 0.3);
    border-radius: 16px;
    padding: 52px 40px;
    text-align: center;
    margin-bottom: 32px;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: #fafafa;
    margin: 0 0 12px 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #999;
    margin-bottom: 28px;
    line-height: 1.6;
}
.stat-pill {
    display: inline-block;
    background: rgba(46, 204, 113, 0.12);
    border: 1px solid rgba(46, 204, 113, 0.4);
    border-radius: 999px;
    padding: 6px 18px;
    margin: 4px;
    font-size: 0.88rem;
    color: #2ecc71;
    font-weight: 600;
}

/* ── Feature Cards ───────────────────────────────────────── */
.feature-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 24px 20px;
    border: 1px solid rgba(46, 204, 113, 0.18);
    margin: 4px 0;
    height: 100%;
}
.feature-card h3 {
    margin: 10px 0 8px 0;
    font-size: 1.05rem;
    color: #fafafa;
}
.feature-card p, .feature-card ul {
    color: #999;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.65;
}
.feature-icon { font-size: 2rem; }

/* ── Tech Badges ─────────────────────────────────────────── */
.badge-row { margin: 8px 0 4px 0; }
.badge-label {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    vertical-align: middle;
    margin-right: 6px;
}
.tech-badge {
    display: inline-block;
    background: #262730;
    border: 1px solid #363640;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px;
    font-size: 0.78rem;
    color: #bbb;
    font-weight: 500;
}

/* ── Section Badge ───────────────────────────────────────── */
.section-badge {
    display: inline-block;
    background: rgba(46, 204, 113, 0.12);
    border: 1px solid rgba(46, 204, 113, 0.4);
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.7rem;
    color: #2ecc71;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* ── Prediction Card ─────────────────────────────────────── */
.prediction-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 28px 28px 24px;
    border-left: 5px solid;
    margin: 4px 0;
}
.prediction-label {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    line-height: 1.2;
}
.prediction-sub {
    color: #777;
    font-size: 0.88rem;
    margin-bottom: 24px;
    font-style: italic;
}
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 4px; }
.stat-item {
    background: #0e1117;
    border-radius: 8px;
    padding: 12px 20px;
    min-width: 110px;
}
.stat-item .s-label {
    font-size: 0.68rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 4px;
}
.stat-item .s-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #fafafa;
}

/* ── Disclaimer ──────────────────────────────────────────── */
.disclaimer {
    background: rgba(231, 76, 60, 0.07);
    border: 1px solid rgba(231, 76, 60, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    color: rgba(231, 76, 60, 0.9);
    font-size: 0.82rem;
    margin-top: 24px;
    line-height: 1.6;
}

/* ── Cancer Info Cards ───────────────────────────────────── */
.cancer-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 24px 28px;
    border-left: 4px solid;
    margin: 16px 0;
    line-height: 1.75;
    color: #bbb;
}
.cancer-card strong { color: #fafafa; }
.cancer-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #fafafa;
    margin-bottom: 12px;
}

/* ── Footer ──────────────────────────────────────────────── */
.footer {
    text-align: center;
    color: #444;
    font-size: 0.8rem;
    padding: 20px 0 4px 0;
    border-top: 1px solid #1e1e2e;
    margin-top: 48px;
    line-height: 2;
}
.footer a { color: #2ecc71; text-decoration: none; }
</style>
"""

_FOOTER = """
<div class="footer">
    🔬 <strong style="color:#2ecc71; letter-spacing:0.1em">LUCIAN</strong>
    <span style="color:#444"> &nbsp;·&nbsp; Lung Carcinoma Histopathology Imaging & Analysis</span><br>
    Undergraduate Thesis &nbsp;·&nbsp; Felix Hardyan &nbsp;·&nbsp; Computer Science 2025 &nbsp;·&nbsp;
    <a href="https://huggingface.co/felixhrdyn/convnextv1-lung-cancer" target="_blank">
        🤗 HuggingFace Model
    </a>
</div>
"""


def apply_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def render_footer():
    st.markdown(_FOOTER, unsafe_allow_html=True)


def render_sidebar_info():
    with st.sidebar:
        # ── Brand header
        st.markdown(
            """
            <div class="sidebar-brand">
                <span class="sidebar-brand-icon">🔬</span>
                <span class="sidebar-brand-name">LUCIAN</span>
                <span class="sidebar-brand-tagline">Lung Carcinoma Histopathology<br>Imaging &amp; Analysis</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # ── Navigation
        st.page_link("pages/Home.py",                label="Home",             icon="🏠")
        st.page_link("pages/1_Classification.py",    label="Classification",   icon="🔬")
        st.page_link("pages/2_Model_Performance.py", label="Model Performance", icon="📊")
        st.page_link("pages/3_Lung_Cancer_Info.py",  label="Lung Cancer Info",  icon="🧬")
        # ── Model info
        st.markdown(
            """
            <div class="sidebar-meta">
                ConvNeXt-Base &nbsp;·&nbsp; Transfer Learning<br>
                Test Accuracy: <strong style='color:#2ecc71'>93.67%</strong><br>
                3 Classes &nbsp;·&nbsp; LC25000 Dataset<br><br>
                <a href='https://huggingface.co/felixhrdyn/convnextv1-lung-cancer'
                   style='color:#2ecc71; text-decoration:none'>
                   🤗 HuggingFace Model
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
