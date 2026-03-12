"""Streamlit entry point — registers all pages and sets the global page config.

Run with: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="LUCIAN · Lung Carcinoma Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.navigation (added in Streamlit 1.36) gives explicit control over page order,
# labels, and icons — unlike the older auto-discovery of the pages/ directory.

pg = st.navigation([
    st.Page("pages/Home.py",               title="Home",             icon="🏠"),
    st.Page("pages/1_Classification.py",   title="Classification",   icon="🔬"),
    st.Page("pages/2_Model_Performance.py",title="Model Performance", icon="📊"),
    st.Page("pages/3_Lung_Cancer_Info.py", title="Lung Cancer Info",  icon="🧬"),
])
pg.run()
