# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# libgomp1  – required by TensorFlow (OpenMP multi-threading)
# curl      – used by the HEALTHCHECK probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer as long as requirements.txt unchanged)
# tensorflow-cpu is used instead of tensorflow — no CUDA libs, ~1 GB smaller image,
# and sufficient for inference on CPU-only hosts (Streamlit Cloud, HF Spaces, App Service B1).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 7860

# Liveness probe — Streamlit exposes a health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
