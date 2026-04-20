---
title: LUCIAN
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# LUCIAN (Docker Space)

This Space runs LUCIAN (Lung Carcinoma Histopathology Imaging & Analysis) as a single Docker container:

- Streamlit UI (public): `http://0.0.0.0:${PORT}` (Hugging Face sets `PORT=7860`)

## Required secrets / variables (Space Settings)

No mandatory secrets are required for standard execution, as the model and dataset are publicly accessible via the Hugging Face Hub.

Recommended for CI/CD:
- Secret: `HF_TOKEN` (Required if using the GitHub Actions workflow to automatically sync updates from your main repository)

## Notes

- **Self-Contained Pipeline**: The application automatically handles model acquisition. If the model file is not found in the local `models/` directory, it is pulled from the Hugging Face Hub on startup.
- **Optimized for CPU**: The container uses `tensorflow-cpu` to keep the image size manageable (~4 GB) and ensure compatibility with standard Space hardware.
- **Cold Starts**: Due to the size of the ConvNeXt-Base model (~500 MB), the initial model loading process during a cold start or first inference may take several seconds.
- **Health Monitoring**: A Docker health check is configured to monitor the Streamlit service status via `/_stcore/health`.
