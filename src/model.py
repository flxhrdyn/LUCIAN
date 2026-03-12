import io
import logging
import os
import time

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.config import IMAGE_SIZE, MODEL_PATH, MODEL_URL

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE_MB = 10


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the ConvNeXt model, downloading from HuggingFace if not cached locally."""
    # size == 0 catches a previous aborted download that left a 0-byte file
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        logger.info("Model not found locally — downloading from HuggingFace Hub.")
        _download_model()
    start = time.time()
    logger.info("Loading model from %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    elapsed = time.time() - start
    logger.info("Model loaded in %.2fs", elapsed)
    return model, elapsed


def _download_model() -> None:
    """Stream-download the model file from HuggingFace Hub.

    Uses a temp file + atomic rename so a failed/interrupted download never
    leaves a corrupt file at MODEL_PATH.
    """
    tmp_path = MODEL_PATH + ".tmp"
    logger.info("Downloading model from %s", MODEL_URL)
    try:
        # 120s timeout — the model is ~420 MB, slow connections need the headroom
        with requests.get(MODEL_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):  # 8 KB keeps memory flat
                    f.write(chunk)
                    downloaded += len(chunk)
            if total:
                logger.info("Downloaded %.1f MB", downloaded / 1e6)
        os.replace(tmp_path, MODEL_PATH)
        logger.info("Model saved to %s", MODEL_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.exception("Model download failed — temp file cleaned up.")
        raise


def preprocess_image(uploaded_file):
    """Load and preprocess an uploaded image for model inference.

    Raises
    ------
    ValueError : if the file exceeds the size limit or is not a valid image.
    """
    raw = uploaded_file.read()
    size_mb = len(raw) / 1e6
    if size_mb > _MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File terlalu besar ({size_mb:.1f} MB). Maksimum {_MAX_FILE_SIZE_MB} MB."
        )
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMAGE_SIZE)
    except UnidentifiedImageError as exc:
        raise ValueError("File yang diunggah bukan gambar yang valid.") from exc

    # scale to [0, 1] floats, add batch axis → shape (1, 224, 224, 3)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
    return img, arr


def predict(model, img_array):
    """Run inference and return class probabilities with elapsed time."""
    start = time.time()
    probs = model.predict(img_array, verbose=0)[0]
    return probs, time.time() - start
