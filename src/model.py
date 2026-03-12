import io
import logging
import os
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from src.config import HF_FILENAME, HF_REPO_ID, IMAGE_SIZE, MODEL_PATH

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
    """Download the model from HuggingFace Hub using the official SDK.

    hf_hub_download handles resumable downloads, caching, and progress
    reporting automatically — replacing the manual requests.get() approach.
    local_dir pins the file to the project folder instead of ~/.cache/huggingface,
    so the app finds it at MODEL_PATH on all platforms including Docker.
    """
    logger.info("Downloading %s from repo %s", HF_FILENAME, HF_REPO_ID)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=os.path.dirname(MODEL_PATH),
    )
    logger.info("Model saved to %s", MODEL_PATH)


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
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    probs = model(img_tensor, training=False).numpy()[0]
    return probs, time.time() - start
