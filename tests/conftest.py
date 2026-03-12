"""Shared pytest fixtures for all test modules."""
import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def fake_png_bytes() -> bytes:
    """Minimal valid 10×10 white PNG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def fake_jpeg_bytes() -> bytes:
    """Minimal valid 32×32 JPEG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def fake_img_array() -> np.ndarray:
    """Preprocessed image array matching model input: (1, 224, 224, 3) float32."""
    return np.zeros((1, 224, 224, 3), dtype=np.float32)


@pytest.fixture()
def fake_pil_image() -> Image.Image:
    """224×224 RGB PIL image."""
    return Image.new("RGB", (224, 224), color=(128, 64, 32))
