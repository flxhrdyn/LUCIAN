"""Unit tests for src/model.py — preprocess_image and predict."""
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Skip entire module gracefully if TensorFlow is not installed.
pytest.importorskip("tensorflow", reason="TensorFlow not installed in this environment")

# Patch st.cache_resource before importing src.model so the decorator becomes a
# no-op — otherwise Streamlit tries to manage state and breaks in a test context.
with patch("streamlit.cache_resource", lambda **_: lambda fn: fn):
    from src.model import _MAX_FILE_SIZE_MB, preprocess_image, predict
from src.config import IMAGE_SIZE


# ---------------------------------------------------------------------------
# preprocess_image — happy path
# ---------------------------------------------------------------------------

class TestPreprocessImageValid:
    def test_returns_pil_image_and_array(self, fake_png_bytes):
        img, arr = preprocess_image(io.BytesIO(fake_png_bytes))
        assert isinstance(img, Image.Image)
        assert isinstance(arr, np.ndarray)

    def test_array_shape(self, fake_png_bytes):
        _, arr = preprocess_image(io.BytesIO(fake_png_bytes))
        assert arr.shape == (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    def test_array_dtype_float32(self, fake_png_bytes):
        _, arr = preprocess_image(io.BytesIO(fake_png_bytes))
        assert arr.dtype == np.float32

    def test_pixel_values_in_0_1(self, fake_png_bytes):
        _, arr = preprocess_image(io.BytesIO(fake_png_bytes))
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_image_resized_to_224(self, fake_png_bytes):
        # Input is 10×10, output PIL image should be 224×224.
        img, _ = preprocess_image(io.BytesIO(fake_png_bytes))
        assert img.size == IMAGE_SIZE

    def test_accepts_jpeg(self, fake_jpeg_bytes):
        img, arr = preprocess_image(io.BytesIO(fake_jpeg_bytes))
        assert img.size == IMAGE_SIZE

    def test_image_mode_is_rgb(self, fake_png_bytes):
        img, _ = preprocess_image(io.BytesIO(fake_png_bytes))
        assert img.mode == "RGB"

    def test_large_image_resizes_correctly(self):
        buf = io.BytesIO()
        Image.new("RGB", (1024, 1024), color=(0, 0, 0)).save(buf, format="PNG")
        img, arr = preprocess_image(io.BytesIO(buf.getvalue()))
        assert img.size == IMAGE_SIZE
        assert arr.shape == (1, 224, 224, 3)


# ---------------------------------------------------------------------------
# preprocess_image — error cases
# ---------------------------------------------------------------------------

class TestPreprocessImageErrors:
    def test_raises_on_corrupt_bytes(self):
        corrupt = io.BytesIO(b"this is not an image")
        with pytest.raises(ValueError, match="bukan gambar yang valid"):
            preprocess_image(corrupt)

    def test_raises_on_empty_bytes(self):
        with pytest.raises(ValueError, match="bukan gambar yang valid"):
            preprocess_image(io.BytesIO(b""))

    def test_raises_on_oversized_file(self):
        # Build in-memory bytes just over the limit.
        limit_bytes = int(_MAX_FILE_SIZE_MB * 1e6) + 1
        oversized = io.BytesIO(b"\x00" * limit_bytes)
        with pytest.raises(ValueError, match="terlalu besar"):
            preprocess_image(oversized)

    def test_exact_limit_does_not_raise(self, fake_png_bytes):
        # A real image well within the limit must not raise.
        assert len(fake_png_bytes) < _MAX_FILE_SIZE_MB * 1e6
        img, arr = preprocess_image(io.BytesIO(fake_png_bytes))
        assert arr is not None


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture()
    def mock_model(self):
        model = MagicMock()
        # Simulate 3-class softmax output
        model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        return model

    def test_returns_tuple(self, mock_model, fake_img_array):
        result = predict(mock_model, fake_img_array)
        assert isinstance(result, tuple) and len(result) == 2

    def test_probs_is_ndarray(self, mock_model, fake_img_array):
        probs, _ = predict(mock_model, fake_img_array)
        assert isinstance(probs, np.ndarray)

    def test_probs_has_three_classes(self, mock_model, fake_img_array):
        probs, _ = predict(mock_model, fake_img_array)
        assert probs.shape == (3,)

    def test_probs_sum_to_one(self, mock_model, fake_img_array):
        probs, _ = predict(mock_model, fake_img_array)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_elapsed_time_positive(self, mock_model, fake_img_array):
        _, elapsed = predict(mock_model, fake_img_array)
        assert elapsed >= 0.0

    def test_model_predict_called_once(self, mock_model, fake_img_array):
        predict(mock_model, fake_img_array)
        mock_model.predict.assert_called_once_with(fake_img_array, verbose=0)
