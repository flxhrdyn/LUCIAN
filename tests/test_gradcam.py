"""Unit tests for src/gradcam.py — overlay_heatmap and compute_gradcam."""
import numpy as np
import pytest
from PIL import Image

# Skip entire module gracefully if TensorFlow is not installed.
tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed in this environment")

from src.gradcam import compute_gradcam, overlay_heatmap


# ---------------------------------------------------------------------------
# Tiny model fixture — mirrors the real model's layer structure but is tiny
# so tests run in milliseconds without loading the full ConvNeXt weights.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model() -> tf.keras.Model:
    """Small CNN with a 'flatten' layer matching GRADCAM_LAYER in config."""
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", name="conv_block")(inputs)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


@pytest.fixture()
def fake_heatmap_small() -> np.ndarray:
    """7×7 heatmap with values in [0, 1], matching ConvNeXt's spatial output."""
    rng = np.random.default_rng(42)
    return rng.random((7, 7)).astype(np.float32)


@pytest.fixture()
def fake_heatmap_large() -> np.ndarray:
    """224×224 heatmap (upscaled scenario)."""
    rng = np.random.default_rng(0)
    return rng.random((224, 224)).astype(np.float32)


# ---------------------------------------------------------------------------
# overlay_heatmap
# ---------------------------------------------------------------------------

class TestOverlayHeatmap:
    def test_returns_pil_image(self, fake_pil_image, fake_heatmap_small):
        result = overlay_heatmap(fake_pil_image, fake_heatmap_small)
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self, fake_pil_image, fake_heatmap_small):
        result = overlay_heatmap(fake_pil_image, fake_heatmap_small)
        assert result.mode == "RGB"

    def test_output_size_matches_input(self, fake_pil_image, fake_heatmap_small):
        result = overlay_heatmap(fake_pil_image, fake_heatmap_small)
        assert result.size == fake_pil_image.size

    def test_large_heatmap_matches_input_size(self, fake_pil_image, fake_heatmap_large):
        result = overlay_heatmap(fake_pil_image, fake_heatmap_large)
        assert result.size == fake_pil_image.size

    def test_pixel_values_in_valid_uint8_range(self, fake_pil_image, fake_heatmap_small):
        import numpy as np
        result = overlay_heatmap(fake_pil_image, fake_heatmap_small)
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_custom_alpha_accepted(self, fake_pil_image, fake_heatmap_small):
        result = overlay_heatmap(fake_pil_image, fake_heatmap_small, alpha=0.0)
        assert isinstance(result, Image.Image)

    def test_non_square_image(self, fake_heatmap_small):
        landscape = Image.new("RGB", (320, 240), color=(10, 20, 30))
        result = overlay_heatmap(landscape, fake_heatmap_small)
        assert result.size == (320, 240)

    def test_zero_heatmap_does_not_crash(self, fake_pil_image):
        zero_heatmap = np.zeros((7, 7), dtype=np.float32)
        result = overlay_heatmap(fake_pil_image, zero_heatmap)
        assert isinstance(result, Image.Image)

    def test_ones_heatmap_does_not_crash(self, fake_pil_image):
        ones_heatmap = np.ones((7, 7), dtype=np.float32)
        result = overlay_heatmap(fake_pil_image, ones_heatmap)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# compute_gradcam
# ---------------------------------------------------------------------------

class TestComputeGradcam:
    @pytest.fixture()
    def sample_input(self) -> np.ndarray:
        """Random (1, 224, 224, 3) float32 array."""
        rng = np.random.default_rng(7)
        return rng.random((1, 224, 224, 3)).astype(np.float32)

    def test_returns_ndarray(self, tiny_model, sample_input):
        heatmap = compute_gradcam(tiny_model, sample_input, class_idx=0)
        assert isinstance(heatmap, np.ndarray)

    def test_heatmap_is_2d(self, tiny_model, sample_input):
        heatmap = compute_gradcam(tiny_model, sample_input, class_idx=0)
        assert heatmap.ndim == 2

    def test_heatmap_values_in_0_1(self, tiny_model, sample_input):
        heatmap = compute_gradcam(tiny_model, sample_input, class_idx=0)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6  # small float tolerance

    def test_all_class_indices_work(self, tiny_model, sample_input):
        for class_idx in range(3):
            heatmap = compute_gradcam(tiny_model, sample_input, class_idx=class_idx)
            assert heatmap.ndim == 2

    def test_different_inputs_give_different_heatmaps(self, tiny_model):
        rng = np.random.default_rng(99)
        a = rng.random((1, 224, 224, 3)).astype(np.float32)
        b = rng.random((1, 224, 224, 3)).astype(np.float32)
        heatmap_a = compute_gradcam(tiny_model, a, class_idx=0)
        heatmap_b = compute_gradcam(tiny_model, b, class_idx=0)
        # Two different images should not produce identical heatmaps
        assert not np.allclose(heatmap_a, heatmap_b)
