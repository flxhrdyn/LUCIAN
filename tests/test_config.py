"""Unit tests for src/config.py — validate constants and paths."""
from pathlib import Path

import pytest

from src.config import (
    ASSETS_73_SPLIT,
    ASSETS_82_SPLIT,
    BASE_DIR,
    CLASS_COLORS,
    CLASS_LABELS_EN,
    CLASS_LABELS_ID,
    DEMO_IMAGES,
    GRADCAM_LAYER,
    IMAGE_SIZE,
    MODEL_PATH,
    MODEL_URL,
)

NUM_CLASSES = 3  # must match len(CLASS_LABELS_EN), CLASS_LABELS_ID, and CLASS_COLORS


class TestBasicConstants:
    def test_image_size_is_224(self):
        assert IMAGE_SIZE == (224, 224)

    def test_num_class_labels_en(self):
        assert len(CLASS_LABELS_EN) == NUM_CLASSES

    def test_num_class_labels_id(self):
        assert len(CLASS_LABELS_ID) == NUM_CLASSES

    def test_num_class_colors(self):
        assert len(CLASS_COLORS) == NUM_CLASSES

    def test_class_colors_are_hex(self):
        for color in CLASS_COLORS:
            assert color.startswith("#"), f"Color {color!r} is not a hex string"
            assert len(color) == 7, f"Color {color!r} is not a 6-digit hex"

    def test_gradcam_layer_nonempty(self):
        assert isinstance(GRADCAM_LAYER, str) and GRADCAM_LAYER


class TestPaths:
    def test_base_dir_exists(self):
        assert Path(BASE_DIR).is_dir()

    def test_model_path_ends_with_keras(self):
        assert MODEL_PATH.endswith(".keras")

    def test_model_url_is_huggingface(self):
        assert "huggingface.co" in MODEL_URL

    def test_assets_82_split_exists(self):
        assert Path(ASSETS_82_SPLIT).is_dir()

    def test_assets_73_split_exists(self):
        assert Path(ASSETS_73_SPLIT).is_dir()


class TestDemoImages:
    def test_demo_images_has_three_keys(self):
        assert len(DEMO_IMAGES) == NUM_CLASSES

    def test_demo_images_files_exist(self):
        for key, path in DEMO_IMAGES.items():
            assert Path(path).is_file(), f"Demo image missing: {key} → {path}"

    def test_demo_images_are_jpg(self):
        for path in DEMO_IMAGES.values():
            assert path.lower().endswith((".jpg", ".jpeg"))
