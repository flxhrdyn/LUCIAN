"""Grad-CAM visualization utilities for the ConvNeXt-Base lung cancer model."""

import functools
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit threads

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import GRADCAM_LAYER


@functools.lru_cache(maxsize=1)
def _build_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return a Model that outputs (last_conv_feature_map, final_logits).

    Cached with lru_cache(maxsize=1) so the grad model is built only once
    per session — eliminates repeated tf.function retracing warnings.
    """
    base_output = model.get_layer(GRADCAM_LAYER).input  # 7×7×1024
    # model.outputs[0] guarantees a single tensor, not a list wrapper that
    # Keras 3 sometimes produces when using model.output as a graph node.
    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[base_output, model.outputs[0]],
    )


def compute_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for *class_idx*.

    Parameters
    ----------
    model     : loaded Keras model
    img_array : preprocessed image array, shape (1, 224, 224, 3), float32 in [0, 1]
    class_idx : index of the class to explain

    Returns
    -------
    heatmap : numpy array of shape (H, W), values in [0, 1]
    """
    grad_model = _build_grad_model(model)
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        results = grad_model(img_tensor, training=False)
        # Keras 3 may return a plain Python list instead of a tuple
        conv_outputs = results[0]
        predictions  = tf.cast(results[1], tf.float32)
        loss = predictions[:, class_idx]

    # Gradient of the class score w.r.t. the feature map  (1, H, W, C)
    grads = tape.gradient(loss, conv_outputs)
    # Global-average-pool gradients over spatial dims → importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_map = conv_outputs[0]  # (H, W, C)

    # Channel-weighted combination.
    # Using a matmul keeps this fast and works for both (7,7,C) and (224,224,C).
    raw_heatmap = tf.squeeze(conv_map @ pooled_grads[..., tf.newaxis])  # (H, W)

    # Standard Grad-CAM uses ReLU to keep only positive contributions.
    heatmap = tf.maximum(raw_heatmap, 0)

    # Some models/inputs (especially tiny untrained test models) can yield an all-zero
    # ReLU heatmap. In that case, fall back to a magnitude-based normalization so the
    # result is still informative and input-dependent.
    eps = tf.constant(1e-8, dtype=heatmap.dtype)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.cond(
        max_val > eps,
        lambda: heatmap / (max_val + eps),
        lambda: tf.abs(raw_heatmap) / (tf.reduce_max(tf.abs(raw_heatmap)) + eps),
    )
    return heatmap.numpy()


def overlay_heatmap(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a Grad-CAM heatmap over the original PIL image.

    Parameters
    ----------
    pil_image : original RGB PIL image (any size)
    heatmap   : (H, W) numpy array in [0, 1] from compute_gradcam()
    alpha     : heatmap opacity (0 = transparent, 1 = opaque)

    Returns
    -------
    PIL Image with the heatmap blended on top
    """
    orig = np.array(pil_image.convert("RGB"))
    h, w = orig.shape[:2]

    # Upscale heatmap to match the image resolution
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = (
        np.array(Image.fromarray(heatmap_uint8).resize((w, h), Image.Resampling.BILINEAR)) / 255.0
    )

    # Apply JET colormap (matplotlib) → (H, W, 3) uint8
    cmap    = plt.get_cmap("jet")
    colored = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    # Alpha blend: original + colored heatmap
    blended = (
        orig.astype(np.float32) * (1 - alpha)
        + colored.astype(np.float32) * alpha
    ).astype(np.uint8)

    return Image.fromarray(blended)
