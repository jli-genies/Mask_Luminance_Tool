"""Segment a region out of a UV-space texture using a UV-space region mask.

Because the UV layout is fixed for a given character topology, one region mask (e.g. a
lips mask baked once via multiview_feature_bake.bake_feature_layer) can be resized and
reused against any texture that shares that topology — no per-texture landmark detection
needed. Pure pipeline logic (no Qt), mirroring the matte_luminance_blend.py /
matte_luminance_ui.py split elsewhere in this repo.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from matte_luminance_blend import composite_weights, load_rgb, resize_to


def extract_region(texture: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
    """Returns an RGBA uint8 array: ``texture``'s RGB untouched, alpha = the mask.

    ``mask_img`` is any image whose luminance encodes region coverage (0 = excluded, 255 =
    included) — the same "weight" convention used by mask channels elsewhere in this repo —
    and is resized (bilinear, to preserve feathered edges) to ``texture``'s resolution.
    """
    weights = resize_to(composite_weights(mask_img), texture.shape[:2], nearest=False)
    alpha = np.clip(weights * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([np.clip(texture, 0, 255).astype(np.uint8), alpha])


def save_rgba(path: str, rgba: np.ndarray) -> None:
    """Writes a uint8 RGBA image."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = np.clip(rgba, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise IOError(f"Failed to write: {path}")


def segment_texture(texture_path: str, mask_path: str, output_path: str) -> str:
    """Loads a texture and a UV-space region mask, extracts the region, writes an RGBA PNG.

    Returns ``output_path``.
    """
    texture = load_rgb(texture_path)
    mask_img = load_rgb(mask_path)
    rgba = extract_region(texture, mask_img)
    save_rgba(output_path, rgba)
    return output_path
