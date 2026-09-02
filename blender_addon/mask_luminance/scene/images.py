"""bpy.types.Image <-> numpy array bridge.

This is the addon's "scene" layer in the HeadGen sense: it's the only place
that touches a live Blender datablock. Everything it produces or consumes is
a plain ``(H, W, 3)`` uint8 RGB array in top-down row order — the exact same
contract ``core.blend.load_rgb``/``save_rgb`` use for file paths — so
``core.blend``/``core.infill`` never need to know whether their input came
from disk or from an Image datablock.

Two things this had to get right, both verified empirically against a real
``bpy.types.Image`` rather than assumed:

* ``Image.pixels`` is stored bottom-up; every array here is top-down, so both
  directions flip vertically.
* ``Image.pixels`` returns literal stored byte values scaled to 0-1 (matching
  ``cv2.imread``'s raw bytes exactly) regardless of ``colorspace_settings`` —
  there is no implicit gamma decode to undo. ``colorspace_settings`` is still
  set explicitly below (``Non-Color`` for masks) as a matter of Blender
  convention for data textures, not because it changes the math.
"""

from __future__ import annotations

from typing import Optional

import bpy
import numpy as np


def image_to_rgb(image: bpy.types.Image) -> np.ndarray:
    """Reads an Image datablock's pixels into a top-down uint8 (H, W, 3) RGB array.

    Drops alpha, matching ``core.blend.load_rgb``'s contract for a file path.
    """
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError(f"Image '{image.name}' has no pixel data (0x0) — is it packed/loaded?")

    channels = image.channels
    flat = np.empty(width * height * channels, dtype=np.float32)
    image.pixels.foreach_get(flat)

    arr = flat.reshape(height, width, channels)
    arr = np.flipud(arr)  # bottom-up (Blender) -> top-down

    rgb_float = arr[..., :3] if channels >= 3 else np.repeat(arr[..., :1], 3, axis=-1)
    return np.clip(rgb_float * 255.0, 0, 255).round().astype(np.uint8)


def rgb_to_image(
    rgb: np.ndarray,
    name: str,
    existing: Optional[bpy.types.Image] = None,
    non_color: bool = False,
) -> bpy.types.Image:
    """Writes a top-down uint8 (H, W, 3) RGB array into an Image datablock.

    Reuses ``existing`` in place when its resolution already matches (so
    re-baking updates the same datablock instead of accumulating a new one
    per bake); otherwise creates a new one named ``name``.
    """
    height, width = rgb.shape[:2]

    image = existing
    if image is None or tuple(image.size) != (width, height):
        image = bpy.data.images.new(name, width=width, height=height, alpha=True)
        image.name = name

    # Must happen before foreach_set below, not after: reassigning
    # colorspace_settings.name once pixel data already exists in the buffer
    # triggers Blender to re-interpret (i.e. corrupt) that existing data
    # under the new colorspace, rather than just re-tagging it — confirmed
    # empirically, this silently produced garbage on ~70% of rows on a
    # real 2048x2048 write when the assignment came after foreach_set.
    image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"

    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = rgb.astype(np.float32) / 255.0
    rgba[..., 3] = 1.0
    rgba = np.flipud(rgba)  # top-down -> bottom-up (Blender)

    image.pixels.foreach_set(rgba.ravel())
    image.update()
    return image


def result_image_name(source: bpy.types.Image, suffix: str = "_matte") -> str:
    """Deterministic output-image name derived from the source texture's name.

    Used with ``rgb_to_image(..., existing=bpy.data.images.get(name))`` so
    repeated bakes update the same result datablock.
    """
    return f"{source.name}{suffix}"
