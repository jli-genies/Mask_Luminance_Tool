from typing import Optional, Union

import numpy as np

from genies.meshutils.shading.texture_utils import (
    apply_extrapolation_blur,
    extend_texture_boundaries,
    load_image_as_numpy,
    save_numpy_as_image,
)

# Blur radius relative to the texture height
SOFTNESS_FACTOR = 0.005



def infill_from_mask(
    image: Union[str, np.ndarray],
    mask: Union[str, np.ndarray],
    softness_px: Optional[float] = None,
    output_path: Optional[str] = None
) -> np.ndarray:
    """Fills the masked region of an image with the colors that surround it.

    Every pixel the mask paints is barred from donating and repainted with the
    nearest surrounding colors, then blurred so the propagation ridge dissolves.
    The mask floor is read off its own histogram, so a feathered or noisy mask
    needs no threshold. Both inputs are accepted either as arrays or as paths.

    Args:
        image (Union[str, np.ndarray]): The source image or the path to read it from.
        mask (Union[str, np.ndarray]): The mask or its path, anything above its
            background level marking the region to repaint.
        softness_px (Optional[float]): Blur radius dissolving the propagation
            ridge. Defaults to a fraction of the texture height.
        output_path (Optional[str]): Destination file, when the result must be saved.

    Returns:
        np.ndarray: The infilled image as a uint8 array.
    """
    # 1. Resolve both inputs, whichever side of the memory boundary they sit on.
    img = image if isinstance(image, np.ndarray) else load_image_as_numpy(image)
    raw_mask = mask if isinstance(mask, np.ndarray) else load_image_as_numpy(mask)

    # 2. Promote a flat buffer to the 3D layout the blur pass expects.
    if img.ndim == 2:
        img = img[..., np.newaxis]

    # 3. Reduce the mask to a single 8-bit plane.
    plane = (raw_mask[..., 0] if raw_mask.ndim == 3 else raw_mask).astype(np.float32)
    if plane.max() <= 1.0:
        plane *= 255.0
    plane = plane.astype(np.uint8)

    # 4. Read the background level off the histogram rather than guessing a threshold.
    background = int(np.argmax(np.bincount(plane.ravel(), minlength=256)))
    holes = plane > background

    # 5. Push the nearest surrounding colors across the whole painted footprint.
    filled = extend_texture_boundaries(img, holes, max_distance=None)

    # 6. Dissolve the ridge the propagation leaves along the medial axis of a hole.
    radius = SOFTNESS_FACTOR * img.shape[0] if softness_px is None else softness_px
    if radius > 0.0:
        filled = apply_extrapolation_blur(filled, holes, radius)

    # 7. Commit the result to disk when a destination is requested.
    if output_path:
        save_numpy_as_image(filled, output_path)

    return filled