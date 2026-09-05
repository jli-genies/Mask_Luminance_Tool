"""Global exposure/gamma grading, applied once to a finished bake result.

Reimplements — in pure numpy, no ``bpy``/``genies`` import — the documented
behavior of the standalone repo's ``texture_edit.apply_exposure_gamma``
(itself a thin wrapper around the internal ``genies.meshutils`` package's
color-correction routine). The addon can't import ``genies`` at all (it isn't
on Blender's bundled Python, and ``core/`` is deliberately numpy/opencv-only —
see ``core/infill.py``), so this is a fresh implementation matching that
function's documented contract rather than a vendor of its source.
"""

from __future__ import annotations

import numpy as np

# Approximate sRGB encode/decode curve. Texture pixels here are always the
# gamma-encoded 0-255 values a bpy.types.Image stores (see scene/images.py),
# not scene-linear light, so exposure — a linear-light operation — has to
# round-trip through this curve rather than multiplying the encoded value
# directly, which would overshoot in the highlights.
_ENCODE_GAMMA = 2.2


def apply_exposure_gamma(
    image: np.ndarray,
    exposure: float = 0.0,
    gamma: float = 1.0,
    shadow_bias: float = 0.0,
) -> np.ndarray:
    """Applies exposure and gamma to a (H, W, 3) uint8 RGB array.

    Exposure is applied in linear light (decode -> scale by 2**exposure ->
    re-encode), like a camera or Photoshop's Exposure tool — multiplying the
    encoded value directly would overshoot since the encoding curve is
    itself non-linear. Gamma is a plain power curve on the encoded signal.
    ``shadow_bias`` blends both toward their full effect in black and toward
    no effect in white (weighted by the pixel's own Rec. 709 luminance), so a
    shadow lift doesn't blow out highlights that were already fine.

    Args:
        image: (H, W, 3) uint8 RGB array.
        exposure: Stops (0.0 is neutral; each +/-1.0 doubles/halves the
            linear-light signal).
        gamma: Power curve on the encoded signal (1.0 is neutral; >1.0 lifts
            midtones, <1.0 crushes them). Non-positive values are clamped.
        shadow_bias: 0-1, weights exposure and gamma by the pixel's own
            luminance -- 0.0 applies them uniformly (default); 1.0 applies
            their full effect in black, tapering to none in white.

    Returns:
        np.ndarray: The corrected image as a uint8 array.
    """
    if exposure == 0.0 and gamma == 1.0:
        return image

    encoded = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)

    linear = np.power(encoded, _ENCODE_GAMMA) * (2.0 ** exposure)
    exposed = np.power(np.clip(linear, 0.0, None), 1.0 / _ENCODE_GAMMA)

    safe_gamma = max(gamma, 1e-6)
    corrected = np.power(np.clip(exposed, 0.0, None), 1.0 / safe_gamma)

    shadow_bias = float(np.clip(shadow_bias, 0.0, 1.0))
    if shadow_bias > 0.0:
        lum = (
            0.2126 * encoded[..., 0]
            + 0.7152 * encoded[..., 1]
            + 0.0722 * encoded[..., 2]
        )
        weight = (1.0 - shadow_bias * lum)[..., None]
        corrected = encoded * (1.0 - weight) + corrected * weight

    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
