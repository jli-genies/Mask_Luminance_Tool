"""Build a luminance-delta blend mask and optionally correct a sample albedo.

Compares a sample UV texture against a flat diffuse reference (UV map or
palette image), marks pixels whose luminance differs by more than a threshold,
softens that mask with a blending radius, then softens the sample via local
blur inside the mask. Diffuse is a comparison reference only by default
(optional ``concept_diffuse_mix`` can blend toward it).

Typical use (UV diffuse of the same layout)::

    python matte_luminance_blend.py \\
        --texture african_female_0003_albedo_from_concept.png \\
        --diffuse  african_female_0003_flat_diffuse.png \\
        --region-mask mask_concept_texture.png \\
        --threshold 12 --radius 8 --strength 0.85 \\
        --out-mask  out/blend_mask.png \\
        --out-texture out/albedo_matte.png

Palette mode (multiview flat render; samples mean non-background skin)::

    python matte_luminance_blend.py \\
        --texture albedo.png \\
        --diffuse Cleaningtexture_005.png \\
        --diffuse-mode palette \\
        --region-mask mask_concept_texture.png \\
        --out-mask blend_mask.png --out-texture albedo_matte.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from genies.meshutils.shading.texture_utils import (
    apply_extrapolation_blur,
    extend_texture_boundaries,
)

logger = logging.getLogger(__name__)

# Default ID colors from the authored region mask (quantized / approximate).
DEFAULT_REGION_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "forehead": (200, 16, 120),   # magenta
    "jaw_cheeks": (232, 232, 232),  # white / light grey
    "back_head": (56, 200, 248),  # cyan
}


# =============================================================================
# I/O
# =============================================================================
def load_rgb(path: str) -> np.ndarray:
    """Loads an image as uint8 RGB (drops alpha if present)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: str, rgb: np.ndarray) -> None:
    """Writes an RGB or single-channel uint8 image."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = np.clip(rgb, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        ok = cv2.imwrite(path, arr)
    else:
        ok = cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Failed to write: {path}")


def resize_to(img: np.ndarray, size_hw: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Resizes to (height, width)."""
    h, w = size_hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


# =============================================================================
# COLOR / LUMINANCE
# =============================================================================
def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance, float32."""
    rgb = rgb.astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def rgb_to_ycbcr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits RGB into Y, Cb, Cr (float32), Rec. 709 throughout.

    Cb/Cr scale factors are ``1 / (2*(1-Kb))`` and ``1 / (2*(1-Kr))`` for the
    same Rec. 709 Kb/Kr used by ``luminance()``. Using BT.601's constants
    here (0.564/0.713) while ``luminance()`` is BT.709 made the encode/decode
    round trip lossy — R and B happened to nearly cancel out, but G did not,
    so every "luminance-only" blend silently pushed a magenta/green cast into
    the result.
    """
    rgb = rgb.astype(np.float32)
    y = luminance(rgb)
    cb = (rgb[..., 2] - y) * 0.5389 + 128.0
    cr = (rgb[..., 0] - y) * 0.6350 + 128.0
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Reconstructs RGB from Y, Cb, Cr (Rec. 709; inverse of ``rgb_to_ycbcr``)."""
    r = y + 1.5748 * (cr - 128.0)
    g = y - 0.1873 * (cb - 128.0) - 0.4681 * (cr - 128.0)
    b = y + 1.8556 * (cb - 128.0)
    return np.stack([r, g, b], axis=-1)


# =============================================================================
# REGION / DIFFUSE TARGETS
# =============================================================================
def color_id_mask(
    id_map: np.ndarray,
    rgb: Sequence[int],
    tolerance: int = 40,
) -> np.ndarray:
    """Binary float mask for pixels matching an ID color within tolerance."""
    diff = np.abs(id_map.astype(np.int16) - np.asarray(rgb, dtype=np.int16))
    return (diff.max(axis=-1) <= tolerance).astype(np.float32)


def build_region_gate(
    id_map: Optional[np.ndarray],
    palette: Dict[str, Tuple[int, int, int]],
    tolerance: int = 40,
    active: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Union of selected ID regions. All-ones if no id_map is provided."""
    if id_map is None:
        return np.ones((), dtype=np.float32)  # broadcast later

    gate = np.zeros(id_map.shape[:2], dtype=np.float32)
    names = list(active) if active else list(palette.keys())
    for name in names:
        if name not in palette:
            raise KeyError(f"Unknown region '{name}'. Known: {sorted(palette)}")
        gate = np.maximum(gate, color_id_mask(id_map, palette[name], tolerance))
    return gate


def sample_palette_skin_rgb(
    palette_img: np.ndarray,
    bg_threshold: int = 245,
    bg_tolerance: int = 24,
) -> np.ndarray:
    """Estimates a flat skin RGB from a multiview / palette render.

    The studio background is not assumed to be white: it is sampled from the
    image corners (flat studio backdrops, whatever their color, sit there)
    and excluded by color distance. A gray/colored backdrop that a plain
    near-white check would miss previously got averaged in as "skin" and
    dragged the estimate toward the backdrop color. Near-black pixels
    (line art / deep shadow) and near-white blowouts are still dropped too.
    """
    corner_px = np.concatenate([
        palette_img[:8, :8].reshape(-1, 3),
        palette_img[:8, -8:].reshape(-1, 3),
        palette_img[-8:, :8].reshape(-1, 3),
        palette_img[-8:, -8:].reshape(-1, 3),
    ]).astype(np.float32)
    bg_color = np.median(corner_px, axis=0)

    dist_to_bg = np.abs(palette_img.astype(np.float32) - bg_color).max(axis=-1)
    lum = luminance(palette_img)
    keep = (dist_to_bg > bg_tolerance) & (lum > 15.0) & (lum < bg_threshold)
    if not np.any(keep):
        raise ValueError("Could not sample skin from diffuse palette (no non-background pixels).")
    skin = palette_img[keep].astype(np.float32)
    mean = skin.mean(axis=0)
    logger.info(
        "Detected palette background ≈ (%.0f, %.0f, %.0f); skin RGB ≈ (%.1f, %.1f, %.1f) from %d pixels",
        bg_color[0], bg_color[1], bg_color[2], mean[0], mean[1], mean[2], int(keep.sum()),
    )
    return mean


def make_diffuse_target(
    sample: np.ndarray,
    diffuse: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Builds a per-pixel diffuse target in the sample's resolution.

    Modes:
        uv      – ``diffuse`` is a UV map; resized to sample size.
        palette – ``diffuse`` is a flat multiview/palette image; fill UV with
                  sampled mean skin color (keeps sample chroma via Y swap later).

    ``self`` mode (deriving the target from the sample itself rather than an
    external diffuse asset) is handled by ``sample_self_reference_skin_rgb``
    in ``process()``, since it needs the concept/highlight/feature masks that
    aren't available here.
    """
    if mode == "uv":
        return resize_to(diffuse, sample.shape[:2], nearest=False).astype(np.float32)

    if mode == "palette":
        skin = sample_palette_skin_rgb(diffuse)
        target = np.empty(sample.shape, dtype=np.float32)
        target[...] = skin
        return target

    raise ValueError(f"Unknown diffuse mode: {mode}")


def sample_self_reference_skin_rgb(
    sample: np.ndarray,
    exclude_gate: np.ndarray,
    envelope: Optional[np.ndarray] = None,
    bg_threshold: float = 8.0,
) -> np.ndarray:
    """Estimates a flat skin RGB from the sample texture's own "clean" pixels.

    Averages sample pixels that are inside ``envelope`` (if given) but
    outside ``exclude_gate`` — typically the union of the concept ID
    regions, the highlight paint, and the feature-preserve mask, i.e.
    whatever has already been flagged as needing correction or as
    non-skin (eyes/mouth). The remaining pixels are presumed-good mid-tone
    skin sourced from the texture itself, sidestepping exposure/color
    mismatches against a separately lit diffuse reference asset.
    """
    lum = luminance(sample)
    keep = (lum > bg_threshold) & (exclude_gate <= 0.5)
    if envelope is not None:
        keep &= envelope > 0.5
    if not np.any(keep):
        raise ValueError(
            "Could not sample self-reference skin: nothing left outside the "
            "concept/highlight/feature-preserve masks. Loosen those masks or "
            "use --diffuse-mode uv/palette instead."
        )
    skin = sample[keep].astype(np.float32)
    mean = skin.mean(axis=0)
    logger.info(
        "Self-reference skin RGB ≈ (%.1f, %.1f, %.1f) from %d pixels (%.1f%% of frame)",
        mean[0], mean[1], mean[2], int(keep.sum()), 100.0 * float(keep.mean()),
    )
    return mean


# =============================================================================
# MASK + BLEND
# =============================================================================
def luminance_delta_mask(
    sample: np.ndarray,
    diffuse_target: np.ndarray,
    threshold: float,
    region_gate: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Soft-threshold mask where |L_sample - L_diffuse| exceeds ``threshold``.

    Returns float32 weights in [0, 1]. Values below the threshold are 0; above
    they ramp toward 1 based on how far they exceed the threshold (capped).
    """
    d_l = np.abs(luminance(sample) - luminance(diffuse_target))
    # Hard gate: only pixels past the threshold participate.
    over = np.maximum(d_l - threshold, 0.0)
    # Soft ramp: full weight once delta is 2x the threshold past the cut.
    ramp = threshold if threshold > 1e-6 else 1.0
    weights = np.clip(over / ramp, 0.0, 1.0).astype(np.float32)

    if region_gate is not None:
        if region_gate.shape == ():
            pass
        else:
            weights *= region_gate.astype(np.float32)

    # Ignore empty UV background (near-black on sample).
    bg = luminance(sample) < 8.0
    weights[bg] = 0.0
    return weights


def apply_blending_radius(mask: np.ndarray, radius: float) -> np.ndarray:
    """Spatially softens the blend mask. ``radius`` is Gaussian sigma in pixels."""
    if radius <= 0:
        return mask
    # Kernel size odd and large enough for the sigma.
    k = int(max(3, round(radius * 6) // 2 * 2 + 1))
    return cv2.GaussianBlur(mask, (k, k), radius)


# =============================================================================
# COMPOSITE SKIN MASK (interior = diffuse, border = blur only)
# =============================================================================
def composite_weights(composite: np.ndarray) -> np.ndarray:
    """Grayscale envelope weights in [0, 1] from a composite skin mask."""
    return (luminance(composite) / 255.0).astype(np.float32)


def composite_skin_envelope(
    composite: np.ndarray,
    support_min: float = 1.0 / 255.0,
) -> np.ndarray:
    """Skin envelope including interior holes (e.g. the face oval in the composite).

    The composite map often has a black face hole surrounded by a white ring.
    Flood-fill identifies those enclosed holes so diffuse can cover the full face
    while still respecting a separate feature preserve mask.
    """
    weights = composite_weights(composite)
    binary = (weights >= support_min).astype(np.uint8)
    if not np.any(binary):
        return binary.astype(np.float32)

    h, w = binary.shape
    inv = (1 - binary).astype(np.uint8)
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 2)
    holes = (flood == 1).astype(np.float32)
    return np.clip(binary.astype(np.float32) + holes, 0.0, 1.0)


def split_composite_zones(
    weights: np.ndarray,
    interior_min: float = 250.0 / 255.0,
    support_min: float = 1.0 / 255.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split composite into full-white interior vs feathered border band.

    Border band receives local blur only (seam softening at composite edges).
    """
    interior = (weights >= interior_min).astype(np.float32)
    border = ((weights >= support_min) & (weights < interior_min)).astype(np.float32)
    border = border * weights
    return interior, border


def mean_luminance_in_mask(
    img: np.ndarray,
    mask: np.ndarray,
    bg_threshold: float = 8.0,
) -> float:
    """Mean Rec. 709 luminance of ``img`` where ``mask`` > 0 (ignoring UV background)."""
    lum = luminance(img)
    keep = (mask.astype(np.float32) > 0.01) & (lum >= bg_threshold)
    if not np.any(keep):
        keep = lum >= bg_threshold
    if not np.any(keep):
        return float(lum.mean())
    return float(lum[keep].mean())


def masked_gaussian_blur(
    rgb: np.ndarray,
    weight_mask: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Blur ``rgb`` weighted by ``weight_mask`` so black UV gutters are not pulled in."""
    if radius <= 0:
        return rgb.astype(np.float32)
    k = int(max(3, round(radius * 6) // 2 * 2 + 1))
    m = weight_mask.astype(np.float32)
    rgb_f = rgb.astype(np.float32)
    num = cv2.GaussianBlur(rgb_f * m[..., None], (k, k), radius)
    # cv2.GaussianBlur on a single-channel map returns (H, W), not (H, W, 1).
    den = cv2.GaussianBlur(m, (k, k), radius)[..., None]
    return num / np.maximum(den, 1e-4)


def blend_toward_flat_luminance(
    sample: np.ndarray,
    target_l: float,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Blend sample luminance toward a single flat target; preserve original chroma."""
    strength = float(np.clip(strength, 0.0, 1.0))
    w = np.clip(mask.astype(np.float32) * strength, 0.0, 1.0)
    sample_f = sample.astype(np.float32)
    y_s, cb, cr = rgb_to_ycbcr(sample_f)
    y_out = y_s * (1.0 - w) + float(target_l) * w
    out = ycbcr_to_rgb(y_out, cb, cr)
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_luminance_from_masked_blur(
    sample: np.ndarray,
    blurred: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Blend sample Y toward blurred Y; preserve original chroma."""
    strength = float(np.clip(strength, 0.0, 1.0))
    w = np.clip(mask.astype(np.float32) * strength, 0.0, 1.0)
    sample_f = sample.astype(np.float32)
    y_s, cb, cr = rgb_to_ycbcr(sample_f)
    y_t = luminance(blurred)
    y_out = y_s * (1.0 - w) + y_t * w
    out = ycbcr_to_rgb(y_out, cb, cr)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_composite_pass(
    sample: np.ndarray,
    diffuse_target: np.ndarray,
    composite: np.ndarray,
    radius: float,
    diffuse_strength: float,
    border_strength: float,
    interior_min: float = 250.0 / 255.0,
    luminance_only: bool = True,
    feature_preserve: Optional[np.ndarray] = None,
    highlight_preserve: Optional[np.ndarray] = None,
    chin_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matte skin toward diffuse, following the composite mask's own gradient.

    The composite mask is itself an authored feather (e.g. white face fading
    to black at the chin/jaw), so the diffuse blend weight is the mask's
    *continuous* grayscale value (softened by ``radius`` to remove authoring
    grain) rather than a hard white-only threshold. This lets a feathered
    edge still receive a proportional amount of flat-luminance correction
    instead of none, which is what was leaving the chin dark and producing a
    hard seam at the interior/border cutoff. A flat mean diffuse luminance
    (not per-pixel UV colors) is used so misaligned diffuse maps do not paint
    a warped / color-shifted pattern onto the face. A second, smaller local
    blur still runs on the low-confidence fringe band to clean up residual
    per-pixel grain right at the mask edge.

    ``chin_mask`` (e.g. ``head_extrapolation_mask_chin_area.png``) extends
    that border band: it marks an extra region that should receive the same
    local-blur-only seam softening even where the composite mask's own
    gradient has already fallen to zero (e.g. below the jaw, outside the
    authored feather). It is unioned into the envelope too, so the border
    blur has real neighboring content to average there instead of degrading
    toward black; it is not added to the diffuse target, so it does not pull
    in extra flat-luminance correction.
    """
    weights = composite_weights(composite)
    envelope = composite_skin_envelope(composite)

    chin_weights = composite_weights(chin_mask) if chin_mask is not None else None
    if chin_weights is not None:
        envelope = np.clip(envelope + (chin_weights > 1.0 / 255.0).astype(np.float32), 0.0, 1.0)

    soft_weights = apply_blending_radius(weights, radius) if radius > 0 else weights
    soft_weights = soft_weights * envelope

    preserve = np.zeros_like(soft_weights)
    if feature_preserve is not None:
        preserve = np.maximum(preserve, composite_weights(feature_preserve))
    if highlight_preserve is not None:
        preserve = np.maximum(preserve, highlight_preserve.astype(np.float32))
    diffuse_mask = soft_weights * (1.0 - np.clip(preserve, 0.0, 1.0))

    working = sample
    if float(diffuse_mask.max()) > 0.0:
        target_l = mean_luminance_in_mask(diffuse_target, envelope)
        logger.info("Composite flat target luminance ≈ %.1f", target_l)
        working = blend_toward_flat_luminance(
            working, target_l, diffuse_mask, diffuse_strength
        )

    _interior, border = split_composite_zones(weights, interior_min)
    if chin_weights is not None:
        border = np.maximum(border, chin_weights)
    border_soft = apply_blending_radius(border, radius) if radius > 0 else border
    if float(border_soft.max()) > 0.0:
        blurred = masked_gaussian_blur(working, envelope, radius)
        working = blend_luminance_from_masked_blur(
            working, blurred, border_soft, border_strength
        )

    return working, diffuse_mask, border_soft


def blend_toward_diffuse(
    sample: np.ndarray,
    diffuse_target: np.ndarray,
    mask: np.ndarray,
    strength: float,
    luminance_only: bool = True,
) -> np.ndarray:
    """Blends sample toward diffuse_target using ``mask * strength``.

    When ``luminance_only`` is True, only Y is blended (chroma preserved from
    the sample) — preferred for matte albedo cleanup.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    w = np.clip(mask.astype(np.float32) * strength, 0.0, 1.0)

    sample_f = sample.astype(np.float32)
    target_f = diffuse_target.astype(np.float32)

    if luminance_only:
        y_s, cb, cr = rgb_to_ycbcr(sample_f)
        y_t = luminance(target_f)
        y_out = y_s * (1.0 - w) + y_t * w
        out = ycbcr_to_rgb(y_out, cb, cr)
    else:
        out = sample_f * (1.0 - w[..., None]) + target_f * w[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# HIGHLIGHT BLUR (paint mask + luminance threshold → local blur, not diffuse)
# =============================================================================
def extract_blue_paint_mask(
    paint_map: np.ndarray,
    min_blue: int = 100,
    blue_margin: int = 40,
) -> np.ndarray:
    """Extracts a float gate from blue-painted highlight areas on a UV paint map.

    Pixels where blue dominates red/green (as in the authored highlight mask)
    become 1; everything else 0.
    """
    r = paint_map[..., 0].astype(np.int16)
    g = paint_map[..., 1].astype(np.int16)
    b = paint_map[..., 2].astype(np.int16)
    blue = (b >= min_blue) & (b > r + blue_margin) & (b > g + blue_margin)
    return blue.astype(np.float32)


def highlight_luminance_mask(
    sample: np.ndarray,
    highlight_gate: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Soft mask of bright pixels inside the highlight paint gate.

    Unlike the concept pass, this does **not** compare to a diffuse target.
    Pixels with luminance above ``threshold`` (0-255) ramp into the mask;
    outside the paint gate they stay 0.
    """
    lum = luminance(sample)
    over = np.maximum(lum - threshold, 0.0)
    ramp = max(threshold * 0.25, 8.0)  # soft shoulder above the cut
    weights = np.clip(over / ramp, 0.0, 1.0).astype(np.float32)
    weights *= highlight_gate.astype(np.float32)
    weights[lum < 8.0] = 0.0
    return weights


def blur_highlights(
    sample: np.ndarray,
    mask: np.ndarray,
    radius: float,
    strength: float,
    diffuse_target: Optional[np.ndarray] = None,
    diffuse_mix: float = 0.0,
    luminance_only: bool = True,
    use_infill: bool = False,
) -> np.ndarray:
    """Softens highlights using local blur and optional diffuse color mix.

    For each masked pixel, builds a correction target::

        target = (1 - diffuse_mix) * blurred_sample + diffuse_mix * diffuse

    then blends ``sample`` toward ``target`` by ``mask * strength``.

    Args:
        sample: Current albedo RGB.
        mask: Soft highlight weights in [0, 1].
        radius: Gaussian sigma (px) for the local blur of ``sample``, or (when
            ``use_infill`` is set) the edge-softening radius after infill.
        strength: Overall mix amount of the correction (0-1).
        diffuse_target: Flat diffuse RGB (same size as sample). Required if
            ``diffuse_mix`` > 0.
        diffuse_mix: Fraction of the correction target taken from diffuse
            (0 = local blur only, 1 = 100% diffuse color).
        luminance_only: Blend Y only (keep sample chroma).
        use_infill: Build the correction target by pushing the nearest
            surrounding (unmasked) colors into the masked region (see
            ``in_fill.py``) instead of Gaussian-blurring the sample in place.
            A plain local blur only softens a bright/mismatched patch — its
            own color stays baked into the kernel average near the center —
            while infill actually replaces the patch with the surrounding
            skin tone.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    diffuse_mix = float(np.clip(diffuse_mix, 0.0, 1.0))
    w = np.clip(mask.astype(np.float32) * strength, 0.0, 1.0)

    sample_f = sample.astype(np.float32)
    if radius <= 0:
        blurred = sample_f
    elif use_infill:
        holes = mask.astype(np.float32) > 0.01
        blurred = extend_texture_boundaries(sample_f, holes, max_distance=None)
        blurred = apply_extrapolation_blur(blurred, holes, radius)
    else:
        k = int(max(3, round(radius * 6) // 2 * 2 + 1))
        blurred = cv2.GaussianBlur(sample_f, (k, k), radius)

    if diffuse_mix <= 0.0 or diffuse_target is None:
        target = blurred
    else:
        diff_f = diffuse_target.astype(np.float32)
        if diff_f.shape[:2] != sample_f.shape[:2]:
            diff_f = resize_to(diff_f, sample_f.shape[:2], nearest=False).astype(np.float32)
        target = blurred * (1.0 - diffuse_mix) + diff_f * diffuse_mix

    if luminance_only:
        y_s, cb, cr = rgb_to_ycbcr(sample_f)
        y_t = luminance(target)
        y_out = y_s * (1.0 - w) + y_t * w
        out = ycbcr_to_rgb(y_out, cb, cr)
    else:
        out = sample_f * (1.0 - w[..., None]) + target * w[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def process(
    texture_path: str,
    diffuse_path: Optional[str],
    out_mask_path: str,
    out_texture_path: Optional[str] = None,
    region_mask_path: Optional[str] = None,
    diffuse_mode: str = "uv",
    threshold: float = 12.0,
    radius: float = 8.0,
    strength: float = 0.85,
    region_tolerance: int = 40,
    regions: Optional[Sequence[str]] = None,
    region_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    luminance_only: bool = True,
    blur_outside_mask: bool = False,
    use_infill: bool = False,
    concept_diffuse_mix: float = 0.0,
    highlight_mask_path: Optional[str] = None,
    hl_threshold: float = 180.0,
    hl_radius: float = 6.0,
    hl_strength: float = 0.7,
    hl_diffuse_mix: float = 0.0,
    hl_blur_outside_mask: bool = False,
    out_hl_mask_path: Optional[str] = None,
    composite_mask_path: Optional[str] = None,
    composite_interior_min: float = 250.0,
    composite_interior_strength: float = 0.85,
    composite_border_strength: float = 0.85,
    composite_radius: float = 8.0,
    feature_preserve_path: Optional[str] = None,
    chin_mask_path: Optional[str] = None,
    exclude_highlights_from_diffuse: bool = True,
    out_composite_border_path: Optional[str] = None,
    out_composite_diffuse_path: Optional[str] = None,
) -> Dict[str, str]:
    """Runs matte correction, optional composite pass, then highlight softening.

    Concept pass: luminance-delta vs diffuse inside ID regions builds the
    mask; the texture is softened toward a local blur of itself (diffuse is
    not written into the result unless ``concept_diffuse_mix`` > 0).

    Composite pass: filled skin envelope (including face interior) blends toward
    diffuse, excluding feature preserve and highlight paint; feathered border
    band gets local blur only for seam softening.

    Highlight pass: inside blue paint mask, soften pixels brighter than
    ``hl_threshold`` by blending toward local blur and/or diffuse color
    (``hl_diffuse_mix`` controls the diffuse percentage of that target).

    When ``blur_outside_mask`` is True, soft mask edges may spill into
    neighboring (unpainted) pixels for **both** concept and highlight passes.
    ``hl_blur_outside_mask`` can still force spill for highlights only.

    ``use_infill`` swaps the concept/highlight correction target from a local
    Gaussian blur to infill (nearest surrounding colors pushed into the
    masked region, see ``in_fill.py``) for both passes. Default: off.
    """
    sample = load_rgb(texture_path)

    id_map = None
    if region_mask_path:
        id_map = resize_to(load_rgb(region_mask_path), sample.shape[:2], nearest=True)

    palette = region_palette or DEFAULT_REGION_PALETTE
    gate = build_region_gate(id_map, palette, region_tolerance, regions)
    if isinstance(gate, np.ndarray) and gate.shape == ():
        gate = np.ones(sample.shape[:2], dtype=np.float32)
    elif id_map is None:
        gate = np.ones(sample.shape[:2], dtype=np.float32)

    if diffuse_mode == "self":
        # Average the sample's own pixels outside whatever is already
        # flagged as needing correction (concept regions), non-skin
        # (feature preserve), or blown-out (highlight paint) — a flat
        # target sourced from the texture itself rather than a separately
        # lit/exposed diffuse asset.
        exclude = gate.copy() if id_map is not None else np.zeros(sample.shape[:2], dtype=np.float32)
        if highlight_mask_path:
            hl_paint_ref = resize_to(load_rgb(highlight_mask_path), sample.shape[:2], nearest=True)
            exclude = np.maximum(exclude, extract_blue_paint_mask(hl_paint_ref))
        if feature_preserve_path:
            feature_ref = resize_to(load_rgb(feature_preserve_path), sample.shape[:2], nearest=False)
            exclude = np.maximum(exclude, composite_weights(feature_ref))
        envelope_ref = None
        if composite_mask_path:
            composite_ref = resize_to(load_rgb(composite_mask_path), sample.shape[:2], nearest=False)
            envelope_ref = composite_skin_envelope(composite_ref)
        skin_rgb = sample_self_reference_skin_rgb(sample, exclude, envelope_ref)
        diffuse_target = np.empty(sample.shape, dtype=np.float32)
        diffuse_target[...] = skin_rgb
    else:
        if not diffuse_path:
            raise ValueError("diffuse_path is required unless diffuse_mode='self'.")
        diffuse_img = load_rgb(diffuse_path)
        diffuse_target = make_diffuse_target(sample, diffuse_img, diffuse_mode)

    # Shared spill switch: one CLI flag covers all blur/feather passes.
    spill_neighbors = bool(blur_outside_mask)

    raw = luminance_delta_mask(sample, diffuse_target, threshold, gate)
    soft = apply_blending_radius(raw, radius)
    if not spill_neighbors:
        soft = soft * gate
    else:
        soft = soft * (luminance(sample) >= 8.0).astype(np.float32)

    mask_u8 = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
    save_rgb(out_mask_path, mask_u8)
    logger.info(
        "Wrote concept blend mask -> %s  (active px: %.1f%%)",
        out_mask_path,
        100.0 * float((soft > 0.01).mean()),
    )

    results: Dict[str, str] = {"mask": out_mask_path}

    working = sample
    if out_texture_path or highlight_mask_path:
        # Diffuse drives the mask only; correction is local blur unless an
        # explicit concept_diffuse_mix pulls toward the flat reference.
        working = blur_highlights(
            sample,
            soft,
            radius,
            strength,
            diffuse_target=diffuse_target if concept_diffuse_mix > 0.0 else None,
            diffuse_mix=concept_diffuse_mix,
            luminance_only=luminance_only,
            use_infill=use_infill,
        )
        logger.info(
            "Applied concept soften (threshold=%.1f, radius=%.1f, "
            "strength=%.2f, diffuse_mix=%.0f%%, infill=%s)",
            threshold, radius, strength, 100.0 * concept_diffuse_mix, use_infill,
        )

    # --- Composite pass (envelope = diffuse, border = blur) ----------------
    composite_border_soft = np.zeros(sample.shape[:2], dtype=np.float32)
    composite_diffuse_mask = np.zeros(sample.shape[:2], dtype=np.float32)
    if composite_mask_path:
        composite = resize_to(load_rgb(composite_mask_path), sample.shape[:2], nearest=False)
        interior_min = float(np.clip(composite_interior_min, 0.0, 255.0)) / 255.0

        feature_preserve = None
        if feature_preserve_path:
            feature_preserve = resize_to(
                load_rgb(feature_preserve_path), sample.shape[:2], nearest=False
            )

        chin_mask = None
        if chin_mask_path:
            chin_mask = resize_to(
                load_rgb(chin_mask_path), sample.shape[:2], nearest=False
            )

        hl_gate_for_diffuse = None
        if exclude_highlights_from_diffuse and highlight_mask_path:
            hl_paint = resize_to(load_rgb(highlight_mask_path), sample.shape[:2], nearest=True)
            hl_gate_for_diffuse = extract_blue_paint_mask(hl_paint)

        if out_texture_path or highlight_mask_path or out_composite_border_path or out_composite_diffuse_path:
            working, composite_diffuse_mask, composite_border_soft = apply_composite_pass(
                working,
                diffuse_target,
                composite,
                composite_radius,
                composite_interior_strength,
                composite_border_strength,
                interior_min=interior_min,
                luminance_only=luminance_only,
                feature_preserve=feature_preserve,
                highlight_preserve=hl_gate_for_diffuse,
                chin_mask=chin_mask,
            )
            logger.info(
                "Applied composite pass (envelope→diffuse minus preserve, border blur "
                "radius=%.1f, diffuse=%.2f, border=%.2f)",
                composite_radius,
                composite_interior_strength,
                composite_border_strength,
            )
        if out_composite_diffuse_path:
            save_rgb(
                out_composite_diffuse_path,
                np.clip(composite_diffuse_mask * 255.0, 0, 255).astype(np.uint8),
            )
            logger.info("Wrote composite diffuse mask -> %s", out_composite_diffuse_path)
            results["composite_diffuse"] = out_composite_diffuse_path
        if out_composite_border_path:
            save_rgb(
                out_composite_border_path,
                np.clip(composite_border_soft * 255.0, 0, 255).astype(np.uint8),
            )
            logger.info("Wrote composite border mask -> %s", out_composite_border_path)
            results["composite_border"] = out_composite_border_path

    # --- Highlight blur pass -------------------------------------------------
    if highlight_mask_path:
        paint = resize_to(load_rgb(highlight_mask_path), working.shape[:2], nearest=True)
        hl_gate = extract_blue_paint_mask(paint)
        hl_raw = highlight_luminance_mask(working, hl_gate, hl_threshold)
        hl_soft = apply_blending_radius(hl_raw, hl_radius)
        # Same neighbor-spill policy as concept, unless highlight-only override.
        hl_spill = spill_neighbors or bool(hl_blur_outside_mask)
        if not hl_spill:
            hl_soft = hl_soft * hl_gate
        else:
            hl_soft = hl_soft * (luminance(working) >= 8.0).astype(np.float32)

        if out_hl_mask_path:
            save_rgb(out_hl_mask_path, np.clip(hl_soft * 255.0, 0, 255).astype(np.uint8))
            logger.info(
                "Wrote highlight blur mask -> %s  (active px: %.1f%%)",
                out_hl_mask_path,
                100.0 * float((hl_soft > 0.01).mean()),
            )
            results["hl_mask"] = out_hl_mask_path

        working = blur_highlights(
            working,
            hl_soft,
            hl_radius,
            hl_strength,
            diffuse_target=diffuse_target,
            diffuse_mix=hl_diffuse_mix,
            luminance_only=luminance_only,
            use_infill=use_infill,
        )
        logger.info(
            "Applied highlight soften (threshold=%.1f, radius=%.1f, "
            "strength=%.2f, diffuse_mix=%.0f%%, infill=%s)",
            hl_threshold, hl_radius, hl_strength, 100.0 * hl_diffuse_mix, use_infill,
        )

    if out_texture_path:
        save_rgb(out_texture_path, working)
        logger.info("Wrote corrected texture -> %s", out_texture_path)
        results["texture"] = out_texture_path

    return results


# =============================================================================
# CLI
# =============================================================================
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Luminance-delta blend mask (+ optional matte correction) "
                    "between a sample albedo and a flat diffuse reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--texture", required=True, help="Sample albedo UV texture.")
    parser.add_argument(
        "--diffuse",
        default=None,
        help="Flat diffuse UV (mode=uv) or multiview/palette flat render (mode=palette). "
             "Not needed when --diffuse-mode self.",
    )
    parser.add_argument(
        "--diffuse-mode",
        choices=("uv", "palette", "self"),
        default="uv",
        help="How to interpret --diffuse. 'self' ignores --diffuse and instead averages "
             "the sample texture's own pixels outside the concept/highlight/feature-preserve "
             "masks as the flat target. Default: uv.",
    )
    parser.add_argument(
        "--region-mask",
        default=None,
        help="Optional color ID mask; limits correction to painted regions.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help=f"Subset of ID regions to use. Default: all. Known: {sorted(DEFAULT_REGION_PALETTE)}",
    )
    parser.add_argument(
        "--region-tolerance",
        type=int,
        default=40,
        help="RGB tolerance when matching ID mask colors. Default: 40.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=12.0,
        help="Min |dLuminance| (0-255) before a pixel enters the blend mask. Default: 12.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=8.0,
        help="Blending radius: Gaussian sigma in pixels applied to the mask. Default: 8.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="Blending strength of the concept soften once masked (0-1). Default: 0.85.",
    )
    parser.add_argument(
        "--concept-diffuse-mix",
        type=float,
        default=0.0,
        help="Fraction of concept correction target taken from diffuse "
             "(0=local blur only — default; 1=bake flat diffuse). Default: 0.",
    )
    parser.add_argument(
        "--full-rgb",
        action="store_true",
        help="Blend full RGB instead of luminance-only (chroma may shift).",
    )
    parser.add_argument(
        "--blur-outside-mask",
        action="store_true",
        help="Allow soft blur/feather to spill into neighboring unpainted pixels "
             "for BOTH the concept pass (--radius) and the highlight pass "
             "(--hl-radius). Default: re-clip each pass to its painted mask.",
    )
    parser.add_argument(
        "--use-infill",
        action="store_true",
        help="Build the concept/highlight correction target by pushing nearest "
             "surrounding colors into the masked region (see in_fill.py) instead "
             "of a local Gaussian blur. Applies to BOTH passes. Default: off.",
    )
    # --- Highlight soften (blue paint; local blur + optional diffuse mix)
    parser.add_argument(
        "--highlight-mask",
        default=None,
        help="Blue-painted UV highlight mask. Enables a second pass that softens "
             "bright pixels inside the paint via local blur and/or diffuse mix.",
    )
    parser.add_argument(
        "--hl-threshold",
        type=float,
        default=180.0,
        help="Min luminance (0-255) inside the highlight paint before soften. Default: 180.",
    )
    parser.add_argument(
        "--hl-radius",
        type=float,
        default=6.0,
        help="Highlight blur radius: Gaussian sigma (px) for mask feather + image blur. Default: 6.",
    )
    parser.add_argument(
        "--hl-strength",
        type=float,
        default=0.7,
        help="How strongly to apply the highlight correction (0-1). Default: 0.7.",
    )
    parser.add_argument(
        "--hl-diffuse-mix",
        type=float,
        default=0.0,
        help="Fraction of highlight correction target taken from diffuse "
             "(0=local blur only, 0.5=50%% diffuse, 1=100%% diffuse). Default: 0.",
    )
    parser.add_argument(
        "--hl-blur-outside-mask",
        action="store_true",
        help="Highlight-only override: spill --hl-radius soft edge outside the "
             "blue paint even if --blur-outside-mask is not set.",
    )
    parser.add_argument(
        "--out-hl-mask",
        default=None,
        help="Optional path to write the highlight blur weight mask (debug).",
    )
    parser.add_argument(
        "--composite-mask",
        default=None,
        help="Grayscale skin composite UV mask. Full-white interior blends toward "
             "diffuse; feathered border band gets local blur only.",
    )
    parser.add_argument(
        "--composite-interior-min",
        type=float,
        default=250.0,
        help="Min composite luminance (0-255) treated as full interior (diffuse). Default: 250.",
    )
    parser.add_argument(
        "--composite-interior-strength",
        type=float,
        default=0.85,
        help="Diffuse blend strength inside full-white composite (0-1). Default: 0.85.",
    )
    parser.add_argument(
        "--composite-border-strength",
        type=float,
        default=0.85,
        help="Local blur strength on composite border/feather band (0-1). Default: 0.85.",
    )
    parser.add_argument(
        "--composite-radius",
        type=float,
        default=8.0,
        help="Gaussian sigma (px) for composite border blur. Default: 8.",
    )
    parser.add_argument(
        "--feature-preserve-mask",
        default=None,
        help="Grayscale mask: bright areas (eyes/mouth/etc.) are preserved from diffuse matte.",
    )
    parser.add_argument(
        "--chin-mask",
        default=None,
        help="Grayscale mask (e.g. head_extrapolation_mask_chin_area.png) extending the "
             "composite pass's border band into an extra region for local-blur-only seam "
             "softening, even where the composite mask's own gradient has fallen to zero.",
    )
    parser.add_argument(
        "--exclude-highlights-from-diffuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude blue highlight paint from composite diffuse coverage (default: on).",
    )
    parser.add_argument(
        "--out-composite-border",
        default=None,
        help="Optional path to write the composite border blur weight mask (debug).",
    )
    parser.add_argument(
        "--out-composite-diffuse",
        default=None,
        help="Optional path to write the composite diffuse coverage mask (debug).",
    )
    parser.add_argument("--out-mask", required=True, help="Output grayscale concept blend mask path.")
    parser.add_argument(
        "--out-texture",
        default=None,
        help="Optional corrected albedo path. If omitted, only the mask is written.",
    )
    parser.add_argument(
        "--palette-json",
        default=None,
        help="Optional JSON overriding DEFAULT_REGION_PALETTE "
             '(e.g. {"forehead": [200,16,120], ...}).',
    )
    args = parser.parse_args(argv)

    if args.diffuse_mode != "self" and not args.diffuse:
        parser.error("--diffuse is required unless --diffuse-mode self is used.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    region_palette = None
    if args.palette_json:
        with open(args.palette_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        region_palette = {k: tuple(v) for k, v in raw.items()}

    process(
        texture_path=args.texture,
        diffuse_path=args.diffuse,
        out_mask_path=args.out_mask,
        out_texture_path=args.out_texture,
        region_mask_path=args.region_mask,
        diffuse_mode=args.diffuse_mode,
        threshold=args.threshold,
        radius=args.radius,
        strength=args.strength,
        region_tolerance=args.region_tolerance,
        regions=args.regions,
        region_palette=region_palette,
        luminance_only=not args.full_rgb,
        blur_outside_mask=args.blur_outside_mask,
        use_infill=args.use_infill,
        concept_diffuse_mix=args.concept_diffuse_mix,
        highlight_mask_path=args.highlight_mask,
        hl_threshold=args.hl_threshold,
        hl_radius=args.hl_radius,
        hl_strength=args.hl_strength,
        hl_diffuse_mix=args.hl_diffuse_mix,
        hl_blur_outside_mask=args.hl_blur_outside_mask,
        out_hl_mask_path=args.out_hl_mask,
        composite_mask_path=args.composite_mask,
        composite_interior_min=args.composite_interior_min,
        composite_interior_strength=args.composite_interior_strength,
        composite_border_strength=args.composite_border_strength,
        composite_radius=args.composite_radius,
        feature_preserve_path=args.feature_preserve_mask,
        chin_mask_path=args.chin_mask,
        exclude_highlights_from_diffuse=args.exclude_highlights_from_diffuse,
        out_composite_border_path=args.out_composite_border,
        out_composite_diffuse_path=args.out_composite_diffuse,
    )


if __name__ == "__main__":
    main()
