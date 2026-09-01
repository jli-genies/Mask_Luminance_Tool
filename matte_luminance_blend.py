"""Build a luminance-delta blend mask and optionally correct a sample albedo.

Any number of independent *mask channels* can be layered onto a sample
texture. Each channel picks one mask image from ``masks/`` (or elsewhere),
a gate mode describing how that image selects pixels, and its own
threshold / blur radius / diffuse strength / diffuse mix. Channels run in
order, each correcting the output of the previous one, all through the same
core algorithm: a soft luminance-delta mask (vs a diffuse target), feathered
by a blur radius, then repainted via in_fill (nearest surrounding colors) or
a local blur, optionally mixed toward the diffuse target.

The diffuse target defaults to ``self`` mode: a flat skin color sampled from
the sample texture's own clean pixels (those outside every enabled channel's
gate and outside the feature-preserve mask), so no separate diffuse asset is
required.

Typical use::

    python matte_luminance_blend.py \\
        --texture african_female_0003_albedo_from_concept.png \\
        --channels-config channels.json \\
        --feature-preserve-mask masks/head_extrapolation_mask.png \\
        --out-texture out/albedo_matte.png \\
        --out-masks-dir out/channel_masks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from genies.meshutils.shading.texture_utils import (
    apply_extrapolation_blur,
    extend_texture_boundaries,
)

logger = logging.getLogger(__name__)

# Default ID colors from the authored region mask (quantized / approximate).
# Used only by "color_id" gate-mode channels (e.g. mask_concept_texture.png).
DEFAULT_REGION_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "forehead": (200, 16, 120),   # magenta
    "jaw_cheeks": (232, 232, 232),  # white / light grey
    "back_head": (56, 200, 248),  # cyan
}

GATE_MODES = ("weight", "blue_paint", "color_id")


# =============================================================================
# MASK CHANNEL CONFIG
# =============================================================================
@dataclass
class MaskChannel:
    """One independently controllable mask-driven correction pass.

    Attributes:
        name: Display / output name. Defaults to the mask filename stem.
        mask_path: Path to the mask image driving this channel.
        enabled: Whether this channel runs at all.
        gate_mode: How ``mask_path`` selects pixels —
            "weight": grayscale luminance/255 as a soft gate (default).
            "blue_paint": blue-dominant paint on a UV map marks the gate
                (e.g. body_mat_mask_C_highlights_00.png).
            "color_id": ``mask_path`` is a multi-region ID color map;
                ``regions``/``region_tolerance`` select which named regions
                (from the palette) participate.
        threshold: Minimum |luminance(sample) - luminance(diffuse_target)|
            (0-255) before a gated pixel enters the mask.
        radius: Gaussian sigma (px) used both to feather the mask and (when
            not using infill) to locally blur the correction target.
        strength: Overall blend-in amount of the correction (0-1).
        diffuse_mix: Fraction of the correction target taken from the flat
            diffuse target (0 = local blur/infill only, 1 = full diffuse).
        use_infill: Build the correction target via in_fill (nearest
            surrounding colors pushed into the masked region) instead of a
            plain Gaussian blur. Default on, per the shared core algorithm.
        spill_outside: Allow the feathered mask to spread into neighboring
            pixels outside the gate rather than being re-clipped to it.
        fill_holes: Only meaningful for gate_mode="weight". Flood-fills
            enclosed holes in the mask (e.g. a face-oval cutout inside a
            white ring) so the gate covers the full interior, not just the
            painted ring.
        regions: gate_mode="color_id" only — subset of palette region names
            to include. None means all.
        region_tolerance: gate_mode="color_id" only — RGB tolerance when
            matching ID mask colors.
        blend_group: Optional group name. Channels sharing the same non-empty
            group are not applied sequentially (one on top of the other's
            output); instead each computes its own correction independently
            against the same input and the results are composited together,
            weighted by each channel's own coverage — see ``apply_blend_group``.
            Every other parameter above still applies per-channel as usual.
        blend_weight: Relative contribution of this channel within its
            ``blend_group`` when two channels' coverage overlaps. Ignored
            for channels with no blend_group.
    """

    name: str
    mask_path: str
    enabled: bool = False
    gate_mode: str = "weight"
    threshold: float = 12.0
    radius: float = 8.0
    strength: float = 0.85
    diffuse_mix: float = 0.0
    use_infill: bool = True
    spill_outside: bool = False
    fill_holes: bool = False
    regions: Optional[Sequence[str]] = None
    region_tolerance: int = 40
    blend_group: Optional[str] = None
    blend_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"Unknown gate_mode '{self.gate_mode}'. Known: {GATE_MODES}")


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
    id_map: np.ndarray,
    palette: Dict[str, Tuple[int, int, int]],
    tolerance: int = 40,
    active: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Union of selected ID regions."""
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
    in ``process()``, since it needs the union of enabled channel gates.
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
    bg_threshold: float = 8.0,
) -> np.ndarray:
    """Estimates a flat skin RGB from the sample texture's own "clean" pixels.

    Averages sample pixels outside ``exclude_gate`` — the union of every
    enabled channel's gate plus the feature-preserve mask, i.e. whatever has
    already been flagged as needing correction or as non-skin (eyes/mouth).
    The remaining pixels are presumed-good mid-tone skin sourced from the
    texture itself, sidestepping exposure/color mismatches against a
    separately lit diffuse reference asset.
    """
    lum = luminance(sample)
    keep = (lum > bg_threshold) & (exclude_gate <= 0.5)
    if not np.any(keep):
        raise ValueError(
            "Could not sample self-reference skin: nothing left outside the "
            "enabled channel masks / feature-preserve mask. Loosen those masks "
            "or use --diffuse-mode uv/palette instead."
        )
    skin = sample[keep].astype(np.float32)
    mean = skin.mean(axis=0)
    logger.info(
        "Self-reference skin RGB ≈ (%.1f, %.1f, %.1f) from %d pixels (%.1f%% of frame)",
        mean[0], mean[1], mean[2], int(keep.sum()), 100.0 * float(keep.mean()),
    )
    return mean


# =============================================================================
# GATES (per mask channel)
# =============================================================================
def composite_weights(mask_img: np.ndarray) -> np.ndarray:
    """Grayscale gate weights in [0, 1] from a mask image."""
    return (luminance(mask_img) / 255.0).astype(np.float32)


def composite_skin_envelope(
    mask_img: np.ndarray,
    support_min: float = 1.0 / 255.0,
) -> np.ndarray:
    """Grayscale gate including enclosed holes (e.g. a face oval cutout).

    Some masks paint a ring around the region of interest with a hole in the
    middle (e.g. the composite skin mask's face oval). Flood-fill identifies
    those enclosed holes so the gate covers the full interior.
    """
    weights = composite_weights(mask_img)
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


def extract_blue_paint_mask(
    paint_map: np.ndarray,
    min_blue: int = 100,
    blue_margin: int = 40,
) -> np.ndarray:
    """Extracts a float gate from blue-painted areas on a UV paint map.

    Pixels where blue dominates red/green (as in an authored highlight mask)
    become 1; everything else 0.
    """
    r = paint_map[..., 0].astype(np.int16)
    g = paint_map[..., 1].astype(np.int16)
    b = paint_map[..., 2].astype(np.int16)
    blue = (b >= min_blue) & (b > r + blue_margin) & (b > g + blue_margin)
    return blue.astype(np.float32)


def compute_channel_gate(
    mask_img: np.ndarray,
    channel: MaskChannel,
    palette: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    """Resolves one channel's gate (0-1 weights) from its mask image."""
    if channel.gate_mode == "blue_paint":
        return extract_blue_paint_mask(mask_img)
    if channel.gate_mode == "color_id":
        return build_region_gate(mask_img, palette, channel.region_tolerance, channel.regions)
    return composite_skin_envelope(mask_img) if channel.fill_holes else composite_weights(mask_img)


# =============================================================================
# MASK + BLEND
# =============================================================================
def luminance_delta_mask(
    sample: np.ndarray,
    diffuse_target: np.ndarray,
    threshold: float,
    gate: np.ndarray,
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
    weights *= gate.astype(np.float32)

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


def build_correction_target(
    sample: np.ndarray,
    mask: np.ndarray,
    radius: float,
    diffuse_target: Optional[np.ndarray] = None,
    diffuse_mix: float = 0.0,
    use_infill: bool = True,
) -> np.ndarray:
    """Builds the full-image correction color a mask channel would blend toward.

    ``target = (1 - diffuse_mix) * local_repaint + diffuse_mix * diffuse``.
    This is the "core in_fill algorithm" shared by every mask channel: pushes
    the nearest surrounding (unmasked) colors into the masked region (see
    ``in_fill.py``) rather than Gaussian-blurring the sample in place, since a
    plain local blur only softens a bright/mismatched patch — its own color
    stays baked into the kernel average near the center — while infill
    actually replaces the patch with the surrounding skin tone.
    """
    diffuse_mix = float(np.clip(diffuse_mix, 0.0, 1.0))
    sample_f = sample.astype(np.float32)
    if radius <= 0:
        repaint = sample_f
    elif use_infill:
        holes = mask.astype(np.float32) > 0.01
        repaint = extend_texture_boundaries(sample_f, holes, max_distance=None)
        repaint = apply_extrapolation_blur(repaint, holes, radius)
    else:
        k = int(max(3, round(radius * 6) // 2 * 2 + 1))
        repaint = cv2.GaussianBlur(sample_f, (k, k), radius)

    if diffuse_mix <= 0.0 or diffuse_target is None:
        return repaint

    diff_f = diffuse_target.astype(np.float32)
    if diff_f.shape[:2] != sample_f.shape[:2]:
        diff_f = resize_to(diff_f, sample_f.shape[:2], nearest=False).astype(np.float32)
    return repaint * (1.0 - diffuse_mix) + diff_f * diffuse_mix


def correct_region(
    sample: np.ndarray,
    mask: np.ndarray,
    radius: float,
    strength: float,
    diffuse_target: Optional[np.ndarray] = None,
    diffuse_mix: float = 0.0,
    luminance_only: bool = True,
    use_infill: bool = True,
) -> np.ndarray:
    """Blends ``sample`` toward ``build_correction_target(...)`` by ``mask * strength``.

    This is the standalone (non-blend-group) application of the shared core
    algorithm — see ``build_correction_target`` for what the target is.
    """
    target = build_correction_target(sample, mask, radius, diffuse_target, diffuse_mix, use_infill)
    strength = float(np.clip(strength, 0.0, 1.0))
    w = np.clip(mask.astype(np.float32) * strength, 0.0, 1.0)

    sample_f = sample.astype(np.float32)
    if luminance_only:
        y_s, cb, cr = rgb_to_ycbcr(sample_f)
        y_t = luminance(target)
        y_out = y_s * (1.0 - w) + y_t * w
        out = ycbcr_to_rgb(y_out, cb, cr)
    else:
        out = sample_f * (1.0 - w[..., None]) + target * w[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def compute_channel_soft_mask(
    working: np.ndarray,
    diffuse_target: np.ndarray,
    mask_img: np.ndarray,
    channel: MaskChannel,
    palette: Dict[str, Tuple[int, int, int]],
    feature_preserve: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resolves one channel's soft weight mask: gate -> threshold -> feather."""
    gate = compute_channel_gate(mask_img, channel, palette)
    if feature_preserve is not None:
        gate = gate * (1.0 - np.clip(feature_preserve, 0.0, 1.0))

    raw = luminance_delta_mask(working, diffuse_target, channel.threshold, gate)
    soft = apply_blending_radius(raw, channel.radius)
    if not channel.spill_outside:
        soft = soft * gate
    else:
        soft = soft * (luminance(working) >= 8.0).astype(np.float32)
    return soft


def apply_mask_channel(
    working: np.ndarray,
    diffuse_target: np.ndarray,
    mask_img: np.ndarray,
    channel: MaskChannel,
    palette: Dict[str, Tuple[int, int, int]],
    luminance_only: bool = True,
    feature_preserve: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs one mask channel's full pass: gate -> threshold -> feather -> correct.

    Returns the updated working texture and the channel's soft weight mask
    (post-feather, pre-strength — useful for debug output).
    """
    soft = compute_channel_soft_mask(working, diffuse_target, mask_img, channel, palette, feature_preserve)
    new_working = correct_region(
        working,
        soft,
        channel.radius,
        channel.strength,
        diffuse_target=diffuse_target if channel.diffuse_mix > 0.0 else None,
        diffuse_mix=channel.diffuse_mix,
        luminance_only=luminance_only,
        use_infill=channel.use_infill,
    )
    return new_working, soft


def composite_correction_targets(
    sample: np.ndarray,
    targets: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    luminance_only: bool = True,
) -> np.ndarray:
    """Alpha-composites ``sample`` with a weighted blend of several correction targets.

    ``weights[i]`` is channel i's own contribution weight at each pixel
    (its soft mask already scaled by its strength and blend_weight). A pixel
    touched by only one channel keeps that channel's own correction
    unchanged; a pixel touched by several blends proportionally between
    their targets instead of one channel overwriting another's result.
    """
    stack_w = np.clip(np.stack(weights, axis=0), 0.0, 1.0)
    total_w = np.maximum(stack_w.sum(axis=0), 1e-6)
    # Union coverage (screen-combine) so overlap never exceeds full strength.
    combined_w = np.clip(1.0 - np.prod(1.0 - stack_w, axis=0), 0.0, 1.0)

    sample_f = sample.astype(np.float32)
    if luminance_only:
        y_s, cb, cr = rgb_to_ycbcr(sample_f)
        y_targets = np.stack([luminance(t) for t in targets], axis=0)
        blended_y = (y_targets * stack_w).sum(axis=0) / total_w
        y_out = y_s * (1.0 - combined_w) + blended_y * combined_w
        out = ycbcr_to_rgb(y_out, cb, cr)
    else:
        stack_t = np.stack([t.astype(np.float32) for t in targets], axis=0)
        blended = (stack_t * stack_w[..., None]).sum(axis=0) / total_w[..., None]
        out = sample_f * (1.0 - combined_w[..., None]) + blended * combined_w[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_blend_group(
    working: np.ndarray,
    diffuse_target: np.ndarray,
    mask_imgs: Dict[str, np.ndarray],
    group_channels: Sequence[MaskChannel],
    palette: Dict[str, Tuple[int, int, int]],
    luminance_only: bool = True,
    feature_preserve: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Blends several channels' independent corrections into one result.

    Each channel in ``group_channels`` computes its own gate, threshold,
    radius, diffuse_mix and use_infill exactly as it would standalone — all
    against the same shared ``working`` input rather than chained onto each
    other's output. The corrected results are then combined per pixel via
    ``composite_correction_targets``, weighted by each channel's own coverage
    (soft mask * strength * blend_weight), so individual per-channel control
    is preserved everywhere except the overlap, which blends smoothly instead
    of one channel hard-overwriting the other.
    """
    targets: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    soft_masks: Dict[str, np.ndarray] = {}

    for ch in group_channels:
        soft = compute_channel_soft_mask(working, diffuse_target, mask_imgs[ch.name], ch, palette, feature_preserve)
        soft_masks[ch.name] = soft
        target = build_correction_target(
            working,
            soft,
            ch.radius,
            diffuse_target=diffuse_target if ch.diffuse_mix > 0.0 else None,
            diffuse_mix=ch.diffuse_mix,
            use_infill=ch.use_infill,
        )
        targets.append(target)
        strength = float(np.clip(ch.strength, 0.0, 1.0))
        weights.append(soft * strength * max(ch.blend_weight, 0.0))

    new_working = composite_correction_targets(working, targets, weights, luminance_only)
    return new_working, soft_masks


# =============================================================================
# TOP-LEVEL PROCESS
# =============================================================================
def run_channel_pipeline(
    working: np.ndarray,
    diffuse_target: np.ndarray,
    mask_imgs: Dict[str, np.ndarray],
    active_channels: Sequence[MaskChannel],
    palette: Dict[str, Tuple[int, int, int]],
    luminance_only: bool = True,
    feature_preserve: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Runs every enabled channel over ``working``, in order.

    Channels with no ``blend_group`` are applied sequentially, each
    correcting the previous one's output. Channels sharing a ``blend_group``
    name are instead run together through ``apply_blend_group`` the first
    time any of them is reached, so their results blend by coverage instead
    of overwriting each other. Returns the final texture and each channel's
    soft mask (for debug output).
    """
    working = working.copy()
    channel_masks: Dict[str, np.ndarray] = {}
    seen_groups = set()

    for ch in active_channels:
        if ch.blend_group:
            if ch.blend_group in seen_groups:
                continue
            seen_groups.add(ch.blend_group)
            group_members = [c for c in active_channels if c.blend_group == ch.blend_group]
            working, group_soft = apply_blend_group(
                working, diffuse_target, mask_imgs, group_members, palette, luminance_only, feature_preserve,
            )
            channel_masks.update(group_soft)
            logger.info(
                "Applied blend group '%s' (%s)", ch.blend_group, ", ".join(c.name for c in group_members),
            )
        else:
            working, soft = apply_mask_channel(
                working, diffuse_target, mask_imgs[ch.name], ch, palette, luminance_only, feature_preserve,
            )
            channel_masks[ch.name] = soft
            logger.info(
                "Applied channel '%s' (gate=%s, threshold=%.1f, radius=%.1f, strength=%.2f, "
                "diffuse_mix=%.0f%%, infill=%s, active px=%.1f%%)",
                ch.name, ch.gate_mode, ch.threshold, ch.radius, ch.strength,
                100.0 * ch.diffuse_mix, ch.use_infill, 100.0 * float((soft > 0.01).mean()),
            )

    return working, channel_masks


def process(
    texture_path: str,
    channels: Sequence[MaskChannel],
    out_texture_path: Optional[str] = None,
    diffuse_path: Optional[str] = None,
    diffuse_mode: str = "self",
    region_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    luminance_only: bool = True,
    feature_preserve_path: Optional[str] = None,
    out_masks_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Runs every enabled mask channel in order over ``texture_path``.

    Each channel corrects the output of the previous one. The diffuse target
    defaults to ``self`` mode: a flat skin color sampled from the texture's
    own pixels outside every enabled channel's gate and outside the
    feature-preserve mask.
    """
    sample = load_rgb(texture_path)
    palette = region_palette or DEFAULT_REGION_PALETTE
    active = [ch for ch in channels if ch.enabled]

    mask_imgs: Dict[str, np.ndarray] = {}
    for ch in active:
        nearest = ch.gate_mode in ("blue_paint", "color_id")
        mask_imgs[ch.name] = resize_to(load_rgb(ch.mask_path), sample.shape[:2], nearest=nearest)

    feature_preserve = None
    if feature_preserve_path:
        feature_preserve = composite_weights(
            resize_to(load_rgb(feature_preserve_path), sample.shape[:2], nearest=False)
        )

    if diffuse_mode == "self":
        exclude = np.zeros(sample.shape[:2], dtype=np.float32)
        for ch in active:
            exclude = np.maximum(exclude, compute_channel_gate(mask_imgs[ch.name], ch, palette))
        if feature_preserve is not None:
            exclude = np.maximum(exclude, feature_preserve)
        skin_rgb = sample_self_reference_skin_rgb(sample, exclude)
        diffuse_target = np.empty(sample.shape, dtype=np.float32)
        diffuse_target[...] = skin_rgb
    else:
        if not diffuse_path:
            raise ValueError("diffuse_path is required unless diffuse_mode='self'.")
        diffuse_img = load_rgb(diffuse_path)
        diffuse_target = make_diffuse_target(sample, diffuse_img, diffuse_mode)

    working, channel_masks = run_channel_pipeline(
        sample, diffuse_target, mask_imgs, active, palette, luminance_only, feature_preserve,
    )

    results: Dict[str, str] = {}
    if out_masks_dir:
        for name, soft in channel_masks.items():
            path = os.path.join(out_masks_dir, f"{name}_mask.png")
            save_rgb(path, np.clip(soft * 255.0, 0, 255).astype(np.uint8))
            results[f"mask:{name}"] = path

    if out_texture_path:
        save_rgb(out_texture_path, working)
        logger.info("Wrote corrected texture -> %s", out_texture_path)
        results["texture"] = out_texture_path

    return results


# =============================================================================
# CLI
# =============================================================================
def _channels_from_json(path: str) -> List[MaskChannel]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [MaskChannel(**c) for c in raw]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Layered mask-channel luminance correction for a sample albedo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--texture", required=True, help="Sample albedo UV texture.")
    parser.add_argument(
        "--channels-config",
        required=True,
        help="JSON file: a list of mask-channel objects, each matching the "
             "MaskChannel fields (name, mask_path, enabled, gate_mode, "
             "threshold, radius, strength, diffuse_mix, use_infill, "
             "spill_outside, fill_holes, regions, region_tolerance).",
    )
    parser.add_argument(
        "--diffuse",
        default=None,
        help="Flat diffuse UV (mode=uv) or multiview/palette flat render (mode=palette). "
             "Not needed when --diffuse-mode self (the default).",
    )
    parser.add_argument(
        "--diffuse-mode",
        choices=("uv", "palette", "self"),
        default="self",
        help="How to build the diffuse target. 'self' (default) averages the sample "
             "texture's own pixels outside every enabled channel's gate and the "
             "feature-preserve mask.",
    )
    parser.add_argument(
        "--feature-preserve-mask",
        default=None,
        help="Grayscale mask: bright areas (eyes/mouth/etc.) are protected from every "
             "channel's correction and excluded from self-mode skin sampling.",
    )
    parser.add_argument(
        "--full-rgb",
        action="store_true",
        help="Blend full RGB instead of luminance-only (chroma may shift).",
    )
    parser.add_argument("--out-texture", default=None, help="Optional corrected albedo path.")
    parser.add_argument(
        "--out-masks-dir",
        default=None,
        help="Optional directory to write each enabled channel's soft mask (debug).",
    )
    parser.add_argument(
        "--palette-json",
        default=None,
        help="Optional JSON overriding DEFAULT_REGION_PALETTE for color_id channels "
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

    channels = _channels_from_json(args.channels_config)

    process(
        texture_path=args.texture,
        channels=channels,
        out_texture_path=args.out_texture,
        diffuse_path=args.diffuse,
        diffuse_mode=args.diffuse_mode,
        region_palette=region_palette,
        luminance_only=not args.full_rgb,
        feature_preserve_path=args.feature_preserve_mask,
        out_masks_dir=args.out_masks_dir,
    )


if __name__ == "__main__":
    main()
