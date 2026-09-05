"""Full-pipeline parity: ported core.blend must match the original CLI tool exactly.

Runs the same channel configuration, against the same real texture/mask
assets already checked into the repo, through both the original
``matte_luminance_blend.py`` (real ``genies`` dependency) and the ported
``mask_luminance.core.blend`` (vendored infill, no ``genies``/``bpy``), then
diffs the two output textures byte-for-byte. This is the test that actually
proves the port didn't change behavior — ``test_infill.py`` only proves the
two infill functions agree in isolation.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mask_luminance.core import blend as core_blend

CHANNEL_SPECS = [
    dict(
        name="shadow_1",
        mask_path="masks/shadow_mask_1.png",
        enabled=True,
        gate_mode="weight",
        threshold=10.0,
        radius=10.0,
        strength=0.8,
        diffuse_mix=0.2,
        use_infill=True,
    ),
    dict(
        name="highlight",
        mask_path="masks/highlight_mask.png",
        enabled=True,
        gate_mode="weight",
        threshold=14.0,
        radius=6.0,
        strength=0.6,
        diffuse_mix=0.0,
        use_infill=False,
    ),
    dict(
        name="chin_shadow",
        mask_path="masks/chin_shadow_mask_1.png",
        enabled=True,
        gate_mode="weight",
        threshold=8.0,
        radius=12.0,
        strength=0.7,
        diffuse_mix=0.3,
        use_infill=True,
        fill_holes=True,
    ),
]


@pytest.fixture
def texture_path(repo_root):
    path = repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png"
    if not path.exists():
        pytest.skip(f"Reference texture missing: {path}")
    return path


@pytest.fixture
def feature_preserve_path(repo_root):
    path = repo_root / "masks" / "eye_mouth_mask.png"
    if not path.exists():
        pytest.skip(f"Reference mask missing: {path}")
    return path


def _resolve_mask_paths(repo_root, specs):
    resolved = []
    for spec in specs:
        spec = dict(spec)
        mask_path = repo_root / spec["mask_path"]
        if not mask_path.exists():
            pytest.skip(f"Reference mask missing: {mask_path}")
        spec["mask_path"] = str(mask_path)
        resolved.append(spec)
    return resolved


def test_process_matches_legacy_tool_self_diffuse(
    legacy_blend, repo_root, texture_path, feature_preserve_path, tmp_path
):
    specs = _resolve_mask_paths(repo_root, CHANNEL_SPECS)

    legacy_channels = [legacy_blend.MaskChannel(**s) for s in specs]
    ported_channels = [core_blend.MaskChannel(**s) for s in specs]

    legacy_out = tmp_path / "legacy.png"
    ported_out = tmp_path / "ported.png"

    legacy_blend.process(
        texture_path=str(texture_path),
        channels=legacy_channels,
        out_texture_path=str(legacy_out),
        diffuse_mode="self",
        feature_preserve_path=str(feature_preserve_path),
    )
    core_blend.process(
        texture_path=str(texture_path),
        channels=ported_channels,
        out_texture_path=str(ported_out),
        diffuse_mode="self",
        feature_preserve_path=str(feature_preserve_path),
    )

    legacy_pixels = core_blend.load_rgb(str(legacy_out))
    ported_pixels = core_blend.load_rgb(str(ported_out))

    np.testing.assert_array_equal(ported_pixels, legacy_pixels)


def test_process_matches_legacy_tool_no_feature_preserve(legacy_blend, repo_root, texture_path, tmp_path):
    specs = _resolve_mask_paths(repo_root, CHANNEL_SPECS[:1])

    legacy_channels = [legacy_blend.MaskChannel(**s) for s in specs]
    ported_channels = [core_blend.MaskChannel(**s) for s in specs]

    legacy_out = tmp_path / "legacy.png"
    ported_out = tmp_path / "ported.png"

    legacy_blend.process(
        texture_path=str(texture_path),
        channels=legacy_channels,
        out_texture_path=str(legacy_out),
        diffuse_mode="self",
    )
    core_blend.process(
        texture_path=str(texture_path),
        channels=ported_channels,
        out_texture_path=str(ported_out),
        diffuse_mode="self",
    )

    legacy_pixels = core_blend.load_rgb(str(legacy_out))
    ported_pixels = core_blend.load_rgb(str(ported_out))

    np.testing.assert_array_equal(ported_pixels, legacy_pixels)


FLAT_FILL_CHANNEL_SPECS = [
    dict(
        name="eyebrow_lip",
        mask_path="masks/eyebrow_lip_mask.png",
        enabled=True,
        gate_mode="blue_paint",
        threshold=1.0,
        radius=20.0,
        strength=1.0,
        flat_fill=True,
    ),
]


def test_process_matches_legacy_tool_flat_fill(legacy_blend, repo_root, texture_path, tmp_path):
    """flat_fill (and its outward-only feathering) must match the original tool too."""
    specs = _resolve_mask_paths(repo_root, FLAT_FILL_CHANNEL_SPECS)

    legacy_channels = [legacy_blend.MaskChannel(**s) for s in specs]
    ported_channels = [core_blend.MaskChannel(**s) for s in specs]

    legacy_out = tmp_path / "legacy.png"
    ported_out = tmp_path / "ported.png"

    # luminance_only=True (the default) on purpose: flat_fill must override it
    # internally on both sides, so this also guards the chroma-leak fix.
    legacy_blend.process(
        texture_path=str(texture_path),
        channels=legacy_channels,
        out_texture_path=str(legacy_out),
        diffuse_mode="self",
    )
    core_blend.process(
        texture_path=str(texture_path),
        channels=ported_channels,
        out_texture_path=str(ported_out),
        diffuse_mode="self",
    )

    legacy_pixels = core_blend.load_rgb(str(legacy_out))
    ported_pixels = core_blend.load_rgb(str(ported_out))

    np.testing.assert_array_equal(ported_pixels, legacy_pixels)


def test_flat_fill_covers_interior_fully_regardless_of_radius(repo_root, texture_path):
    """A flat_fill channel's painted interior must stay fully opaque at any radius.

    Regression guard for the bug this feature was built to avoid:
    apply_blending_radius's edge-straddling taper erodes a thin shape's whole
    interior once radius exceeds its half-width. feather_mask_outward must not.
    """
    mask_path = repo_root / "masks" / "eyebrow_lip_mask.png"
    if not mask_path.exists():
        pytest.skip(f"Reference mask missing: {mask_path}")

    sample = core_blend.load_rgb(str(texture_path))
    mask_img = core_blend.resize_to(core_blend.load_rgb(str(mask_path)), sample.shape[:2], nearest=True)
    gate = core_blend.extract_blue_paint_mask(mask_img) > 0
    assert gate.any(), "eyebrow_lip_mask.png produced an empty gate at this texture's resolution"

    dist = cv2.distanceTransform(gate.astype(np.uint8), cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    half_width = float(dist[cy, cx])

    expected_target = core_blend.build_local_diffuse_target(
        sample, gate.astype(np.float32), core_blend.DEFAULT_SELF_LOCALITY_RADIUS, flat=True
    )[cy, cx]

    mask_imgs = {"eyebrow_lip": mask_img}
    for radius in (6.0, half_width * 3.0):
        ch = core_blend.MaskChannel(
            name="eyebrow_lip", mask_path="masks/eyebrow_lip_mask.png", enabled=True,
            gate_mode="blue_paint", threshold=1.0, radius=radius, strength=1.0, flat_fill=True,
        )
        working, _ = core_blend.process_arrays(sample, [ch], mask_imgs, diffuse_mode="self")
        # Full RGB, full strength, deep interior: must land on the flat
        # self-fill target almost exactly, at every radius.
        assert np.abs(working[cy, cx].astype(np.float32) - expected_target).max() < 5


def test_flat_fill_covers_dark_painted_content_not_just_bright(repo_root, texture_path):
    """flat_fill must cover legitimately dark painted pixels (e.g. an inner lip
    line, near-black brow hairs) at small radius, not just bright ones.

    Regression guard: luminance_delta_mask's "sample luminance < 8 == empty UV
    gutter" background check can't tell that dark content apart from real
    background and used to zero it out completely — the darkest, most visible
    part of the painted feature would silently stay uncorrected, needing an
    unreasonably large radius before feather_mask_outward's outward bleed
    happened to bridge across it from nearby brighter gated pixels.
    """
    mask_path = repo_root / "masks" / "eyebrow_lip_mask.png"
    if not mask_path.exists():
        pytest.skip(f"Reference mask missing: {mask_path}")

    sample = core_blend.load_rgb(str(texture_path))
    mask_img = core_blend.resize_to(core_blend.load_rgb(str(mask_path)), sample.shape[:2], nearest=True)
    gate = core_blend.extract_blue_paint_mask(mask_img) > 0

    dark_in_gate = gate & (core_blend.luminance(sample) < 8.0)
    if not dark_in_gate.any():
        pytest.skip("This texture/mask pairing has no near-black painted pixels to test against.")
    ys, xs = np.where(dark_in_gate)
    dy, dx = int(ys[len(ys) // 2]), int(xs[len(ys) // 2])

    expected_target = core_blend.build_local_diffuse_target(
        sample, gate.astype(np.float32), core_blend.DEFAULT_SELF_LOCALITY_RADIUS, flat=True
    )[dy, dx]

    ch = core_blend.MaskChannel(
        name="eyebrow_lip", mask_path="masks/eyebrow_lip_mask.png", enabled=True,
        gate_mode="blue_paint", threshold=12.0, radius=2.0, strength=1.0, flat_fill=True,
    )
    working, _ = core_blend.process_arrays(sample, [ch], {"eyebrow_lip": mask_img}, diffuse_mode="self")
    assert np.abs(working[dy, dx].astype(np.float32) - expected_target).max() < 5


def test_mask_authoritative_ignores_luminance_delta_threshold():
    """mask_authoritative=True must give full coverage even under-threshold.

    Regression guard for blotchiness: a pixel whose luminance happens to
    already be close to the diffuse target gets zero weight from the normal
    luminance-delta ramp (dL=2 here, well under threshold=12) — leaving
    speckles of the original pixel visible inside an otherwise "fully
    covered" mask. mask_authoritative bypasses that ramp and trusts the
    mask's own opacity directly, for both the flat_fill and normal paths.
    """
    working = np.full((10, 10, 3), 120, dtype=np.uint8)
    diffuse_target = np.full((10, 10, 3), 118, dtype=np.float32)
    mask_img = np.full((10, 10, 3), 255, dtype=np.uint8)
    palette: dict = {}

    for flat_fill in (True, False):
        authoritative = core_blend.MaskChannel(
            name="c", mask_path="unused", enabled=True, gate_mode="weight",
            threshold=12.0, radius=0.0, flat_fill=flat_fill, mask_authoritative=True,
        )
        default = core_blend.MaskChannel(
            name="c", mask_path="unused", enabled=True, gate_mode="weight",
            threshold=12.0, radius=0.0, flat_fill=flat_fill, mask_authoritative=False,
        )

        soft_authoritative = core_blend.compute_channel_soft_mask(working, diffuse_target, mask_img, authoritative, palette)
        soft_default = core_blend.compute_channel_soft_mask(working, diffuse_target, mask_img, default, palette)

        assert np.all(soft_authoritative > 0.99), f"flat_fill={flat_fill}"
        assert np.all(soft_default < 1e-6), f"flat_fill={flat_fill}"


def test_process_arrays_matches_process_file_path_entry_point(repo_root, texture_path, feature_preserve_path, tmp_path):
    """The addon calls process_arrays() directly; it must agree with process()."""
    specs = _resolve_mask_paths(repo_root, CHANNEL_SPECS)
    channels = [core_blend.MaskChannel(**s) for s in specs]

    file_out = tmp_path / "via_process.png"
    core_blend.process(
        texture_path=str(texture_path),
        channels=channels,
        out_texture_path=str(file_out),
        diffuse_mode="self",
        feature_preserve_path=str(feature_preserve_path),
    )

    sample = core_blend.load_rgb(str(texture_path))
    mask_imgs = {
        ch.name: core_blend.resize_to(
            core_blend.load_rgb(ch.mask_path), sample.shape[:2], nearest=ch.gate_mode in ("blue_paint", "color_id")
        )
        for ch in channels
    }
    feature_preserve_img = core_blend.resize_to(
        core_blend.load_rgb(str(feature_preserve_path)), sample.shape[:2], nearest=False
    )

    working, _ = core_blend.process_arrays(
        sample,
        channels,
        mask_imgs,
        diffuse_mode="self",
        feature_preserve_img=feature_preserve_img,
    )

    file_pixels = core_blend.load_rgb(str(file_out))
    np.testing.assert_array_equal(working, file_pixels)
