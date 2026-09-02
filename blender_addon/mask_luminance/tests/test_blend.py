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
