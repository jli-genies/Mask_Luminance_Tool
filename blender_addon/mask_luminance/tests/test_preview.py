"""run_preview() — run_bake() at a downsampled resolution.

Verifies the one thing that makes proxy preview correct rather than just
fast: downsampling only the source array and letting every mask/diffuse/
feature-preserve array resize to match (already how prepare_bake works)
means run_preview isn't a different algorithm, just the same one at a
smaller size — so a preview at full resolution should equal a real bake,
and a preview at a smaller size should equal running the full pipeline on
a pre-shrunk source directly.
"""

from __future__ import annotations

import bpy
import numpy as np
import pytest

from mask_luminance.core import blend as core_blend
from mask_luminance.scene import bake as scene_bake
from mask_luminance.scene.images import image_to_rgb, rgb_to_image

CHANNEL_SPECS = [
    dict(name="shadow_1", mask_path="x", enabled=True, gate_mode="weight", threshold=10.0, radius=10.0, strength=0.8, diffuse_mix=0.2, use_infill=True),
    dict(name="highlight", mask_path="x", enabled=True, gate_mode="weight", threshold=14.0, radius=6.0, strength=0.6, diffuse_mix=0.0, use_infill=False),
]

MASK_FILES = {
    "shadow_1": "masks/shadow_mask_1.png",
    "highlight": "masks/highlight_mask.png",
}


@pytest.fixture
def loaded_images(repo_root):
    paths = {
        "source": repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png",
        **{name: repo_root / rel for name, rel in MASK_FILES.items()},
    }
    for path in paths.values():
        if not path.exists():
            pytest.skip(f"Reference asset missing: {path}")
    loaded = {key: bpy.data.images.load(str(path), check_existing=True) for key, path in paths.items()}
    try:
        yield loaded
    finally:
        for image in loaded.values():
            bpy.data.images.remove(image)


def _channels():
    return [core_blend.MaskChannel(**spec) for spec in CHANNEL_SPECS]


def test_downsample_to_max_dimension_preserves_aspect_and_never_upsamples():
    from mask_luminance.scene.bake import _downsample_to_max_dimension

    portrait = np.zeros((200, 100, 3), dtype=np.uint8)
    resized = _downsample_to_max_dimension(portrait, 50)
    assert resized.shape[:2] == (50, 25)

    small = np.zeros((40, 40, 3), dtype=np.uint8)
    unchanged = _downsample_to_max_dimension(small, 128)
    assert unchanged.shape == small.shape


def test_run_preview_produces_a_smaller_image_named_with_preview_suffix(loaded_images):
    result = scene_bake.run_preview(
        source=loaded_images["source"],
        channels=_channels(),
        mask_images={name: loaded_images[name] for name in MASK_FILES},
        diffuse_mode="self",
        max_dimension=256,
    )
    try:
        assert result.name == f"{loaded_images['source'].name}_preview"
        assert max(result.size) <= 256
        assert max(result.size) < max(loaded_images["source"].size)
    finally:
        bpy.data.images.remove(result)


def test_run_preview_at_full_resolution_matches_run_bake(loaded_images):
    masks = {name: loaded_images[name] for name in MASK_FILES}
    source_hw = (loaded_images["source"].size[1], loaded_images["source"].size[0])

    preview = scene_bake.run_preview(
        source=loaded_images["source"], channels=_channels(), mask_images=masks,
        diffuse_mode="self", max_dimension=max(source_hw),
    )
    try:
        baked, _ = scene_bake.run_bake(
            source=loaded_images["source"], channels=_channels(), mask_images=masks,
            diffuse_mode="self", result_name="full_res_reference",
        )
        try:
            np.testing.assert_array_equal(image_to_rgb(preview), image_to_rgb(baked))
        finally:
            bpy.data.images.remove(baked)
    finally:
        bpy.data.images.remove(preview)


def test_run_preview_matches_running_the_pipeline_on_a_pre_shrunk_source(loaded_images):
    from mask_luminance.scene.bake import _downsample_to_max_dimension

    masks = {name: loaded_images[name] for name in MASK_FILES}

    preview = scene_bake.run_preview(
        source=loaded_images["source"], channels=_channels(), mask_images=masks,
        diffuse_mode="self", max_dimension=200,
    )

    shrunk_rgb = _downsample_to_max_dimension(image_to_rgb(loaded_images["source"]), 200)
    shrunk_source = rgb_to_image(shrunk_rgb, name="shrunk_source_reference")
    try:
        reference, _ = scene_bake.run_bake(
            source=shrunk_source, channels=_channels(), mask_images=masks,
            diffuse_mode="self", result_name="shrunk_reference_result",
        )
        try:
            np.testing.assert_array_equal(image_to_rgb(preview), image_to_rgb(reference))
        finally:
            bpy.data.images.remove(reference)
    finally:
        bpy.data.images.remove(shrunk_source)
        bpy.data.images.remove(preview)
