"""scene.bake, driven entirely by bpy.types.Image datablocks, must match core.blend.process().

This is the test that actually proves the Image <-> array bridge doesn't
change the result: same channel config as test_blend.py's legacy-tool parity
test, but every input arrives as a live Image datablock (as it will from the
addon's future PointerProperty pickers) instead of a file path.
"""

from __future__ import annotations

import bpy
import numpy as np
import pytest

from mask_luminance.core import blend as core_blend
from mask_luminance.scene import bake as scene_bake

CHANNEL_SPECS = [
    dict(
        name="shadow_1",
        mask_path="unused-in-the-addon-path",
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
        mask_path="unused-in-the-addon-path",
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
        mask_path="unused-in-the-addon-path",
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

MASK_FILES = {
    "shadow_1": "masks/shadow_mask_1.png",
    "highlight": "masks/highlight_mask.png",
    "chin_shadow": "masks/chin_shadow_mask_1.png",
}


@pytest.fixture
def loaded_images(repo_root):
    """Loads the texture/masks/feature-preserve mask as real Image datablocks."""
    paths = {
        "source": repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png",
        "feature_preserve": repo_root / "masks" / "eye_mouth_mask.png",
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


def test_bake_matches_process_file_path_entry_point(repo_root, loaded_images, tmp_path):
    channels = [core_blend.MaskChannel(**spec) for spec in CHANNEL_SPECS]

    result_image, _ = scene_bake.run_bake(
        source=loaded_images["source"],
        channels=channels,
        mask_images={name: loaded_images[name] for name in MASK_FILES},
        diffuse_mode="self",
        feature_preserve_image=loaded_images["feature_preserve"],
    )
    try:
        from mask_luminance.scene.images import image_to_rgb

        baked = image_to_rgb(result_image)

        file_out = tmp_path / "via_process.png"
        core_blend.process(
            texture_path=str(repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png"),
            channels=[core_blend.MaskChannel(**{**spec, "mask_path": str(repo_root / MASK_FILES[spec["name"]])}) for spec in CHANNEL_SPECS],
            out_texture_path=str(file_out),
            diffuse_mode="self",
            feature_preserve_path=str(repo_root / "masks" / "eye_mouth_mask.png"),
        )
        file_pixels = core_blend.load_rgb(str(file_out))

        np.testing.assert_array_equal(baked, file_pixels)
    finally:
        bpy.data.images.remove(result_image)


def test_bake_reuses_result_image_by_default_name(loaded_images):
    channels = [core_blend.MaskChannel(**CHANNEL_SPECS[0])]

    result_image, _ = scene_bake.run_bake(
        source=loaded_images["source"],
        channels=channels,
        mask_images={"shadow_1": loaded_images["shadow_1"]},
        diffuse_mode="self",
    )
    try:
        assert result_image.name == f"{loaded_images['source'].name}_matte"

        result_image_again, _ = scene_bake.run_bake(
            source=loaded_images["source"],
            channels=channels,
            mask_images={"shadow_1": loaded_images["shadow_1"]},
            diffuse_mode="self",
        )
        assert result_image_again.as_pointer() == result_image.as_pointer()
    finally:
        bpy.data.images.remove(result_image)


def test_bake_missing_mask_for_enabled_channel_raises(loaded_images):
    channels = [core_blend.MaskChannel(**CHANNEL_SPECS[0])]

    with pytest.raises(KeyError):
        scene_bake.run_bake(
            source=loaded_images["source"],
            channels=channels,
            mask_images={},
            diffuse_mode="self",
        )
