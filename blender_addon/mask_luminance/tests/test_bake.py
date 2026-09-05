"""scene.bake, driven entirely by bpy.types.Image datablocks, must match core.blend.process().

This is the test that actually proves the Image <-> array bridge doesn't
change the result: same channel config as test_blend.py's legacy-tool parity
test, but every input arrives as a live Image datablock (as it will from the
addon's future PointerProperty pickers) instead of a file path.
"""

from __future__ import annotations

import bmesh
import bpy
import numpy as np
import pytest

from mask_luminance.core import blend as core_blend
from mask_luminance.scene import bake as scene_bake
from mask_luminance.scene.images import image_to_rgb, rgb_to_image

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


def _make_quad_mesh_object(name, uv_rect):
    """One-quad-face mesh Object whose UV loop coords span uv_rect=(u0, v0, u1, v1)."""
    u0, v0, u1, v1 = uv_rect
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    try:
        v_a = bm.verts.new((0.0, 0.0, 0.0))
        v_b = bm.verts.new((1.0, 0.0, 0.0))
        v_c = bm.verts.new((1.0, 1.0, 0.0))
        v_d = bm.verts.new((0.0, 1.0, 0.0))
        face = bm.faces.new((v_a, v_b, v_c, v_d))
        layer = bm.loops.layers.uv.new("UVMap")
        for loop, uv in zip(face.loops, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]):
            loop[layer].uv = uv
        bm.to_mesh(mesh)
    finally:
        bm.free()
    return bpy.data.objects.new(name, mesh)


def test_uv_source_object_clips_correction_to_real_uv_bounds():
    """A channel that would otherwise correct the whole canvas must stop at the real UV edge.

    Regression guard for the original bug report: without real UV geometry,
    blur/infill only guesses "background" from pixel brightness and can spill
    across a UV gutter narrower than the configured radius. With a
    uv_source_object given, correction must be clipped to pixels the mesh's
    UV layout actually covers — here, only the top half of the canvas (UV
    v in [0.5, 1.0], which is the top half of the top-down array).
    """
    size = 20
    sample_arr = np.full((size, size, 3), 120, dtype=np.uint8)
    mask_arr = np.full((size, size, 3), 255, dtype=np.uint8)  # gate = 1 everywhere

    source_image = rgb_to_image(sample_arr, "uv_bounds_test_source")
    mask_image = rgb_to_image(mask_arr, "uv_bounds_test_mask", non_color=True)
    obj = _make_quad_mesh_object("uv_bounds_test_obj", (0.0, 0.5, 1.0, 1.0))

    channel = core_blend.MaskChannel(
        name="c", mask_path="unused", enabled=True, gate_mode="weight",
        threshold=1.0, radius=0.0, strength=1.0, flat_fill=True, mask_authoritative=True,
    )
    try:
        result_image, _ = scene_bake.run_bake(
            source=source_image,
            channels=[channel],
            mask_images={"c": mask_image},
            diffuse_mode="self",
            diffuse_color_override=(200.0, 50.0, 50.0),
            uv_source_object=obj,
        )
        try:
            baked = image_to_rgb(result_image)
        finally:
            bpy.data.images.remove(result_image)

        # Row 2: well inside the rasterized UV footprint (rows 0-10), corrected.
        assert np.any(baked[2] != sample_arr[2])
        # Row 17: well outside the UV footprint (rows 11-19), must stay byte-identical.
        # (Row 10 itself sits exactly on the rasterizer's rounded boundary and is
        # excluded from this check on purpose — see test_uv_bounds.py for the
        # rasterizer's own boundary behavior.)
        np.testing.assert_array_equal(baked[17], sample_arr[17])
    finally:
        bpy.data.images.remove(source_image)
        bpy.data.images.remove(mask_image)
        mesh = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
