"""The preview array cache: what makes run_preview cheap on repeated calls.

Extracting a bpy.types.Image's pixels costs time proportional to its native
resolution — benchmarked at ~500-650ms per preview recompute against a real
4096x4096 mask before this cache existed, ~15-100ms after (see
scene.bake._cached_downsampled_rgb's docstring). These tests check the
functional correctness of that cache (identity/reuse, key separation,
invalidation) rather than timing, which isn't a reliable pytest assertion.
"""

from __future__ import annotations

import bmesh
import bpy
import numpy as np
import pytest

from mask_luminance.scene import bake as scene_bake
from mask_luminance.scene import uv_bounds
from mask_luminance.scene.images import image_to_rgb


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    scene_bake.clear_preview_cache()
    yield
    scene_bake.clear_preview_cache()


@pytest.fixture
def mask_image(repo_root):
    path = repo_root / "masks" / "shadow_mask_1.png"
    if not path.exists():
        pytest.skip(f"Reference mask missing: {path}")
    image = bpy.data.images.load(str(path), check_existing=True)
    yield image
    bpy.data.images.remove(image)


def test_repeated_calls_return_the_same_cached_array_object(mask_image):
    first = scene_bake._cached_downsampled_rgb(mask_image, 256)
    second = scene_bake._cached_downsampled_rgb(mask_image, 256)
    assert first is second


def test_different_max_dimension_is_a_different_cache_entry(mask_image):
    small = scene_bake._cached_downsampled_rgb(mask_image, 128)
    big = scene_bake._cached_downsampled_rgb(mask_image, 256)
    assert small is not big
    assert max(small.shape[:2]) <= 128
    assert max(big.shape[:2]) <= 256


def test_nearest_flag_is_a_different_cache_entry_with_identical_pixels(mask_image):
    """blue_paint/color_id masks need nearest resize; bilinear caching them would corrupt IDs."""
    bilinear = scene_bake._cached_downsampled_rgb(mask_image, 256, nearest=False)
    nearest = scene_bake._cached_downsampled_rgb(mask_image, 256, nearest=True)
    assert bilinear is not nearest


def test_cached_result_matches_uncached_downsample(mask_image):
    cached = scene_bake._cached_downsampled_rgb(mask_image, 200)
    uncached = scene_bake._downsample_to_max_dimension(image_to_rgb(mask_image), 200)
    np.testing.assert_array_equal(cached, uncached)


def test_clear_preview_cache_forces_a_fresh_array(mask_image):
    first = scene_bake._cached_downsampled_rgb(mask_image, 256)
    scene_bake.clear_preview_cache()
    second = scene_bake._cached_downsampled_rgb(mask_image, 256)
    assert first is not second
    np.testing.assert_array_equal(first, second)


def test_clear_preview_cache_by_name_only_clears_that_image(repo_root, mask_image):
    other_path = repo_root / "masks" / "highlight_mask.png"
    if not other_path.exists():
        pytest.skip(f"Reference mask missing: {other_path}")
    other_image = bpy.data.images.load(str(other_path), check_existing=True)
    try:
        first_a = scene_bake._cached_downsampled_rgb(mask_image, 256)
        first_b = scene_bake._cached_downsampled_rgb(other_image, 256)

        scene_bake.clear_preview_cache(mask_image.name)

        second_a = scene_bake._cached_downsampled_rgb(mask_image, 256)
        second_b = scene_bake._cached_downsampled_rgb(other_image, 256)

        assert second_a is not first_a  # invalidated
        assert second_b is first_b  # untouched
    finally:
        bpy.data.images.remove(other_image)


def test_pointer_property_change_invalidates_the_cache(request, mask_image, repo_root):
    """Reassigning which Image a channel points at must not serve a stale cached array."""
    import mask_luminance

    mask_luminance.register()
    request.addfinalizer(mask_luminance.unregister)

    settings = bpy.context.scene.mask_luminance
    settings.channels.clear()

    other_path = repo_root / "masks" / "highlight_mask.png"
    if not other_path.exists():
        pytest.skip(f"Reference mask missing: {other_path}")
    other_image = bpy.data.images.load(str(other_path), check_existing=True)
    try:
        bpy.ops.mask_luminance.channel_add()
        channel = settings.channels[0]

        channel.mask_image = mask_image
        cached_first = scene_bake._cached_downsampled_rgb(mask_image, 256)

        # Reassigning the pointer should clear the whole cache (see
        # operators._on_preview_relevant_pointer_change), not just leave the
        # old entry sitting there unused.
        channel.mask_image = other_image

        cached_again = scene_bake._cached_downsampled_rgb(mask_image, 256)
        assert cached_again is not cached_first
    finally:
        bpy.data.images.remove(other_image)


def test_clear_preview_cache_also_clears_uv_bounds_cache():
    """The one "Clear Preview Cache" button/no-arg call must cover uv_bounds too.

    Every pointer-property change (including the UV source object) routes
    through clear_preview_cache()'s no-arg path — see
    operators._on_preview_relevant_pointer_change — so UV rasterizations need
    no separate invalidation wiring as long as this holds.
    """
    mesh = bpy.data.meshes.new("uv_cache_test_mesh")
    bm = bmesh.new()
    try:
        v_a = bm.verts.new((0.0, 0.0, 0.0))
        v_b = bm.verts.new((1.0, 0.0, 0.0))
        v_c = bm.verts.new((1.0, 1.0, 0.0))
        v_d = bm.verts.new((0.0, 1.0, 0.0))
        face = bm.faces.new((v_a, v_b, v_c, v_d))
        layer = bm.loops.layers.uv.new("UVMap")
        for loop, uv in zip(face.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            loop[layer].uv = uv
        bm.to_mesh(mesh)
    finally:
        bm.free()
    obj = bpy.data.objects.new("uv_cache_test_obj", mesh)

    try:
        first = uv_bounds.rasterize_uv_bounds(obj, (32, 32))
        scene_bake.clear_preview_cache()
        second = uv_bounds.rasterize_uv_bounds(obj, (32, 32))
        assert second is not first
        np.testing.assert_array_equal(first, second)
    finally:
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
