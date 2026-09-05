"""scene.uv_bounds: rasterizing a mesh's real UV layout into a per-pixel boundary mask.

Real bmesh-built meshes throughout — no mocking needed, since this repo's test
suite runs against the real ``bpy`` pip package (see blender_addon/README.md),
and ``bmesh`` is the real compiled submodule bundled with it.
"""

from __future__ import annotations

from typing import Dict, Tuple

import bmesh
import bpy
import numpy as np
import pytest

from mask_luminance.scene import uv_bounds


def _make_quad_object(name: str, uv_layers: Dict[str, Tuple[float, float, float, float]]):
    """Builds a one-quad-face mesh Object.

    ``uv_layers`` maps a UV layer name to a ``(u0, v0, u1, v1)`` rect applied
    to that face's four loops, in insertion order (the first entry becomes
    bmesh's default "active" UV layer). An empty dict creates no UV layer at
    all.
    """
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    try:
        v_a = bm.verts.new((0.0, 0.0, 0.0))
        v_b = bm.verts.new((1.0, 0.0, 0.0))
        v_c = bm.verts.new((1.0, 1.0, 0.0))
        v_d = bm.verts.new((0.0, 1.0, 0.0))
        face = bm.faces.new((v_a, v_b, v_c, v_d))

        for layer_name, (u0, v0, u1, v1) in uv_layers.items():
            # bm.loops.layers.uv.new() invalidates previously fetched BMLoop
            # references, so face.loops must be re-fetched fresh after each
            # new layer is created, not reused from before it.
            layer = bm.loops.layers.uv.new(layer_name)
            for loop, uv in zip(face.loops, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]):
                loop[layer].uv = uv

        bm.to_mesh(mesh)
    finally:
        bm.free()

    return bpy.data.objects.new(name, mesh)


@pytest.fixture
def cleanup_objects():
    created = []
    yield created
    for obj in created:
        mesh = obj.data
        bpy.data.objects.remove(obj)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


@pytest.fixture(autouse=True)
def _clear_uv_bounds_cache():
    uv_bounds.clear_uv_bounds_cache()
    yield
    uv_bounds.clear_uv_bounds_cache()


def test_rasterize_uv_bounds_matches_known_quad(cleanup_objects):
    obj = _make_quad_object("centered_quad", {"UVMap": (0.25, 0.25, 0.75, 0.75)})
    cleanup_objects.append(obj)

    mask = uv_bounds.rasterize_uv_bounds(obj, (100, 100))

    assert mask.dtype == bool
    assert mask[50, 50]  # dead center: inside the UV footprint
    for corner in ((0, 0), (0, 99), (99, 0), (99, 99)):
        assert not mask[corner]


def test_out_of_bounds_mask_is_exact_inverse(cleanup_objects):
    obj = _make_quad_object("centered_quad", {"UVMap": (0.25, 0.25, 0.75, 0.75)})
    cleanup_objects.append(obj)

    inside = uv_bounds.rasterize_uv_bounds(obj, (64, 64))
    outside = uv_bounds.out_of_bounds_mask(obj, (64, 64))
    np.testing.assert_array_equal(outside, ~inside)


def test_v_is_flipped_to_match_top_down_array_convention(cleanup_objects):
    """UV v=0 (Blender-bottom) must land in the BOTTOM rows of the top-down array.

    Mirrors scene/images.py's documented bottom-up-Blender / top-down-array
    convention — getting this backwards would silently flip every correction
    vertically relative to the mesh.
    """
    obj = _make_quad_object("bottom_strip", {"UVMap": (0.0, 0.0, 1.0, 0.3)})
    cleanup_objects.append(obj)

    mask = uv_bounds.rasterize_uv_bounds(obj, (100, 100))

    assert mask[95, 50]  # near v=0 -> bottom row of the top-down array
    assert not mask[5, 50]  # near v=1 -> top row, outside this v in [0, 0.3] strip


def test_non_mesh_object_raises():
    empty = bpy.data.objects.new("not_a_mesh", None)
    try:
        with pytest.raises(ValueError):
            uv_bounds.rasterize_uv_bounds(empty, (32, 32))
    finally:
        bpy.data.objects.remove(empty)


def test_degenerate_resolution_raises(cleanup_objects):
    obj = _make_quad_object("quad", {"UVMap": (0.0, 0.0, 1.0, 1.0)})
    cleanup_objects.append(obj)

    with pytest.raises(ValueError):
        uv_bounds.rasterize_uv_bounds(obj, (0, 32))


def test_missing_uv_layer_raises(cleanup_objects):
    obj = _make_quad_object("no_uv", {})
    cleanup_objects.append(obj)

    with pytest.raises(ValueError):
        uv_bounds.rasterize_uv_bounds(obj, (32, 32))


def test_unknown_uv_layer_name_raises(cleanup_objects):
    obj = _make_quad_object("quad", {"UVMap": (0.0, 0.0, 1.0, 1.0)})
    cleanup_objects.append(obj)

    with pytest.raises(ValueError):
        uv_bounds.rasterize_uv_bounds(obj, (32, 32), uv_layer_name="DoesNotExist")


def test_named_uv_layer_selects_correct_layer(cleanup_objects):
    """A second, smaller UV layer must be picked when named explicitly, not the active/first one."""
    obj = _make_quad_object(
        "multi_uv_quad",
        {"Full": (0.0, 0.0, 1.0, 1.0), "Small": (0.25, 0.25, 0.75, 0.75)},
    )
    cleanup_objects.append(obj)

    full_mask = uv_bounds.rasterize_uv_bounds(obj, (100, 100), uv_layer_name="Full")
    small_mask = uv_bounds.rasterize_uv_bounds(obj, (100, 100), uv_layer_name="Small")
    default_mask = uv_bounds.rasterize_uv_bounds(obj, (100, 100))  # no name -> active ("Full", inserted first)

    assert full_mask.all()
    assert small_mask[50, 50]
    assert not small_mask[0, 0]
    np.testing.assert_array_equal(default_mask, full_mask)


def test_cache_hit_and_clear(cleanup_objects):
    obj = _make_quad_object("cached_quad", {"UVMap": (0.0, 0.0, 1.0, 1.0)})
    cleanup_objects.append(obj)

    first = uv_bounds.rasterize_uv_bounds(obj, (32, 32))
    second = uv_bounds.rasterize_uv_bounds(obj, (32, 32))
    assert first is second

    uv_bounds.clear_uv_bounds_cache()
    third = uv_bounds.rasterize_uv_bounds(obj, (32, 32))
    assert third is not first
    np.testing.assert_array_equal(first, third)


def test_clear_uv_bounds_cache_by_object_name_only_clears_that_object(cleanup_objects):
    obj_a = _make_quad_object("quad_a", {"UVMap": (0.0, 0.0, 1.0, 1.0)})
    obj_b = _make_quad_object("quad_b", {"UVMap": (0.0, 0.0, 1.0, 1.0)})
    cleanup_objects.extend([obj_a, obj_b])

    first_a = uv_bounds.rasterize_uv_bounds(obj_a, (32, 32))
    first_b = uv_bounds.rasterize_uv_bounds(obj_b, (32, 32))

    uv_bounds.clear_uv_bounds_cache(obj_a.name)

    second_a = uv_bounds.rasterize_uv_bounds(obj_a, (32, 32))
    second_b = uv_bounds.rasterize_uv_bounds(obj_b, (32, 32))

    assert second_a is not first_a  # invalidated
    assert second_b is first_b  # untouched
