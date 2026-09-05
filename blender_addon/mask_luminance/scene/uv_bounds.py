"""Rasterizes a mesh's real UV layout into a per-pixel boundary mask.

This is the addon's one dependency beyond ``bpy.types.Image`` pixel buffers:
every other module in ``scene/`` only ever touches Image datablocks, but
fixing correction spilling past a real UV island edge (rather than the
"near-black = background" luminance guess ``core.blend`` otherwise falls back
to) needs actual mesh/UV geometry. This logic cannot live in ``core/`` —
that package is deliberately kept free of any ``bpy``/``bmesh`` dependency
(see ``core/infill.py``'s vendoring rationale) — so it lives here instead,
producing a plain boolean numpy array that ``core.blend`` consumes as an
opaque ``out_of_bounds`` mask with no further mesh knowledge required.

Follows ``scene/images.py``'s conventions: top-down arrays (row 0 = top),
an explicit vertical flip at the bpy boundary, and a ``ValueError`` guard for
degenerate input.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import bpy
import bmesh
import cv2
import numpy as np


def rasterize_uv_bounds(
    obj: bpy.types.Object,
    resolution: Tuple[int, int],
    uv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Returns a top-down bool ``(H, W)`` array: True where a real UV triangle covers that texel.

    ``resolution`` is ``(height, width)``, matching every other array
    convention in this addon. ``uv_layer_name`` selects a specific UV layer;
    ``None`` (default) uses the mesh's active UV layer.

    UVs are read straight from ``obj.data`` (mesh-local topology) rather than
    a depsgraph-evaluated mesh — UV coordinates are static per-loop data that
    a pose/modifier stack does not move, so no evaluation is needed. Cached
    by ``(obj.name, uv_layer_name, resolution)`` — see ``clear_uv_bounds_cache``
    for the same staleness caveat ``scene/bake.py``'s preview cache documents
    (keyed on name, not mesh content).

    Benchmarked on a 200x200 grid (40,000 quad faces, 80,000 triangles after
    triangulation): ~450-500ms per call, essentially independent of
    ``resolution`` (384px and 2048px both landed in that range) — the cost is
    dominated by the per-triangle Python loop reading loop UVs, not by
    ``cv2.fillPoly`` itself. A cache hit is ~0.03ms. In practice this means a
    one-time pause the first time a given (object, UV layer, resolution)
    combination is rasterized — e.g. right after picking a UV source object,
    or the first live-preview tick at a new preview resolution — and free
    afterward.
    """
    height, width = resolution
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid rasterization resolution: {resolution!r}")
    if obj.type != "MESH":
        raise ValueError(f"Object '{obj.name}' is not a mesh (type={obj.type!r}).")

    key = (obj.name, uv_layer_name, (height, width))
    cached = _uv_bounds_cache.get(key)
    if cached is not None:
        return cached

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])

        uv_layer = bm.loops.layers.uv.get(uv_layer_name) if uv_layer_name else bm.loops.layers.uv.active
        if uv_layer is None:
            available = [layer.name for layer in bm.loops.layers.uv.values()]
            raise ValueError(
                f"Mesh '{obj.data.name}' has no UV layer"
                + (f" named '{uv_layer_name}'" if uv_layer_name else "")
                + f" (available: {available})."
            )

        triangles = []
        for face in bm.faces:
            pts = np.empty((len(face.loops), 2), dtype=np.int32)
            for i, loop in enumerate(face.loops):
                u, v = loop[uv_layer].uv
                pts[i, 0] = round(u * width)
                pts[i, 1] = round((1.0 - v) * height)  # bottom-up UV V -> top-down row
            triangles.append(pts)
    finally:
        bm.free()

    canvas = np.zeros((height, width), dtype=np.uint8)
    if triangles:
        cv2.fillPoly(canvas, triangles, 1)

    mask = canvas.astype(bool)
    _uv_bounds_cache[key] = mask
    return mask


def out_of_bounds_mask(
    obj: bpy.types.Object,
    resolution: Tuple[int, int],
    uv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """The logical inverse of ``rasterize_uv_bounds`` — True where NOT covered by real UV geometry."""
    return ~rasterize_uv_bounds(obj, resolution, uv_layer_name)


# (object.name, uv_layer_name, (height, width)) -> rasterized bool mask.
_uv_bounds_cache: Dict[Tuple[str, Optional[str], Tuple[int, int]], np.ndarray] = {}


def clear_uv_bounds_cache(object_name: Optional[str] = None) -> None:
    """Drops cached rasterizations. With no argument, clears every entry.

    Known limitation (same as ``scene/bake.py``'s ``_cached_downsampled_rgb``):
    keyed by Object *name*, not mesh content, so editing UVs in place on the
    same Object won't be picked up until this is called.
    """
    if object_name is None:
        _uv_bounds_cache.clear()
        return
    for key in [k for k in _uv_bounds_cache if k[0] == object_name]:
        del _uv_bounds_cache[key]
