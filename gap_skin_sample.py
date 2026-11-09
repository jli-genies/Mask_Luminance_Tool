"""Skin color sampled the way genies-gap-core's autorigger does it.

``task_texture_analysis`` averages the head texture inside the faces touching the head
template's "sampling" GeomSubset family (``sample_skin``: two forehead bands and an arc on the
scalp island) via ``genies.gap.utils.image_processing.sample_average_color_from_uvs``. The
authored body albedos share the head template's UV layout, so the same region lands on clean
forehead/scalp skin here too.

gap-core isn't installed in this env (it would fight the mediapipe pin), so its
``image_processing.py`` is loaded straight from the repo checkout; it only needs pxr, PIL,
numpy and cv2. Point ``GAP_CORE_ROOT`` at the checkout if it isn't the sibling folder.
"""
from __future__ import annotations

import functools
import importlib.util
import os
from typing import List

import numpy as np
from PIL import Image

GAP_CORE_ROOT = os.environ.get(
    "GAP_CORE_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "genies-gap-core"),
)
_GAP_SRC = os.path.join(GAP_CORE_ROOT, "src", "genies", "gap")
HEAD_TEMPLATE = os.path.join(_GAP_SRC, "assets", "templates", "components", "head", "humanoid", "head.usd")
HEAD_GEO_PATH = "/head/geo/head_geo"
SAMPLING_SUBSETS_FAMILY = "sampling"  # cconstants.SAMPLING_SUBSETS_FAMILY


@functools.lru_cache(maxsize=1)
def _image_processing():
    """gap-core's ``genies.gap.utils.image_processing``, loaded from its file."""
    path = os.path.join(_GAP_SRC, "utils", "image_processing.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"genies-gap-core not found at {GAP_CORE_ROOT} (set GAP_CORE_ROOT).")
    spec = importlib.util.spec_from_file_location("_gap_image_processing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def sample_skin_uv_faces() -> List[np.ndarray]:
    """UV polygons of the head template's skin sampling region, as the autorigger builds them."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(HEAD_TEMPLATE)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(HEAD_GEO_PATH))
    subsets = UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(mesh.GetPrim()), familyName=SAMPLING_SUBSETS_FAMILY)
    indices = sorted({int(i) for s in subsets for i in s.GetIndicesAttr().Get()})
    if not indices:
        raise LookupError(f"No '{SAMPLING_SUBSETS_FAMILY}' subsets on {HEAD_GEO_PATH} in {HEAD_TEMPLATE}.")
    interpolation = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").GetInterpolation()
    return _image_processing().get_uvs_from_vertices(mesh, indices, interpolation)


def sample_gap_skin_color(sample: np.ndarray) -> np.ndarray:
    """Mean RGB (float32, shape (3,)) of ``sample`` over gap-core's skin sampling region.

    Raises LookupError if the region covers only background (e.g. a different UV layout).
    """
    rgb = _image_processing().sample_average_color_from_uvs(
        Image.fromarray(np.ascontiguousarray(sample, dtype=np.uint8)), sample_skin_uv_faces()
    )
    if rgb is None or float(np.mean(rgb)) < 8.0:
        raise LookupError("gap-core's skin sampling region is empty on this texture.")
    return np.asarray(rgb, dtype=np.float32)
