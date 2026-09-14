"""Bake an eyebrows+lips feature layer from a diffuse multiview render set into UV space.

Pure pipeline logic (no Qt) — importable, scriptable, and independently testable, mirroring
the matte_luminance_blend.py / matte_luminance_ui.py split elsewhere in this repo.

Pipeline (validated against test_textures/african_female_0003 this session):
    1. Parse precomputed 2D landmarks per view (OpenPose-style JSON, 100 points matching
       genies' FACIAL_LANDMARKS_NAMES order).
    2. Rasterize an eyebrows+lips mask per view directly from those 2D landmarks.
    3. Load the shared 3D landmark template (a USD Points prim) and approximately transfer it
       onto the character's own head mesh (the template is defined on a generic whole-body
       mesh, not on any specific character's head topology).
    4. Bake the per-view masks into UV space via multiview_gen.tm_texture_from_images, using
       the same landmark-based alignment already validated for diffuse-photo baking.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image
from pxr import Usd, UsdGeom

from genies.meshutils.geometry.registration.landmarks.constants import FACIAL_LANDMARKS_NAMES
from genies.meshutils.geometry.registration.landmarks.landmarks_utils import convert_landmarks_names

from multiview_gen import tm_texture_from_images

VIEW_LABELS = ("front", "left", "right")


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------
def discover_character_assets(folder: str) -> Dict[str, Optional[str]]:
    """Best-effort discovery of the fixed set of inputs this pipeline needs.

    Looks for ``*_mvDiff_<front|left|right>.png`` (+ matching ``_landmark.json`` by suffix
    convention) and the first ``*.glb`` in ``folder``. Missing pieces come back as None rather
    than raising — the caller (UI) decides how to surface gaps; discovery only prefills.
    """
    result: Dict[str, Optional[str]] = {
        "front_image": None, "front_landmarks": None,
        "left_image": None, "left_landmarks": None,
        "right_image": None, "right_landmarks": None,
        "head_glb": None,
    }
    if not folder or not os.path.isdir(folder):
        return result

    for label in VIEW_LABELS:
        matches = sorted(glob.glob(os.path.join(folder, f"*_mvDiff_{label}.png")))
        if matches:
            img_path = matches[0]
            result[f"{label}_image"] = img_path
            base, _ = os.path.splitext(img_path)
            lm_path = f"{base}_landmark.json"
            if os.path.isfile(lm_path):
                result[f"{label}_landmarks"] = lm_path

    glb_matches = sorted(glob.glob(os.path.join(folder, "*.glb")))
    if glb_matches:
        result["head_glb"] = glb_matches[0]

    return result


# ---------------------------------------------------------------------------
# 2D landmarks
# ---------------------------------------------------------------------------
def load_openpose_landmarks(json_path: str) -> Dict[str, np.ndarray]:
    """Parses an OpenPose-style face_keypoints_2d JSON into {landmark_name: [x, y]} (normalized, y-down).

    Point order matches genies' FACIAL_LANDMARKS_NAMES exactly (validated against this
    session's test asset) — the third value per point is a visibility flag, not used here.
    """
    with open(json_path) as f:
        d = json.load(f)
    kp = d["people"][0]["face_keypoints_2d"]
    pts = np.array(kp).reshape(-1, 3)
    if len(pts) != len(FACIAL_LANDMARKS_NAMES):
        raise ValueError(
            f"{json_path}: {len(pts)} landmark points, expected {len(FACIAL_LANDMARKS_NAMES)}"
        )
    return {name: pts[i, :2] for i, name in enumerate(FACIAL_LANDMARKS_NAMES)}


def build_view_data(view_files: Dict[str, Tuple[str, str]]) -> Tuple[Dict[str, Any], List[str]]:
    """Builds the view_data/view_tokens shape multiview_gen expects.

    Args:
        view_files: {view_token: (image_path, landmark_json_path)}. Tokens must already be in
            the "y_<angle>[_neg]" form multiview_gen's solver requires (e.g. "y_0", "y_45_neg").
    """
    view_data: Dict[str, Any] = {}
    view_tokens: List[str] = []
    for token, (img_path, lm_path) in view_files.items():
        img_path = img_path.replace("\\", "/")
        landmark_data = load_openpose_landmarks(lm_path)

        with Image.open(img_path) as im:
            w, h = im.width, im.height

        names_raw, pos_px, pos_norm = [], [], []
        for name, norm_coords in landmark_data.items():
            names_raw.append(name)
            nx, ny_up = norm_coords[0], 1.0 - norm_coords[1]
            pos_norm.append([nx, ny_up])
            pos_px.append([nx * w, norm_coords[1] * h])

        view_data[token] = {
            "image_path": img_path,
            "names_raw": names_raw,
            "names": convert_landmarks_names(names_raw, usd_to_maya=True),
            "pos_px": np.array(pos_px),
            "pos": np.array(pos_norm),
            "width": w,
            "height": h,
        }
        view_tokens.append(token)
    return view_data, view_tokens


# ---------------------------------------------------------------------------
# Eyebrows + lips rasterization
# ---------------------------------------------------------------------------
FEATURE_GROUPS = ("right_brow", "left_brow", "lip")


def rasterize_feature_mask(
    names: List[str],
    pos_px: np.ndarray,
    width: int,
    height: int,
    feather_px: int = 6,
    groups: Tuple[str, ...] = FEATURE_GROUPS,
) -> np.ndarray:
    """Fills convex hulls over the requested landmark groups onto a blank RGBA canvas.

    ``groups`` selects which of FEATURE_GROUPS to rasterize — the default renders eyebrows
    and lips together (the original "eyebrows and lips" feature the multiview render was meant
    to isolate); pass e.g. ``("lip",)`` to isolate just the lips. RGB is the actual signal (255
    = feature, 0 = skin); alpha marks which pixels were actually observed by this camera view at
    all (a dilated hull over the *full* landmark set, i.e. an approximate face silhouette), 0
    outside it.

    The alpha channel matters, not just the RGB: TmBakeTextureFromImages only runs its own
    photograph-oriented background-removal floodfill when the source image has no meaningful
    alpha of its own (genies/meshutils/shading/texture_from_images.py:554-560 — floodfill is
    skipped whenever the image already carries a non-fully-opaque alpha channel). Without an
    explicit alpha here, that floodfill treats our near-black "not a feature" pixels the same as
    a photo's black backdrop, discards nearly the whole canvas as background, and the bake's
    extrapolation step (meant only to fill small camera-blind-spot gaps) then smears the tiny
    remaining bright brow/lip islands across the entire UV face region. Supplying our own alpha
    (valid = actually within this view's face silhouette) sidesteps that entirely.
    """
    rgb = np.zeros((height, width), dtype=np.uint8)
    by_name = dict(zip(names, pos_px))

    group_members = {
        "right_brow": [n for n in names if "right_brow" in n],
        "left_brow": [n for n in names if "left_brow" in n],
        "lip": [n for n in names if "_lip_" in n],
    }
    for key in groups:
        group_names = group_members[key]
        if len(group_names) < 3:
            continue
        pts = np.array([by_name[n] for n in group_names], dtype=np.float32)
        hull = cv2.convexHull(pts).astype(np.int32)
        cv2.fillConvexPoly(rgb, hull, 255)

    alpha = np.zeros((height, width), dtype=np.uint8)
    all_pts = np.array(pos_px, dtype=np.float32)
    if len(all_pts) >= 3:
        face_hull = cv2.convexHull(all_pts).astype(np.int32)
        cv2.fillConvexPoly(alpha, face_hull, 255)
        margin = max(feather_px, 1) * 3
        alpha = cv2.dilate(alpha, np.ones((margin, margin), np.uint8))

    if feather_px > 0:
        k = feather_px * 2 + 1
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)

    return np.dstack([rgb, rgb, rgb, alpha])


def _write_feature_masks(
    view_data: Dict[str, Any],
    view_tokens: List[str],
    out_dir: str,
    feather_px: int,
    groups: Tuple[str, ...] = FEATURE_GROUPS,
) -> None:
    """Rasterizes each view's feature mask and repoints view_data's image_path at it in place.

    image_path is stored as an absolute path: tm_texture_from_images is called below with
    input_images_dir="" (see bake_feature_layer) specifically so genies doesn't re-join it with
    another base directory — TmBakeTextureFromImages unconditionally joins concept_images_dir
    with each view's path when concept_images_dir is non-empty, which corrupts an
    already-complete relative path (genies/meshutils/shading/texture_from_images.py:169).
    """
    os.makedirs(out_dir, exist_ok=True)
    for token in view_tokens:
        data = view_data[token]
        mask = rasterize_feature_mask(
            data["names"], data["pos_px"], data["width"], data["height"], feather_px, groups
        )
        mask_path = os.path.abspath(os.path.join(out_dir, f"feature_mask_{token}.png")).replace("\\", "/")
        cv2.imwrite(mask_path, mask)
        data["image_path"] = mask_path


# ---------------------------------------------------------------------------
# 3D landmark template + transfer
# ---------------------------------------------------------------------------
def load_usd_landmarks(landmarks_usd_path: str, variant: str = "coco_extended") -> Tuple[List[str], np.ndarray]:
    """Reads the shared 3D landmark template (USD Points prim, R_/L_/C_-prefixed names)."""
    stage = Usd.Stage.Open(landmarks_usd_path)
    prim = stage.GetPrimAtPath("/landmarks")
    prim.GetVariantSets().GetVariantSet("landmark_set").SetVariantSelection(variant)
    usd_names = list(prim.GetAttribute("names").Get())
    usd_points = np.array(UsdGeom.Points(prim).GetPointsAttr().Get())
    return convert_landmarks_names(usd_names, usd_to_maya=True), usd_points


def load_head_mesh(glb_path: str) -> trimesh.Trimesh:
    """Loads a character glb and returns the geometry node whose name contains 'head'."""
    scene = trimesh.load(glb_path, process=False)
    head_node = next(n for n in scene.graph.nodes_geometry if "head" in n.lower())
    _, geom_key = scene.graph[head_node]
    return scene.geometry[geom_key]


def transfer_landmarks_to_head(usd_points: np.ndarray, head_mesh: trimesh.Trimesh) -> np.ndarray:
    """Approximate transfer of the generic template's landmark points onto a specific head mesh.

    APPROXIMATION, not a proper wrap/refit: the 3D landmark template lives in the rest-pose
    space of a generic whole-body mesh (not the same topology/space as any specific character's
    head). This remaps axes (derived from the glb's baked node transform: local +X -> world +X,
    local -Z -> world +Y/up, local +Y -> world +Z/forward), bbox-fits the point cloud into the
    head mesh's local bounding box, then snaps each point to the mesh surface. Good enough for a
    recognizable feature bake; a genies.meshutils wrap/refit-based registration (see wrap.py /
    refit.py in this repo) would be more correct for production use.
    """
    local_v = head_mesh.vertices
    remapped_local = np.column_stack([local_v[:, 0], -local_v[:, 2], local_v[:, 1]])

    src_min, src_max = usd_points.min(axis=0), usd_points.max(axis=0)
    tgt_min, tgt_max = remapped_local.min(axis=0), remapped_local.max(axis=0)
    scale = (tgt_max - tgt_min) / (src_max - src_min)
    approx_remapped = (usd_points - src_min) * scale + tgt_min

    approx_local = np.column_stack([
        approx_remapped[:, 0],
        approx_remapped[:, 2],
        -approx_remapped[:, 1],
    ])

    closest, _distance, _triangle_id = trimesh.proximity.closest_point(head_mesh, approx_local)
    return closest


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def bake_feature_layer(
    view_files: Dict[str, Tuple[str, str]],
    template_landmarks_usd: str,
    landmarks_variant: str,
    glb_path: str,
    output_dir: str,
    output_image_name: str,
    output_size: int = 1024,
    feather_px: int = 6,
    feature_groups: Tuple[str, ...] = FEATURE_GROUPS,
) -> str:
    """Bakes a feature UV mask from a diffuse multiview render set.

    Args:
        view_files: {view_token: (diffuse_image_path, landmark_json_path)}, tokens already in
            "y_<angle>[_neg]" form (e.g. {"y_0": (...), "y_45_neg": (...), "y_45": (...)}).
        template_landmarks_usd: path to the shared 3D landmark template (analysis/landmarks.usd).
        landmarks_variant: "coco" or "coco_extended".
        glb_path: the character's head mesh.
        output_dir: where the baked mask (and intermediate per-view masks) are written.
        output_image_name: filename for the baked UV mask.
        feature_groups: which of FEATURE_GROUPS to include, e.g. ("lip",) for a lips-only mask
            instead of the default combined eyebrows+lips mask.

    Returns:
        Absolute path to the baked UV mask PNG.
    """
    view_data, view_tokens = build_view_data(view_files)

    masks_dir = os.path.join(output_dir, "feature_masks")
    _write_feature_masks(view_data, view_tokens, masks_dir, feather_px, feature_groups)

    lm_names, lm_points = load_usd_landmarks(template_landmarks_usd, landmarks_variant)
    head_mesh = load_head_mesh(glb_path)
    head_lm_points = transfer_landmarks_to_head(lm_points, head_mesh)
    template_3d_landmarks = (lm_names, head_lm_points)

    return tm_texture_from_images(
        mesh_tm=head_mesh,
        view_data=view_data,
        template_3d_landmarks=template_3d_landmarks,
        view_tokens=view_tokens,
        input_images_dir="",
        input_masks_dir="",
        output_dir=output_dir,
        output_image_name=output_image_name,
        main_color_mask_file=None,
        extrapolation_mask_file=None,
        color_composite_file=None,
        output_size=output_size,
    )
