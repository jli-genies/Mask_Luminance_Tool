"""Detect face features (lips, eyebrows) directly on a UV-space texture.

Runs a real face-landmark model on the 2D texture image itself, instead of the 3D
multiview route in multiview_feature_bake.py. That works here because this project's UV
layout places the face as a roughly frontal projection in its own island — so a model
trained on ordinary photos can be pointed straight at that island crop and produce a mask
that's already in UV space, with no 3D landmark transfer (and its misregistration risk,
confirmed in practice: bake_feature_layer's approximate mesh-surface snapping lit up the
wrong UV islands entirely instead of the mouth).

This is landmark-based, not pixel-wise semantic segmentation: MediaPipe FaceMesh regresses
478 face landmark points, and a mask is built by convex-hulling the points that belong to a
named feature group (its own per-feature grouping of connections, e.g. FACEMESH_LIPS) — the
same grouping approach multiview_feature_bake.rasterize_feature_mask used, just sourced from
one FaceMesh pass over the texture instead of per-view OpenPose landmarks + a 3D transfer.

Pure pipeline logic (no Qt) — importable, scriptable, and independently testable, mirroring
the matte_luminance_blend.py / matte_luminance_ui.py split elsewhere in this repo.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# UV atlases pack disconnected islands (face, ears, back-of-head, ...) onto one canvas
# separated by empty (black) space. Skip slivers not worth trying to detect a face in, and
# cap how many islands we try the model against (largest first).
_MIN_ISLAND_AREA = 4000
_MAX_ISLANDS_TRIED = 5

# Named feature groups -> the FaceMesh connection set each is built from. Each group gets its
# own convex hull (a single hull across e.g. "lips" + "left_eyebrow" would wrongly fill
# everything between the mouth and the eyebrow).
FEATURE_GROUPS = ("lips", "left_eyebrow", "right_eyebrow")
_FEATURE_CONNECTION_ATTRS = {
    "lips": "FACEMESH_LIPS",
    "left_eyebrow": "FACEMESH_LEFT_EYEBROW",
    "right_eyebrow": "FACEMESH_RIGHT_EYEBROW",
}


def find_islands(texture: np.ndarray, min_area: int = _MIN_ISLAND_AREA) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes (x, y, w, h) of the texture's non-empty UV islands, largest first."""
    gray = cv2.cvtColor(texture[..., :3], cv2.COLOR_RGB2GRAY) if texture.ndim == 3 else texture
    nonblank = (gray > 8).astype(np.uint8)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(nonblank, connectivity=8)
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    boxes.sort(key=lambda b: -(b[2] * b[3]))
    return boxes


def detect_landmarks(image_rgb: np.ndarray, min_detection_confidence: float = 0.3) -> Optional[np.ndarray]:
    """Runs MediaPipe FaceMesh on one image; returns all landmark pixel coords, or None."""
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh
    h, w = image_rgb.shape[:2]
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
    ) as mesh:
        result = mesh.process(image_rgb)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)


def _feature_point_indices(feature: str) -> List[int]:
    import mediapipe as mp

    connections = getattr(mp.solutions.face_mesh, _FEATURE_CONNECTION_ATTRS[feature])
    return sorted(set(i for pair in connections for i in pair))


def bake_feature_mask(
    texture: np.ndarray,
    features: Sequence[str] = ("lips",),
    feather_px: int = 6,
) -> np.ndarray:
    """Finds the face island in a UV texture and rasterizes the requested features' mask.

    ``features`` are names from FEATURE_GROUPS, e.g. ``("lips",)``, ``("left_eyebrow",
    "right_eyebrow")``, or all three combined — each gets its own convex hull, unioned onto
    one mask. Tries each UV island (largest first) until the face model actually detects a
    face in it — rather than assuming the face is always the single largest island, since a
    different UV layout could pack things differently.

    Returns a single-channel uint8 mask (255 = feature, feathered edges, 0 elsewhere) at the
    same resolution as ``texture``. Raises ValueError if no face is found in any island.
    """
    h, w = texture.shape[:2]
    for (x, y, iw, ih) in find_islands(texture)[:_MAX_ISLANDS_TRIED]:
        all_pts = detect_landmarks(texture[y:y + ih, x:x + iw])
        if all_pts is None:
            continue
        offset = np.array([x, y], dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        for feature in features:
            idx = _feature_point_indices(feature)
            hull = cv2.convexHull((all_pts[idx] + offset).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        if feather_px > 0:
            k = feather_px * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask
    raise ValueError("No face detected in any UV island of this texture.")
