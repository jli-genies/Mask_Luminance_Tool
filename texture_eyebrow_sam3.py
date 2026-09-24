"""Bake feature masks on a UV-space texture via GenieSAM's fine-tuned SAM3 text-prompt
segmentation, as a drop-in alternative to texture_face_segment.bake_feature_mask's landmark
convex-hull masks.

GenieSAM (github.com/geniesinc/GenieSAM, a separate repo) already defines "Left Eyebrow" /
"Right Eyebrow" as text-prompt categories for its own SAM3 fine-tune — segmenting the actual
painted eyebrow shape directly, with no landmark seeding needed (unlike SAM1/2, SAM3 is
prompted by text, not points/boxes, so a convex-hull-shaped prompt region is never a ceiling
on the output shape the way it is for texture_face_segment's hull masks). GenieSAM's own
dependencies (torch, a git-installed `sam3` package, opencv<4.10) conflict with this
project's pins and are already set up in a separate `geniesam` conda env, so this module
shells out to sam3_bridge.py running under that env's own python rather than importing
GenieSAM in-process.

Pure pipeline logic (no Qt) — mirrors texture_face_segment.py's split from Qt glue.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from matte_luminance_blend import load_rgb, save_rgb
from texture_face_segment import _MAX_ISLANDS_TRIED, detect_landmarks, find_islands

_BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_bridge.py")

# texture_face_segment feature names -> GenieSAM's text-prompt category strings, so the
# existing FEATURE_CHOICES combo in the UI works unchanged against either backend.
_FEATURE_TO_CATEGORIES: Dict[str, List[str]] = {
    "lips": ["Upper Lip", "Lower Lip"],
    "left_eyebrow": ["Left Eyebrow"],
    "right_eyebrow": ["Right Eyebrow"],
    "beard": ["Beard"],
}

Box = Tuple[int, int, int, int]


def _categories_for(features: Sequence[str]) -> List[str]:
    categories: List[str] = []
    for feature in features:
        categories.extend(_FEATURE_TO_CATEGORIES[feature])
    return categories


def _safe_filename(category: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in category) + ".png"


def _find_face_island(texture: np.ndarray) -> Box:
    """Same multi-island search as bake_feature_mask: try islands largest-first, keep the
    first one a face model actually detects a face in (MediaPipe is only used here to pick
    the right island — SAM3 itself needs no landmark points to segment a text-prompt category).
    """
    for box in find_islands(texture)[:_MAX_ISLANDS_TRIED]:
        x, y, iw, ih = box
        if detect_landmarks(texture[y:y + ih, x:x + iw]) is not None:
            return box
    raise ValueError("No face detected in any UV island of this texture.")


def _crop_face_island(texture: np.ndarray) -> Tuple[np.ndarray, Box]:
    box = _find_face_island(texture)
    x, y, iw, ih = box
    return texture[y:y + ih, x:x + iw], box


def _union_categories(out_dir: str, categories: Sequence[str], shape_hw: Tuple[int, int]) -> np.ndarray:
    island_mask = np.zeros(shape_hw, dtype=np.uint8)
    for category in categories:
        mask_path = os.path.join(out_dir, _safe_filename(category))
        if os.path.isfile(mask_path):
            island_mask = np.maximum(island_mask, cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
    return island_mask


def _paste_and_feather(shape_hw: Tuple[int, int], island_mask: np.ndarray, box: Box, feather_px: int) -> np.ndarray:
    h, w = shape_hw
    x, y, iw, ih = box
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y:y + ih, x:x + iw] = island_mask
    if feather_px > 0:
        k = feather_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _run_bridge(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SAM3 bridge failed:\n{proc.stderr or proc.stdout}")


def _beard_threshold_args(beard_score_threshold: Optional[float]) -> List[str]:
    return ["--beard-score-threshold", str(beard_score_threshold)] if beard_score_threshold is not None else []


def bake_feature_mask_sam3(
    texture: np.ndarray,
    features: Sequence[str],
    feather_px: int = 6,
    *,
    geniesam_repo: str,
    sam3_checkpoint: str,
    geniesam_python: str,
    device: str = "cuda",
    image_size: int = 1008,
    beard_score_threshold: Optional[float] = None,
) -> np.ndarray:
    """Finds the face island in a UV texture and rasterizes the requested features' mask via
    GenieSAM's SAM3 text-prompt segmentation, run out-of-process in the `geniesam` conda env.

    ``features`` uses the same vocabulary as texture_face_segment.bake_feature_mask (e.g.
    ``("left_eyebrow", "right_eyebrow")``). Returns a single-channel uint8 mask (255 =
    feature, feathered edges, 0 elsewhere) at the same resolution as ``texture`` — a drop-in
    replacement for bake_feature_mask's output, so texture_segment.segment_texture needs no
    changes to consume it. Raises ValueError if no face island is found, RuntimeError if the
    subprocess call into the geniesam env fails.

    ``beard_score_threshold``, if given, overrides GenieSAM's own config.yaml beard_score for
    this call only — lower it when "beard" detection is missing hair on darker skin textures
    (SAM3's confidence there is naturally lower, since hair/skin color contrast is smaller).
    Only affects the "beard" category; ignored for other features.

    For more than a handful of textures, use bake_feature_masks_sam3_batch instead — calling
    this in a loop reloads the ~3GB SAM3 checkpoint (a fresh subprocess) for every texture.
    """
    categories = _categories_for(features)
    crop, box = _crop_face_island(texture)

    with tempfile.TemporaryDirectory() as tmp_dir:
        crop_path = os.path.join(tmp_dir, "crop.png")
        save_rgb(crop_path, crop)
        out_dir = os.path.join(tmp_dir, "out")
        _run_bridge([
            geniesam_python, _BRIDGE_SCRIPT,
            "--image", crop_path,
            "--output-dir", out_dir,
            "--checkpoint", sam3_checkpoint,
            "--geniesam-repo", geniesam_repo,
            "--categories", *categories,
            "--device", device,
            "--image-size", str(image_size),
            *_beard_threshold_args(beard_score_threshold),
        ])
        island_mask = _union_categories(out_dir, categories, crop.shape[:2])

    return _paste_and_feather(texture.shape[:2], island_mask, box, feather_px)


def bake_feature_masks_sam3_batch(
    texture_paths: Sequence[str],
    features: Sequence[str],
    feather_px: int = 6,
    *,
    geniesam_repo: str,
    sam3_checkpoint: str,
    geniesam_python: str,
    device: str = "cuda",
    image_size: int = 1008,
    beard_score_threshold: Optional[float] = None,
    on_crop_done: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Batch counterpart to bake_feature_mask_sam3: crops every texture's face island, then
    runs ONE sam3_bridge.py subprocess (one SAM3 checkpoint load, reused for every texture)
    instead of one subprocess per texture — the checkpoint load is the dominant cost, so this
    is far faster than calling bake_feature_mask_sam3 in a loop for more than a couple files.

    ``beard_score_threshold``: see bake_feature_mask_sam3 — applied to the whole batch.

    ``on_crop_done(index, total, texture_path)`` fires after each texture's crop step (which
    still runs one-by-one in this process, since it needs MediaPipe to pick the face island),
    before the single shared SAM3 subprocess call.

    Returns ``(masks_by_texture_path, no_face_paths)``: ``masks_by_texture_path`` covers every
    texture where a face island was found AND at least one requested category was detected;
    ``no_face_paths`` lists textures where no face island was found at all. A texture that had
    a face island but no detected category (e.g. below GenieSAM's score threshold) appears in
    neither — the caller can treat "not in either" as that distinct failure mode.
    """
    categories = _categories_for(features)
    total = len(texture_paths)

    with tempfile.TemporaryDirectory() as tmp_dir:
        crops_dir = os.path.join(tmp_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        out_root = os.path.join(tmp_dir, "out")

        entries: Dict[str, dict] = {}
        no_face: List[str] = []
        for i, texture_path in enumerate(texture_paths):
            texture = load_rgb(texture_path)
            try:
                crop, box = _crop_face_island(texture)
            except ValueError:
                no_face.append(texture_path)
            else:
                stem = f"{i:04d}"
                save_rgb(os.path.join(crops_dir, f"{stem}.png"), crop)
                entries[stem] = {
                    "texture_path": texture_path,
                    "box": box,
                    "crop_hw": crop.shape[:2],
                    "full_hw": texture.shape[:2],
                }
            if on_crop_done:
                on_crop_done(i + 1, total, texture_path)

        results: Dict[str, np.ndarray] = {}
        if entries:
            _run_bridge([
                geniesam_python, _BRIDGE_SCRIPT,
                "--image-dir", crops_dir,
                "--output-root", out_root,
                "--checkpoint", sam3_checkpoint,
                "--geniesam-repo", geniesam_repo,
                "--categories", *categories,
                "--device", device,
                "--image-size", str(image_size),
                *_beard_threshold_args(beard_score_threshold),
            ])
            for stem, entry in entries.items():
                island_mask = _union_categories(os.path.join(out_root, stem), categories, entry["crop_hw"])
                if island_mask.any():
                    results[entry["texture_path"]] = _paste_and_feather(
                        entry["full_hw"], island_mask, entry["box"], feather_px
                    )

    return results, no_face
