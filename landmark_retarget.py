"""Warp an mvDiff render onto its mvConcept counterpart's facial landmark layout.

Pure pipeline logic (no Qt) -- importable, scriptable, and independently testable, mirroring
the matte_luminance_blend.py / multiview_feature_bake.py split elsewhere in this repo.

*_mvConcept_<view>_landmark.json and *_mvDiff_<view>_landmark.json share the same
FACIAL_LANDMARKS_NAMES order point-for-point (see multiview_feature_bake.load_openpose_landmarks),
so retargeting needs no correspondence/matching step: it's a direct thin-plate-spline warp from
one named point set onto the other, applied to the diffuse image as a backward map (cv2.remap)
so the diffuse texture's appearance is preserved while its proportions follow the concept.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from genies.meshutils.geometry.rbf import rbf_deform

from multiview_feature_bake import discover_character_assets, load_openpose_landmarks

VIEW_LABELS = ("front", "left", "right")


def _landmarks_to_pixels(landmarks: Dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    """{name: [x, y]} normalized (0-1, y-down) -> (N, 2) pixel array, in dict insertion order."""
    norm = np.array(list(landmarks.values()), dtype=np.float64)
    return norm * np.array([width, height], dtype=np.float64)


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """Least-squares isotropic scale + translation mapping source -> target (no rotation).

    The mvConcept and mvDiff camera rigs turned out to use different zoom/crop per character
    (confirmed on african_female_0003: diffuse landmarks span ~59% of canvas width vs concept's
    ~50%) -- a coarse framing difference, not a facial-proportion one. Left for the per-point TPS
    to absorb alone, that scale gap dominates the field and is least constrained exactly where
    there are no landmarks (the bald scalp above the brows), which is what previously showed up
    as a stretched-looking forehead. Removing it first as a plain scale+translation (well-behaved
    everywhere, no extrapolation risk) leaves the TPS to fit only the much smaller residual --
    average per-landmark deviation drops from ~40px to ~5.6px (at 1024x1024) once this is
    removed. No rotation term: the fixed front/left/right rig has no camera roll to correct for.

    Returns: (scale, translation) such that target ~= source * scale + translation.
    """
    src_mean, tgt_mean = source.mean(axis=0), target.mean(axis=0)
    src_c, tgt_c = source - src_mean, target - tgt_mean
    scale = float(np.sqrt((tgt_c ** 2).sum() / (src_c ** 2).sum()))
    translation = tgt_mean - scale * src_mean
    return scale, translation


def _boundary_anchor_points(width: int, height: int, points_per_side: int = 4) -> np.ndarray:
    """A ring of points along the canvas edge, for pinning the spline outside the face region.

    Facial landmarks only cover eyes/brows/lips/jaw/contour -- they say nothing about the bald
    scalp or background filling the rest of the canvas. Thin-plate splines extrapolate that far
    outside their control points essentially unconstrained, which folds those regions onto
    themselves (observed as a stray warped fragment above the head in an unpinned test run).
    Anchoring the border to itself (identity, zero displacement) keeps the warp local to the
    face and lets the untouched scalp/background fade in smoothly toward the edges.
    """
    ts = np.linspace(0.0, 1.0, points_per_side, endpoint=False)
    xs, ys = ts * width, ts * height
    top = np.stack([xs, np.zeros_like(xs)], axis=1)
    bottom = np.stack([xs, np.full_like(xs, height - 1)], axis=1)
    left = np.stack([np.zeros_like(ys), ys], axis=1)
    right = np.stack([np.full_like(ys, width - 1), ys], axis=1)
    return np.concatenate([top, bottom, left, right], axis=0)


def retarget_diffuse_to_concept(
    diffuse_image_path: str,
    diffuse_landmarks_json: str,
    concept_landmarks_json: str,
    output_size: Optional[Tuple[int, int]] = None,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: Optional[Tuple[float, ...]] = None,
    pin_boundary: bool = True,
) -> np.ndarray:
    """Warps ``diffuse_image_path`` so its facial landmarks land on the concept's positions.

    Two-stage warp: first a global similarity transform (scale + translation, see
    _fit_similarity) absorbs the coarse camera-framing difference between the mvConcept/mvDiff
    rigs, then a thin-plate-spline over the residual (via genies' rbf_deform -- scipy
    RBFInterpolator, dimension-agnostic, unlike genies.meshutils.geometry.rbf.retarget/RBFKernel
    which hard-code 3D mesh vertices) captures the finer per-feature proportion differences.
    Applied as a backward map: for every pixel in the output canvas, the field gives the
    corresponding location to sample in the diffuse image.

    Args:
        diffuse_image_path: the mvDiff render to warp.
        diffuse_landmarks_json / concept_landmarks_json: matching *_landmark.json files.
        output_size: (width, height) of the output canvas; defaults to the diffuse image's own
            size. Concept landmarks are re-expressed in that canvas via their normalized (0-1)
            coordinates, since both views share the same face framing/camera setup.
        kernel: passed through to scipy's RBFInterpolator via rbf_deform; "thin_plate_spline" is
            the smooth global warp suited to subtle proportion correction.
        smoothing: 0.0 = exact interpolation at landmarks; raise slightly (e.g. 1e-3) if the
            landmarks are noisy and an exact fit produces visible ripples.
        border_mode: cv2.remap border handling for sampling outside the diffuse image's edges.
            Matching concept's framing legitimately needs pixels beyond the diffuse crop
            whenever diffuse is the more tightly-zoomed render (the common case), so this
            defaults to BORDER_CONSTANT (flat fill) rather than a reflect/replicate mode --
            reflecting would mirror real head geometry into that region, which reads as a
            folded/duplicated fragment rather than plain background.
        border_value: fill color for border_mode=BORDER_CONSTANT. None (default) auto-detects it
            from the diffuse image's own corner pixel, so it matches that image's actual
            background (these renders are transparent white -- (255, 255, 255, 0) -- but this
            isn't hard-coded in case a given asset differs).
        pin_boundary: add a ring of anchor points along the canvas edge holding the *residual*
            spline at zero there (see _boundary_anchor_points) so only the coarse global scale
            applies near the edges, not extra local wiggle from the fine TPS extrapolating
            unconstrained. This does NOT cancel the coarse scale itself at the boundary -- that
            scale is the real, needed correction (confirmed against a manual landmark-overlay
            comparison) and holds all the way to the edge; only the small residual is pinned.

    Returns:
        The warped image, same dtype/channel count as the loaded diffuse image (BGR or BGRA).
    """
    diffuse_img = cv2.imread(diffuse_image_path, cv2.IMREAD_UNCHANGED)
    if diffuse_img is None:
        raise FileNotFoundError(diffuse_image_path)
    src_h, src_w = diffuse_img.shape[:2]
    out_w, out_h = output_size if output_size is not None else (src_w, src_h)

    diffuse_landmarks = load_openpose_landmarks(diffuse_landmarks_json)
    concept_landmarks = load_openpose_landmarks(concept_landmarks_json)
    if list(diffuse_landmarks) != list(concept_landmarks):
        raise ValueError(
            f"{diffuse_landmarks_json} and {concept_landmarks_json} don't share the same "
            "landmark names/order -- can't retarget without point correspondence"
        )

    diffuse_px = _landmarks_to_pixels(diffuse_landmarks, src_w, src_h)
    concept_px = _landmarks_to_pixels(concept_landmarks, out_w, out_h)

    # Stage 1: coarse camera-framing correction (plain scale + translation, concept -> diffuse).
    coarse_scale, coarse_translation = _fit_similarity(concept_px, diffuse_px)
    coarse_landmarks = concept_px * coarse_scale + coarse_translation

    # Stage 2: TPS over the residual only (coarse-mapped landmark estimate -> actual diffuse
    # position). Boundary anchors pin the *residual* to zero at the canvas edge (coarse_anchor ->
    # coarse_anchor, i.e. "no extra correction beyond the coarse scale here") so the fine TPS
    # doesn't extrapolate unconstrained far from any landmark -- the coarse scale itself still
    # applies at full strength all the way to the edge, which is what actually needs to happen:
    # the head silhouette fills nearly the whole canvas, so pinning the *total* map to identity
    # at the boundary (an earlier attempt) would cancel the real correction almost everywhere it
    # matters, not just in an unused background margin.
    residual_src, residual_tgt = coarse_landmarks, diffuse_px
    if pin_boundary:
        anchors = _boundary_anchor_points(out_w, out_h)
        coarse_anchors = anchors * coarse_scale + coarse_translation
        residual_src = np.concatenate([residual_src, coarse_anchors], axis=0)
        residual_tgt = np.concatenate([residual_tgt, coarse_anchors], axis=0)

    grid_x, grid_y = np.meshgrid(
        np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64)
    )
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    coarse_grid = grid_points * coarse_scale + coarse_translation

    # Residual displacement field maps coarse-mapped-concept-space -> actual diffuse-space;
    # applying it to the coarse-mapped output grid gives, per output pixel, where to sample.
    sample_points = rbf_deform(
        coarse_grid,
        src_landmarks_pos=residual_src,
        tgt_landmarks_pos=residual_tgt,
        degree=1,
        kernel=kernel,
        smoothing=smoothing,
    )

    map_x = sample_points[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = sample_points[:, 1].reshape(out_h, out_w).astype(np.float32)

    remap_kwargs = {}
    if border_mode == cv2.BORDER_CONSTANT:
        if border_value is None:
            border_value = tuple(float(c) for c in diffuse_img[0, 0])
        remap_kwargs["borderValue"] = border_value

    return cv2.remap(
        diffuse_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border_mode, **remap_kwargs
    )


def retarget_and_save(
    diffuse_image_path: str,
    diffuse_landmarks_json: str,
    concept_landmarks_json: str,
    output_path: str,
    **kwargs,
) -> str:
    """retarget_diffuse_to_concept + cv2.imwrite convenience wrapper. Returns output_path."""
    warped = retarget_diffuse_to_concept(
        diffuse_image_path, diffuse_landmarks_json, concept_landmarks_json, **kwargs
    )
    cv2.imwrite(output_path, warped)
    return output_path


def discover_concept_landmarks(folder: str, view_label: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(folder, f"*_mvConcept_{view_label}_landmark.json")))
    return matches[0] if matches else None


def retarget_character_views(
    folder: str,
    output_dir: str,
    view_labels: Tuple[str, ...] = VIEW_LABELS,
    **kwargs,
) -> Dict[str, str]:
    """Retargets every discovered mvDiff view in ``folder`` onto its matching mvConcept landmarks.

    Reuses multiview_feature_bake.discover_character_assets for the diffuse side; concept
    landmarks are located by the same *_mvConcept_<view>_landmark.json naming convention. Views
    missing any of {diffuse image, diffuse landmarks, concept landmarks} are skipped.

    Returns: {view_label: output_image_path} for the views actually retargeted.
    """
    assets = discover_character_assets(folder)
    os.makedirs(output_dir, exist_ok=True)
    outputs: Dict[str, str] = {}
    for view in view_labels:
        diffuse_img = assets.get(f"{view}_image")
        diffuse_lm = assets.get(f"{view}_landmarks")
        concept_lm = discover_concept_landmarks(folder, view)
        if not (diffuse_img and diffuse_lm and concept_lm):
            continue
        out_path = os.path.join(output_dir, f"retargeted_{view}.png")
        outputs[view] = retarget_and_save(diffuse_img, diffuse_lm, concept_lm, out_path, **kwargs)
    return outputs


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Warp a character's mvDiff renders onto their mvConcept landmark layout."
    )
    parser.add_argument("character_folder", help="e.g. test_textures/african_female_0003")
    parser.add_argument("output_dir", help="where retargeted_<view>.png files are written")
    parser.add_argument("--views", nargs="+", default=list(VIEW_LABELS), choices=VIEW_LABELS)
    parser.add_argument("--no-pin-boundary", action="store_true", help="disable edge anchoring")
    parser.add_argument("--smoothing", type=float, default=0.0)
    args = parser.parse_args()

    outputs = retarget_character_views(
        args.character_folder,
        args.output_dir,
        view_labels=tuple(args.views),
        pin_boundary=not args.no_pin_boundary,
        smoothing=args.smoothing,
    )
    if not outputs:
        raise SystemExit(
            f"No views retargeted -- check that {args.character_folder} has matching "
            "*_mvDiff_<view>.png / *_mvDiff_<view>_landmark.json / *_mvConcept_<view>_landmark.json"
        )
    for view, path in outputs.items():
        print(f"{view}: {path}")


if __name__ == "__main__":
    _main()
