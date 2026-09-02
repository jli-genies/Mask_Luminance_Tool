"""Nearest-color infill and extrapolation-boundary blur.

Vendored from ``genies.meshutils.shading.texture_utils`` (the internal Genies
package ``matte_luminance_blend.py`` originally depended on) so this addon has
no dependency outside numpy/scipy/opencv. Algorithms are unchanged from the
source; only the module-level docs were trimmed to the two functions actually
used by ``core.blend``.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import scipy.ndimage


def extend_texture_boundaries(
    image: np.ndarray,
    empty_mask: np.ndarray,
    max_distance: Optional[float] = 2,
) -> np.ndarray:
    """Extrapolates valid edge pixels outward into uninhabited grid regions.

    Runs a Euclidean distance transform to find the nearest valid coordinate
    for every empty pixel, then floods that color/weight data outward. An
    optional maximum distance restricts the padding to a margin around the
    valid region instead of filling every empty pixel in the image.

    Args:
        image: The source 2D or 3D array requiring padding.
        empty_mask: Boolean mask designating uninhabited pixels.
        max_distance: Pixel limit for the extrapolation; ``None`` fills every
            empty pixel regardless of distance from the nearest valid one.

    Returns:
        The padded array with valid data pushed into empty regions.
    """
    padded_image = np.copy(image)
    if not np.any(empty_mask):
        return padded_image

    distances, indices = scipy.ndimage.distance_transform_edt(empty_mask, return_indices=True)

    if max_distance is None:
        if image.ndim == 3:
            padded_image[empty_mask] = image[indices[0][empty_mask], indices[1][empty_mask], :]
        else:
            padded_image[empty_mask] = image[indices[0][empty_mask], indices[1][empty_mask]]
    else:
        margin_mask = empty_mask & (distances <= max_distance)
        if image.ndim == 3:
            padded_image[margin_mask] = image[indices[0][margin_mask], indices[1][margin_mask], :]
        else:
            padded_image[margin_mask] = image[indices[0][margin_mask], indices[1][margin_mask]]

    return padded_image


def apply_extrapolation_blur(
    img: np.ndarray,
    extra_mask: np.ndarray,
    global_radius: float,
) -> np.ndarray:
    """Applies an adaptive, island-aware blur to softened extrapolated regions.

    Args:
        img: The (H, W, C) image buffer.
        extra_mask: Mask of pixels identified as extrapolated.
        global_radius: The maximum blur radius in pixels.

    Returns:
        Image with softened extrapolation boundaries.
    """
    if not np.any(extra_mask):
        return img

    height, width = img.shape[:2]
    pad = int(np.ceil(global_radius * 2))

    rows = np.any(extra_mask, axis=1)
    cols = np.any(extra_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin, rmax = max(0, rmin - pad), min(height, rmax + pad + 1)
    cmin, cmax = max(0, cmin - pad), min(width, cmax + pad + 1)

    sub_img = img[rmin:rmax, cmin:cmax].astype(np.float32)
    sub_extra = extra_mask[rmin:rmax, cmin:cmax]

    dist = scipy.ndimage.distance_transform_edt(~sub_extra)
    labeled_islands, n_islands = scipy.ndimage.label(~sub_extra)
    max_dists = scipy.ndimage.maximum(dist, labels=labeled_islands, index=np.arange(1, n_islands + 1))

    if np.isscalar(max_dists):
        max_dists = np.array([max_dists])

    local_radius_map = np.zeros_like(dist)
    mask_loc = labeled_islands > 0
    local_radius_map[mask_loc] = max_dists[labeled_islands[mask_loc] - 1]

    adaptive_radius = np.minimum(global_radius, local_radius_map * 1.5)
    adaptive_radius[adaptive_radius < 1.0] = 1.0
    weights = np.clip(1.0 - (dist / adaptive_radius), 0.0, 1.0)

    sigma = global_radius * 0.4
    content_mask = (np.max(sub_img, axis=2) > 0).astype(np.float32)

    blurred_sub = np.zeros_like(sub_img)
    m_blur = scipy.ndimage.gaussian_filter(content_mask, sigma=sigma)
    m_blur[m_blur < 1e-4] = 1.0

    for c in range(sub_img.shape[2]):
        c_blur = scipy.ndimage.gaussian_filter(sub_img[:, :, c], sigma=sigma)
        blurred_sub[:, :, c] = c_blur / m_blur

    w_expanded = weights[..., np.newaxis]
    interpolated = (sub_img * (1.0 - w_expanded)) + (blurred_sub * w_expanded)

    img[rmin:rmax, cmin:cmax] = np.clip(interpolated, 0, 255).astype(np.uint8)
    return img
