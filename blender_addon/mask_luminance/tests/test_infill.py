"""Vendored infill functions must match the real genies implementation exactly.

``core/infill.py`` is a copy of two functions from
``genies.meshutils.shading.texture_utils`` (see that module's docstring for
why). These tests diff our copy against the real thing so a future edit to
either side doesn't silently drift without anyone noticing.
"""

from __future__ import annotations

import numpy as np

from mask_luminance.core import infill


def _random_rgb(rng: np.random.Generator, h: int = 64, w: int = 96) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _random_hole_mask(rng: np.random.Generator, h: int = 64, w: int = 96) -> np.ndarray:
    # A handful of blobs rather than fully random noise, closer to a real
    # feathered mask footprint than uniform per-pixel noise would be.
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(5):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(3, 15), rng.integers(3, 15)
        yy, xx = np.ogrid[:h, :w]
        mask |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    return mask


def test_extend_texture_boundaries_matches_genies_unbounded(genies_texture_utils):
    rng = np.random.default_rng(0)
    image = _random_rgb(rng)
    holes = _random_hole_mask(rng)

    ours = infill.extend_texture_boundaries(image, holes, max_distance=None)
    theirs = genies_texture_utils.extend_texture_boundaries(image, holes, max_distance=None)

    np.testing.assert_array_equal(ours, theirs)


def test_extend_texture_boundaries_matches_genies_bounded(genies_texture_utils):
    rng = np.random.default_rng(1)
    image = _random_rgb(rng)
    holes = _random_hole_mask(rng)

    ours = infill.extend_texture_boundaries(image, holes, max_distance=12.0)
    theirs = genies_texture_utils.extend_texture_boundaries(image, holes, max_distance=12.0)

    np.testing.assert_array_equal(ours, theirs)


def test_extend_texture_boundaries_no_holes_is_a_noop(genies_texture_utils):
    rng = np.random.default_rng(2)
    image = _random_rgb(rng)
    holes = np.zeros(image.shape[:2], dtype=bool)

    ours = infill.extend_texture_boundaries(image, holes, max_distance=None)
    np.testing.assert_array_equal(ours, image)


def test_apply_extrapolation_blur_matches_genies(genies_texture_utils):
    rng = np.random.default_rng(3)
    image = _random_rgb(rng).astype(np.float32)
    holes = _random_hole_mask(rng)
    # apply_extrapolation_blur mutates its input in place, so give each call
    # its own copy.
    ours = infill.apply_extrapolation_blur(image.copy(), holes, global_radius=6.0)
    theirs = genies_texture_utils.apply_extrapolation_blur(image.copy(), holes, global_radius=6.0)

    np.testing.assert_array_equal(ours, theirs)


def test_apply_extrapolation_blur_no_mask_is_a_noop(genies_texture_utils):
    rng = np.random.default_rng(4)
    image = _random_rgb(rng).astype(np.float32)
    empty = np.zeros(image.shape[:2], dtype=bool)

    result = infill.apply_extrapolation_blur(image.copy(), empty, global_radius=6.0)
    np.testing.assert_array_equal(result, image)
