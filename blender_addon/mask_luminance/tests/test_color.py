"""core.color.apply_exposure_gamma: sanity checks plus a parity check.

``core/color.py`` reimplements the standalone repo's
``texture_edit.apply_exposure_gamma`` in pure numpy (the addon can't import
``genies`` — see that module's docstring). The parity test below confirms the
two stay in lockstep; the rest are plain behavioral checks that don't need
the standalone tool or genies at all.
"""

from __future__ import annotations

import numpy as np

from mask_luminance.core import color


def _random_rgb(rng: np.random.Generator, h: int = 32, w: int = 48) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_neutral_settings_are_a_noop():
    rng = np.random.default_rng(0)
    image = _random_rgb(rng)
    result = color.apply_exposure_gamma(image, exposure=0.0, gamma=1.0, shadow_bias=0.0)
    np.testing.assert_array_equal(result, image)


def test_positive_exposure_brightens():
    rng = np.random.default_rng(1)
    # Keep away from 0/255 so a real brightening always has room to show.
    image = np.clip(rng.integers(40, 200, size=(16, 16, 3)), 0, 255).astype(np.uint8)
    result = color.apply_exposure_gamma(image, exposure=1.0)
    assert result.astype(np.int16).mean() > image.astype(np.int16).mean()


def test_negative_exposure_darkens():
    rng = np.random.default_rng(2)
    image = np.clip(rng.integers(40, 200, size=(16, 16, 3)), 0, 255).astype(np.uint8)
    result = color.apply_exposure_gamma(image, exposure=-1.0)
    assert result.astype(np.int16).mean() < image.astype(np.int16).mean()


def test_gamma_above_one_lifts_midtones():
    midtone = np.full((4, 4, 3), 128, dtype=np.uint8)
    result = color.apply_exposure_gamma(midtone, gamma=2.0)
    assert result.mean() > 128


def test_gamma_below_one_crushes_midtones():
    midtone = np.full((4, 4, 3), 128, dtype=np.uint8)
    result = color.apply_exposure_gamma(midtone, gamma=0.5)
    assert result.mean() < 128


def test_shadow_bias_tapers_effect_toward_white():
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    white = np.full((4, 4, 3), 255, dtype=np.uint8)

    black_full_bias = color.apply_exposure_gamma(black, gamma=2.0, shadow_bias=1.0)
    black_no_bias = color.apply_exposure_gamma(black, gamma=2.0, shadow_bias=0.0)
    # Full shadow_bias applies the full effect in black, same as no bias at all.
    np.testing.assert_array_equal(black_full_bias, black_no_bias)

    white_full_bias = color.apply_exposure_gamma(white, gamma=2.0, shadow_bias=1.0)
    # Full shadow_bias tapers to no effect in white -- gamma=2.0 would
    # otherwise leave 255 unchanged anyway, so use exposure instead, which
    # would otherwise clip a still-brighter white.
    white_full_bias = color.apply_exposure_gamma(white, exposure=-2.0, shadow_bias=1.0)
    np.testing.assert_array_equal(white_full_bias, white)


def test_matches_standalone_texture_edit(standalone_texture_edit):
    rng = np.random.default_rng(3)
    image = _random_rgb(rng)

    for exposure, gamma, shadow_bias in ((0.7, 1.0, 0.0), (-1.2, 1.6, 0.0), (0.5, 0.7, 0.8)):
        ours = color.apply_exposure_gamma(image, exposure=exposure, gamma=gamma, shadow_bias=shadow_bias)
        theirs = standalone_texture_edit.apply_exposure_gamma(
            image, exposure=exposure, gamma=gamma, shadow_bias=shadow_bias
        )
        np.testing.assert_array_equal(ours, theirs)
