"""scene.images <-> a real bpy.types.Image must round-trip exactly.

Verifies the two things that would silently corrupt every downstream result
if wrong: Blender's bottom-up pixel row order, and that ``Image.pixels``
carries raw stored byte values (no implicit gamma decode to undo) — both
confirmed empirically against a real Image datablock, not assumed from docs.
"""

from __future__ import annotations

import bpy
import numpy as np
import pytest

from mask_luminance.core import blend as core_blend
from mask_luminance.scene import images as scene_images


@pytest.fixture
def texture_path(repo_root):
    path = repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png"
    if not path.exists():
        pytest.skip(f"Reference texture missing: {path}")
    return path


def test_image_to_rgb_matches_cv2_raw_bytes(texture_path):
    reference = core_blend.load_rgb(str(texture_path))

    image = bpy.data.images.load(str(texture_path), check_existing=True)
    try:
        arr = scene_images.image_to_rgb(image)
        np.testing.assert_array_equal(arr, reference)
    finally:
        bpy.data.images.remove(image)


def test_rgb_to_image_round_trips_exactly(texture_path):
    reference = core_blend.load_rgb(str(texture_path))

    image = scene_images.rgb_to_image(reference, name="round_trip_test")
    try:
        back = scene_images.image_to_rgb(image)
        np.testing.assert_array_equal(back, reference)
    finally:
        bpy.data.images.remove(image)


def test_rgb_to_image_reuses_existing_datablock_of_same_size():
    rng = np.random.default_rng(0)
    first = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    second = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)

    image = scene_images.rgb_to_image(first, name="reuse_test")
    try:
        image_id = image.as_pointer()
        updated = scene_images.rgb_to_image(second, name="reuse_test", existing=image)
        assert updated.as_pointer() == image_id
        np.testing.assert_array_equal(scene_images.image_to_rgb(updated), second)
    finally:
        bpy.data.images.remove(image)


def test_rgb_to_image_replaces_existing_datablock_on_size_change():
    small = np.zeros((4, 4, 3), dtype=np.uint8)
    big = np.zeros((8, 8, 3), dtype=np.uint8)

    image = scene_images.rgb_to_image(small, name="resize_test")
    try:
        replaced = scene_images.rgb_to_image(big, name="resize_test", existing=image)
        assert tuple(replaced.size) == (8, 8)
    finally:
        bpy.data.images.remove(image)
        if replaced is not image:
            bpy.data.images.remove(replaced)
