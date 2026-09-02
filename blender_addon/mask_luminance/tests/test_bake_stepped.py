"""prepare_bake/bake_step/bake_generator/finalize_bake — the API MASKLUM_OT_bake's

modal() actually drives, one step per timer tick. ``run_bake`` already proves
these produce the same pixels as ``core.blend.process()`` (it's implemented
in terms of them); these tests instead pin down the *stepping* behavior
itself: how many steps there are, that progress advances one at a time, and
that driving it manually one step per call still lands on the exact same
result as running it straight through.
"""

from __future__ import annotations

import bpy
import numpy as np
import pytest

from mask_luminance.core import blend as core_blend
from mask_luminance.scene import bake as scene_bake

CHANNEL_SPECS = [
    dict(name="shadow_1", mask_path="x", enabled=True, gate_mode="weight", threshold=10.0, radius=10.0, strength=0.8),
    dict(name="highlight", mask_path="x", enabled=True, gate_mode="weight", threshold=14.0, radius=6.0, strength=0.6),
    dict(name="chin_shadow", mask_path="x", enabled=True, gate_mode="weight", threshold=8.0, radius=12.0, strength=0.7),
]

MASK_FILES = {
    "shadow_1": "masks/shadow_mask_1.png",
    "highlight": "masks/highlight_mask.png",
    "chin_shadow": "masks/chin_shadow_mask_1.png",
}


@pytest.fixture
def loaded_images(repo_root):
    paths = {
        "source": repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png",
        **{name: repo_root / rel for name, rel in MASK_FILES.items()},
    }
    for path in paths.values():
        if not path.exists():
            pytest.skip(f"Reference asset missing: {path}")
    loaded = {key: bpy.data.images.load(str(path), check_existing=True) for key, path in paths.items()}
    try:
        yield loaded
    finally:
        for image in loaded.values():
            bpy.data.images.remove(image)


def _channels():
    return [core_blend.MaskChannel(**spec) for spec in CHANNEL_SPECS]


def test_prepare_bake_has_one_work_item_per_ungrouped_channel(loaded_images):
    state = scene_bake.prepare_bake(
        source=loaded_images["source"],
        channels=_channels(),
        mask_images={name: loaded_images[name] for name in MASK_FILES},
        diffuse_mode="self",
    )
    assert state.total_steps == 3
    assert not state.done
    assert state.step_index == 0


def test_bake_step_advances_one_at_a_time_then_reports_done(loaded_images):
    state = scene_bake.prepare_bake(
        source=loaded_images["source"],
        channels=_channels(),
        mask_images={name: loaded_images[name] for name in MASK_FILES},
        diffuse_mode="self",
    )

    for expected_step in (1, 2, 3):
        assert scene_bake.bake_step(state) is True
        assert state.step_index == expected_step
        assert state.done == (expected_step == 3)

    assert scene_bake.bake_step(state) is False
    assert state.step_index == 3


def test_bake_generator_yields_progress_tuples(loaded_images):
    state = scene_bake.prepare_bake(
        source=loaded_images["source"],
        channels=_channels(),
        mask_images={name: loaded_images[name] for name in MASK_FILES},
        diffuse_mode="self",
    )
    progress = list(scene_bake.bake_generator(state))
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_stepped_result_matches_run_bake(loaded_images):
    channels_a = _channels()
    channels_b = _channels()
    masks = {name: loaded_images[name] for name in MASK_FILES}

    state = scene_bake.prepare_bake(
        source=loaded_images["source"], channels=channels_a, mask_images=masks,
        diffuse_mode="self", result_name="stepped_result",
    )
    for _ in scene_bake.bake_generator(state):
        pass
    stepped_image = scene_bake.finalize_bake(state)

    try:
        run_bake_image, _ = scene_bake.run_bake(
            source=loaded_images["source"], channels=channels_b, mask_images=masks,
            diffuse_mode="self", result_name="run_bake_result",
        )
        try:
            from mask_luminance.scene.images import image_to_rgb

            np.testing.assert_array_equal(image_to_rgb(stepped_image), image_to_rgb(run_bake_image))
        finally:
            bpy.data.images.remove(run_bake_image)
    finally:
        bpy.data.images.remove(stepped_image)


def test_prepare_bake_missing_mask_for_enabled_channel_raises(loaded_images):
    with pytest.raises(KeyError):
        scene_bake.prepare_bake(
            source=loaded_images["source"],
            channels=_channels(),
            mask_images={},
            diffuse_mode="self",
        )


def test_prepare_bake_no_active_channels_gives_zero_steps(loaded_images):
    channels = [core_blend.MaskChannel(name="off", mask_path="x", enabled=False)]
    state = scene_bake.prepare_bake(
        source=loaded_images["source"], channels=channels, mask_images={}, diffuse_mode="self",
    )
    assert state.total_steps == 0
    assert state.done
    assert list(scene_bake.bake_generator(state)) == []
