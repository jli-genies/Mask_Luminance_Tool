"""Live-preview wiring: which property changes trigger a recompute, and how it's throttled.

``_run_preview_now``/``_channels_and_masks_ready_for_preview`` are plain
functions (fully callable here, unlike bpy.types.Operator subclasses — see
test_operators.py's module docstring for why those can't be instantiated
directly). What genuinely can't be proven headlessly is whether
``bpy.app.timers`` actually *fires* the debounced callback — that needs
Blender's real event loop, same limitation as the modal bake operator — so
these tests instead check the two things that don't need the timer to fire:
that scheduling it is idempotent (``is_registered`` stays True rather than
stacking up duplicate registrations), and that property `update=` callbacks
actually call the scheduler when a value changes.
"""

from __future__ import annotations

import bpy
import numpy as np
import pytest

from mask_luminance import operators


@pytest.fixture
def addon(request):
    import mask_luminance

    mask_luminance.register()
    request.addfinalizer(mask_luminance.unregister)
    return mask_luminance


@pytest.fixture
def scene_settings(addon, request):
    settings = bpy.context.scene.mask_luminance
    settings.channels.clear()
    settings.active_channel_index = 0
    settings.source = None
    settings.diffuse_image = None
    settings.feature_preserve_image = None
    settings.diffuse_mode = "self"
    settings.live_preview = True

    def _cleanup_timer():
        if bpy.app.timers.is_registered(operators._run_preview_timer):
            bpy.app.timers.unregister(operators._run_preview_timer)

    _cleanup_timer()
    request.addfinalizer(_cleanup_timer)
    return settings


@pytest.fixture
def loaded_images(repo_root):
    paths = {
        "source": repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png",
        "mask": repo_root / "masks" / "shadow_mask_1.png",
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


def test_channels_ready_for_preview_skips_disabled_and_unassigned(scene_settings, loaded_images):
    bpy.ops.mask_luminance.channel_add()  # enabled, no mask -> should be skipped
    scene_settings.channels[0].channel_name = "no_mask"

    bpy.ops.mask_luminance.channel_add()  # enabled, has mask -> should be included
    scene_settings.channels[1].channel_name = "ready"
    scene_settings.channels[1].mask_image = loaded_images["mask"]

    bpy.ops.mask_luminance.channel_add()  # disabled, has mask -> should be skipped
    scene_settings.channels[2].channel_name = "disabled"
    scene_settings.channels[2].mask_image = loaded_images["mask"]
    scene_settings.channels[2].enabled = False

    channels, mask_images = operators._channels_and_masks_ready_for_preview(scene_settings)

    assert [c.name for c in channels] == ["ready"]
    assert set(mask_images) == {"ready"}


def test_run_preview_now_writes_a_preview_image(scene_settings, loaded_images):
    scene_settings.source = loaded_images["source"]
    bpy.ops.mask_luminance.channel_add()
    scene_settings.channels[0].channel_name = "shadow_1"
    scene_settings.channels[0].mask_image = loaded_images["mask"]
    scene_settings.preview_max_dimension = 128

    operators._run_preview_now(scene_settings)

    preview = bpy.data.images.get(f"{loaded_images['source'].name}_preview")
    try:
        assert preview is not None
        assert max(preview.size) <= 128
    finally:
        if preview is not None:
            bpy.data.images.remove(preview)


def test_preview_now_operator_matches_run_preview_now(scene_settings, loaded_images):
    scene_settings.source = loaded_images["source"]
    bpy.ops.mask_luminance.channel_add()
    scene_settings.channels[0].channel_name = "shadow_1"
    scene_settings.channels[0].mask_image = loaded_images["mask"]
    scene_settings.preview_max_dimension = 128

    result = bpy.ops.mask_luminance.preview_now()
    assert result == {"FINISHED"}

    preview = bpy.data.images.get(f"{loaded_images['source'].name}_preview")
    try:
        assert preview is not None
    finally:
        if preview is not None:
            bpy.data.images.remove(preview)


def test_preview_now_operator_requires_source(scene_settings):
    assert bpy.ops.mask_luminance.preview_now.poll() is False


def test_schedule_preview_update_is_idempotent_while_pending(scene_settings):
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is False

    operators._schedule_preview_update()
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is True

    # Calling it again before the first has fired must not raise or stack up
    # a second registration — this is the whole throttling mechanism.
    operators._schedule_preview_update()
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is True


def test_changing_a_threshold_schedules_the_preview_timer(scene_settings, loaded_images):
    bpy.ops.mask_luminance.channel_add()
    channel = scene_settings.channels[0]
    channel.mask_image = loaded_images["mask"]  # this itself schedules the timer too

    bpy.app.timers.unregister(operators._run_preview_timer)
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is False

    channel.threshold = 42.0
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is True


def test_renaming_a_channel_does_not_schedule_the_preview_timer(scene_settings):
    """channel_name has no update= callback — see MASKLUM_PG_channel's comment on why."""
    bpy.ops.mask_luminance.channel_add()  # its own execute() schedules the timer already
    bpy.app.timers.unregister(operators._run_preview_timer)
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is False

    scene_settings.channels[0].channel_name = "renamed"
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is False


def test_channel_move_schedules_the_preview_timer(scene_settings):
    bpy.ops.mask_luminance.channel_add()
    bpy.ops.mask_luminance.channel_add()
    if bpy.app.timers.is_registered(operators._run_preview_timer):
        bpy.app.timers.unregister(operators._run_preview_timer)

    scene_settings.active_channel_index = 0
    bpy.ops.mask_luminance.channel_move(direction="DOWN")
    assert bpy.app.timers.is_registered(operators._run_preview_timer) is True


def test_run_preview_timer_is_a_noop_without_a_source(scene_settings):
    """Must not raise even with nothing configured — it runs on every property change."""
    assert operators._run_preview_timer() is None


def test_run_preview_timer_respects_live_preview_flag(scene_settings, loaded_images):
    scene_settings.source = loaded_images["source"]
    bpy.ops.mask_luminance.channel_add()
    scene_settings.channels[0].mask_image = loaded_images["mask"]
    scene_settings.live_preview = False

    preview_name = f"{loaded_images['source'].name}_preview"
    assert bpy.data.images.get(preview_name) is None

    operators._run_preview_timer()

    assert bpy.data.images.get(preview_name) is None
