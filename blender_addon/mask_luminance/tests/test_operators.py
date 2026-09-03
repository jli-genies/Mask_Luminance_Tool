"""End-to-end through the actual bpy.ops entry points, not just core/scene directly.

This is the layer a user actually clicks: the panel calls
``bpy.ops.mask_luminance.*``, which reads ``context.scene.mask_luminance``
(the property groups in ``operators.py``) and drives ``scene.bake``'s stepped
API.

``MASKLUM_OT_bake`` is a *modal* operator (see its docstring for why), which
puts a real hard limit on what these tests can prove: advancing past the
first tick requires Blender's actual window/event loop to deliver ``TIMER``
events to ``modal()``, and that loop doesn't exist in ``--background`` mode
or in the ``bpy`` pip package pytest runs against — calling the operator here
can only be checked up to the point where it returns ``{'RUNNING_MODAL'}``.
Full step-by-step correctness (including the exact pixels produced) is
proven instead by ``test_bake_stepped.py``, which drives ``scene.bake``'s
``prepare_bake``/``bake_step``/``bake_generator`` directly — the very
functions ``MASKLUM_OT_bake.modal()`` calls. What's left for this file is:
does the operator build the right ``MaskChannel``s from the property groups
and reach modal state without error, and do its two paths that *don't*
require the event loop (upfront validation, and the zero-active-channels
immediate-finish shortcut) behave correctly.
"""

from __future__ import annotations

import bpy
import numpy as np
import pytest

CHANNEL_SPECS = [
    dict(
        channel_name="shadow_1",
        gate_mode="weight",
        threshold=10.0,
        radius=10.0,
        strength=0.8,
        diffuse_mix=0.2,
        use_infill=True,
        mask_file="masks/shadow_mask_1.png",
    ),
    dict(
        channel_name="highlight",
        gate_mode="weight",
        threshold=14.0,
        radius=6.0,
        strength=0.6,
        diffuse_mix=0.0,
        use_infill=False,
        mask_file="masks/highlight_mask.png",
    ),
]


@pytest.fixture
def addon(request):
    import mask_luminance

    mask_luminance.register()
    request.addfinalizer(mask_luminance.unregister)
    return mask_luminance


@pytest.fixture
def scene_settings(addon):
    settings = bpy.context.scene.mask_luminance
    # register()/unregister() reconnect to the same underlying Scene
    # IDProperty storage when re-declared under the same name, so leftover
    # channels/pointers from a previous test otherwise leak into this one —
    # start every test from an explicitly empty state instead of trusting
    # re-registration to have reset it.
    settings.channels.clear()
    settings.active_channel_index = 0
    settings.source = None
    settings.diffuse_image = None
    settings.feature_preserve_image = None
    settings.diffuse_mode = "self"
    return settings


def test_channel_add_remove_move(scene_settings):
    bpy.ops.mask_luminance.channel_add()
    bpy.ops.mask_luminance.channel_add()
    assert len(scene_settings.channels) == 2

    scene_settings.channels[0].channel_name = "first"
    scene_settings.channels[1].channel_name = "second"
    scene_settings.active_channel_index = 0

    bpy.ops.mask_luminance.channel_move(direction="DOWN")
    assert [c.channel_name for c in scene_settings.channels] == ["second", "first"]

    scene_settings.active_channel_index = 0
    bpy.ops.mask_luminance.channel_remove()
    assert len(scene_settings.channels) == 1
    assert scene_settings.channels[0].channel_name == "first"


def test_flat_fill_available_regardless_of_gate_mode(scene_settings):
    """flat_fill must stay a usable property no matter which Gate Mode is selected.

    Regression guard for a report that the "Flat Fill" checkbox vanished
    specifically under gate_mode="blue_paint" in a live Blender session — the
    panel draw code has no such conditional (only Diffuse Mix/Use Infill/
    Spill Outside hide, and only based on flat_fill itself, never on gate
    mode), so on a freshly-created channel this must hold for every mode.
    """
    bpy.ops.mask_luminance.channel_add()
    channel = scene_settings.channels[0]

    for mode in ("weight", "blue_paint", "color_id"):
        channel.gate_mode = mode
        channel.flat_fill = True
        assert channel.flat_fill is True
        assert channel.to_mask_channel().flat_fill is True
        assert channel.to_mask_channel().gate_mode == mode

        channel.flat_fill = False
        assert channel.flat_fill is False
        assert channel.to_mask_channel().flat_fill is False


def test_bake_operator_requires_source(scene_settings):
    assert bpy.ops.mask_luminance.bake.poll() is False


def test_bake_operator_reports_error_when_enabled_channel_has_no_mask(scene_settings, repo_root):
    texture_path = repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png"
    if not texture_path.exists():
        pytest.skip(f"Reference texture missing: {texture_path}")

    scene_settings.source = bpy.data.images.load(str(texture_path), check_existing=True)
    bpy.ops.mask_luminance.channel_add()
    scene_settings.channels[0].channel_name = "unassigned"
    scene_settings.channels[0].enabled = True

    # bpy.ops raises RuntimeError (rather than returning {'CANCELLED'}) when
    # an operator both reports an ERROR-level message and cancels.
    with pytest.raises(RuntimeError, match="no mask image assigned"):
        bpy.ops.mask_luminance.bake()


def test_bake_operator_enters_modal_state_for_a_valid_config(scene_settings, repo_root):
    texture_path = repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png"
    for spec in CHANNEL_SPECS:
        if not (repo_root / spec["mask_file"]).exists():
            pytest.skip(f"Reference mask missing: {repo_root / spec['mask_file']}")
    if not texture_path.exists():
        pytest.skip("Reference texture missing")

    scene_settings.source = bpy.data.images.load(str(texture_path), check_existing=True)
    scene_settings.diffuse_mode = "self"

    for spec in CHANNEL_SPECS:
        bpy.ops.mask_luminance.channel_add()
        channel = scene_settings.channels[-1]
        channel.channel_name = spec["channel_name"]
        channel.mask_image = bpy.data.images.load(str(repo_root / spec["mask_file"]), check_existing=True)

    # Reaching RUNNING_MODAL means invoke()'s validation and prepare_bake()
    # both succeeded — everything short of the event-loop-driven stepping
    # itself, which test_bake_stepped.py covers directly instead (see the
    # module docstring for why that split exists).
    result = bpy.ops.mask_luminance.bake()
    assert result == {"RUNNING_MODAL"}
    assert bpy.context.window_manager.mask_luminance_bake_progress.active is True
    assert bpy.context.window_manager.mask_luminance_bake_progress.total == len(CHANNEL_SPECS)

    # Nothing will ever call modal() again in this headless test — there's
    # no window/event loop to deliver it a TIMER event — so tidy up the
    # progress state by hand instead of leaving it "active" for whichever
    # test runs next.
    bpy.context.window_manager.mask_luminance_bake_progress.active = False


def test_bake_operator_finishes_synchronously_with_zero_active_channels(scene_settings, repo_root):
    """No enabled channels needs no modal stepping at all — invoke() finishes inline."""
    texture_path = repo_root / "test_textures" / "african_female_0003_albedo_from_concept.png"
    if not texture_path.exists():
        pytest.skip("Reference texture missing")

    scene_settings.source = bpy.data.images.load(str(texture_path), check_existing=True)

    result = bpy.ops.mask_luminance.bake()
    assert result == {"FINISHED"}

    from mask_luminance.scene.images import image_to_rgb

    baked_image = bpy.data.images[f"{scene_settings.source.name}_matte"]
    np.testing.assert_array_equal(image_to_rgb(baked_image), image_to_rgb(scene_settings.source))
