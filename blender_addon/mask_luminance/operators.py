"""Property groups, operators and panels — the thin Blender-facing layer.

No algorithm logic lives here — see ``core/blend.py`` for the luminance-blend
math and ``scene/`` for the ``bpy.types.Image`` <-> numpy array glue.
``MASKLUM_OT_bake`` only converts property-group values into a
``core.blend.MaskChannel`` list and drives ``scene.bake``'s stepped API.

``MASKLUM_OT_bake`` is a *modal* operator, not a background ``threading``
worker, even though the actual per-channel work (numpy/opencv calls, which
release the GIL) would run fine off the main thread. The reason is
``bpy.data``: every step still has to read pixels from and, at the end,
write pixels into ``bpy.types.Image`` datablocks, and touching ``bpy.data``
from any thread other than the main one is unsafe in Blender, full stop —
there's no lock to take, it's simply not supported. A modal operator gets
UI responsiveness the supported way instead: each timer tick asks
``scene.bake`` for exactly one more step, applies it, then returns control to
Blender's own event loop before the next tick.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup, UIList

from .core import blend as core_blend
from .scene import bake as scene_bake
from .scene.progress_overlay import MASKLUM_PG_BakeProgress, overlay_begin, overlay_end, overlay_update

GATE_MODE_ITEMS = (
    ("weight", "Weight", "Grayscale luminance/255 as a soft gate"),
    ("blue_paint", "Blue Paint", "Blue-dominant paint on a UV map marks the gate"),
    ("color_id", "Color ID", "Multi-region ID color map; select regions by name"),
)

DIFFUSE_MODE_ITEMS = (
    ("self", "Self", "Reconstruct the diffuse target from the texture's own clean pixels"),
    ("uv", "UV Map", "Use a separate flat-diffuse UV texture as the target"),
    ("palette", "Palette", "Sample a flat skin color from a multiview/palette render"),
)


# =============================================================================
# PROPERTY GROUPS
# =============================================================================
class MASKLUM_PG_channel(PropertyGroup):
    """Mirrors core.blend.MaskChannel, minus mask_path (a live Image pointer instead)."""

    channel_name: StringProperty(name="Name", default="channel")
    mask_image: PointerProperty(name="Mask", type=bpy.types.Image)
    enabled: BoolProperty(name="Enabled", default=True)
    gate_mode: EnumProperty(name="Gate Mode", items=GATE_MODE_ITEMS, default="weight")
    threshold: FloatProperty(name="Threshold", default=12.0, min=0.0, max=255.0)
    radius: FloatProperty(name="Radius", default=8.0, min=0.0, max=512.0)
    strength: FloatProperty(name="Strength", default=0.85, min=0.0, max=1.0)
    diffuse_mix: FloatProperty(name="Diffuse Mix", default=0.0, min=0.0, max=1.0)
    use_infill: BoolProperty(name="Use Infill", default=True)
    spill_outside: BoolProperty(name="Spill Outside", default=False)
    fill_holes: BoolProperty(name="Fill Holes", default=False)
    region_tolerance: IntProperty(name="Region Tolerance", default=40, min=0, max=255)
    blend_group: StringProperty(name="Blend Group", default="")
    blend_weight: FloatProperty(name="Blend Weight", default=1.0, min=0.0)

    def to_mask_channel(self) -> core_blend.MaskChannel:
        """Builds the core.blend.MaskChannel this property group represents.

        ``mask_path`` is a vestigial field on the dataclass (only the
        file-path-based ``core.blend.process()`` CLI/test entry point reads
        it) — ``scene.run_bake`` resolves the mask by channel name against
        the ``mask_images`` dict it's given instead, so any placeholder value
        here is fine.
        """
        return core_blend.MaskChannel(
            name=self.channel_name,
            mask_path=self.channel_name,
            enabled=self.enabled,
            gate_mode=self.gate_mode,
            threshold=self.threshold,
            radius=self.radius,
            strength=self.strength,
            diffuse_mix=self.diffuse_mix,
            use_infill=self.use_infill,
            spill_outside=self.spill_outside,
            fill_holes=self.fill_holes,
            region_tolerance=self.region_tolerance,
            blend_group=self.blend_group or None,
            blend_weight=self.blend_weight,
        )


class MASKLUM_PG_settings(PropertyGroup):
    """Top-level addon state, stored on the Scene."""

    source: PointerProperty(name="Source Texture", type=bpy.types.Image)
    diffuse_mode: EnumProperty(name="Diffuse Mode", items=DIFFUSE_MODE_ITEMS, default="self")
    diffuse_image: PointerProperty(name="Diffuse Image", type=bpy.types.Image)
    feature_preserve_image: PointerProperty(name="Feature Preserve Mask", type=bpy.types.Image)
    self_locality_radius: FloatProperty(
        name="Self Locality Radius", default=core_blend.DEFAULT_SELF_LOCALITY_RADIUS, min=1.0, max=4096.0
    )
    luminance_only: BoolProperty(name="Luminance Only", default=True)

    channels: CollectionProperty(type=MASKLUM_PG_channel)
    active_channel_index: IntProperty(default=0)


# =============================================================================
# OPERATORS
# =============================================================================
class MASKLUM_OT_channel_add(Operator):
    bl_idname = "mask_luminance.channel_add"
    bl_label = "Add Mask Channel"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.mask_luminance
        channel = settings.channels.add()
        channel.channel_name = f"channel_{len(settings.channels)}"
        settings.active_channel_index = len(settings.channels) - 1
        return {"FINISHED"}


class MASKLUM_OT_channel_remove(Operator):
    bl_idname = "mask_luminance.channel_remove"
    bl_label = "Remove Mask Channel"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.mask_luminance.channels)

    def execute(self, context):
        settings = context.scene.mask_luminance
        settings.channels.remove(settings.active_channel_index)
        settings.active_channel_index = max(0, settings.active_channel_index - 1)
        return {"FINISHED"}


class MASKLUM_OT_channel_move(Operator):
    bl_idname = "mask_luminance.channel_move"
    bl_label = "Move Mask Channel"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=(("UP", "Up", ""), ("DOWN", "Down", "")))

    @classmethod
    def poll(cls, context):
        return len(context.scene.mask_luminance.channels) > 1

    def execute(self, context):
        settings = context.scene.mask_luminance
        index = settings.active_channel_index
        target = index - 1 if self.direction == "UP" else index + 1
        if 0 <= target < len(settings.channels):
            settings.channels.move(index, target)
            settings.active_channel_index = target
        return {"FINISHED"}


class MASKLUM_OT_bake(Operator):
    """Modal: one channel/blend-group per timer tick, with a progress overlay.

    See the module docstring for why this is modal rather than a background
    thread. ``invoke()`` does the non-steppable setup (pixel extraction,
    diffuse-target construction — see ``scene.bake.prepare_bake``) up front;
    ``modal()`` then advances one work item per ``TIMER`` event via
    ``scene.bake.bake_generator`` until it's exhausted, and Esc sets
    ``cancel_requested`` on the progress property group, checked at the next
    tick.
    """

    bl_idname = "mask_luminance.bake"
    bl_label = "Bake Mask Luminance"
    bl_options = {"REGISTER"}

    _timer = None
    _gen = None
    _state = None

    @classmethod
    def poll(cls, context):
        return context.scene.mask_luminance.source is not None

    def invoke(self, context, event):
        settings = context.scene.mask_luminance

        channels = [ch.to_mask_channel() for ch in settings.channels]
        mask_images = {}
        for ch in settings.channels:
            if not ch.enabled:
                continue
            if ch.mask_image is None:
                self.report({"ERROR"}, f"Channel '{ch.channel_name}' is enabled but has no mask image assigned.")
                return {"CANCELLED"}
            mask_images[ch.channel_name] = ch.mask_image

        try:
            self._state = scene_bake.prepare_bake(
                source=settings.source,
                channels=channels,
                mask_images=mask_images,
                diffuse_mode=settings.diffuse_mode,
                diffuse_image=settings.diffuse_image if settings.diffuse_mode != "self" else None,
                feature_preserve_image=settings.feature_preserve_image or None,
                luminance_only=settings.luminance_only,
                self_locality_radius=settings.self_locality_radius,
            )
        except (ValueError, KeyError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if self._state.total_steps == 0:
            return self._finish(context)

        self._gen = scene_bake.bake_generator(self._state)
        overlay_begin(context, title="Mask Luminance", total_steps=self._state.total_steps)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            overlay_update(context, phase="Cancelling…", step=self._state.step_index, total=self._state.total_steps)
            self._stop_timer(context)
            overlay_end(context)
            self.report({"WARNING"}, "Bake cancelled")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            step, total = next(self._gen)
        except StopIteration:
            return self._finish(context)

        item = self._state.work_items[step - 1]
        names = item.name if not isinstance(item, list) else "+".join(c.name for c in item)
        if not overlay_update(context, phase=f"Channel {step}/{total}: {names}", step=step, total=total):
            self._stop_timer(context)
            overlay_end(context)
            self.report({"WARNING"}, "Bake cancelled")
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def _finish(self, context):
        self._stop_timer(context)
        overlay_end(context)
        result_image = scene_bake.finalize_bake(self._state)
        self.report({"INFO"}, f"Baked '{result_image.name}' ({result_image.size[0]}x{result_image.size[1]})")
        return {"FINISHED"}

    def _stop_timer(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def execute(self, context):
        return self.invoke(context, None)


# =============================================================================
# UI
# =============================================================================
class MASKLUM_UL_channels(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "channel_name", text="", emboss=False)
        row.label(text=item.gate_mode)


class MASKLUM_PT_main(Panel):
    bl_idname = "MASKLUM_PT_main"
    bl_label = "Mask Luminance"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Mask Luminance"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mask_luminance

        layout.prop(settings, "source")
        layout.prop(settings, "diffuse_mode")
        if settings.diffuse_mode != "self":
            layout.prop(settings, "diffuse_image")
        else:
            layout.prop(settings, "self_locality_radius")
        layout.prop(settings, "feature_preserve_image")
        layout.prop(settings, "luminance_only")

        layout.separator()
        layout.label(text="Mask Channels")

        row = layout.row()
        row.template_list(
            "MASKLUM_UL_channels", "", settings, "channels", settings, "active_channel_index", rows=4
        )
        col = row.column(align=True)
        col.operator("mask_luminance.channel_add", icon="ADD", text="")
        col.operator("mask_luminance.channel_remove", icon="REMOVE", text="")
        col.separator()
        col.operator("mask_luminance.channel_move", icon="TRIA_UP", text="").direction = "UP"
        col.operator("mask_luminance.channel_move", icon="TRIA_DOWN", text="").direction = "DOWN"

        if 0 <= settings.active_channel_index < len(settings.channels):
            channel = settings.channels[settings.active_channel_index]
            box = layout.box()
            box.prop(channel, "mask_image")
            box.prop(channel, "gate_mode")
            box.prop(channel, "threshold")
            box.prop(channel, "radius")
            box.prop(channel, "strength")
            box.prop(channel, "diffuse_mix")
            box.prop(channel, "use_infill")
            box.prop(channel, "spill_outside")
            if channel.gate_mode == "weight":
                box.prop(channel, "fill_holes")
            if channel.gate_mode == "color_id":
                box.prop(channel, "region_tolerance")
            box.prop(channel, "blend_group")
            if channel.blend_group:
                box.prop(channel, "blend_weight")

        layout.separator()
        layout.operator("mask_luminance.bake", icon="RENDER_STILL")


CLASSES = [
    MASKLUM_PG_BakeProgress,
    MASKLUM_PG_channel,
    MASKLUM_PG_settings,
    MASKLUM_OT_channel_add,
    MASKLUM_OT_channel_remove,
    MASKLUM_OT_channel_move,
    MASKLUM_OT_bake,
    MASKLUM_UL_channels,
    MASKLUM_PT_main,
]
