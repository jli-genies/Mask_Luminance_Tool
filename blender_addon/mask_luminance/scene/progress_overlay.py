"""Fullscreen bake progress overlay — gpu draw handler + WM state.

Structurally identical to Genies' HeadGen addon's own
``scene/progress_overlay.py`` (same gpu/blf drawing approach, same WM
progress + status-text wiring), renamed for this addon. Kept here rather
than duplicated per-addon knowledge because a modal operator is the only
correct way to show live progress for a long-running bake without touching
``bpy.data`` off the main thread — see ``operators.MASKLUM_OT_bake`` for why
this isn't a background ``threading`` worker instead.
"""

from __future__ import annotations

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

_HANDLES: list = []
_SHADER = None


def _shader():
    global _SHADER
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _SHADER


class MASKLUM_PG_BakeProgress(bpy.types.PropertyGroup):
    active: bpy.props.BoolProperty(name="Active", default=False)
    title: bpy.props.StringProperty(name="Title", default="")
    phase: bpy.props.StringProperty(name="Phase", default="")
    detail: bpy.props.StringProperty(name="Detail", default="")
    step: bpy.props.IntProperty(name="Step", default=0, min=0)
    total: bpy.props.IntProperty(name="Total", default=1, min=1)
    pct: bpy.props.IntProperty(name="Percent", default=0, min=0, max=100)
    cancel_requested: bpy.props.BoolProperty(name="Cancel", default=False)


def progress_props(context: bpy.types.Context) -> MASKLUM_PG_BakeProgress:
    return context.window_manager.mask_luminance_bake_progress


def _rect_batch(x: float, y: float, w: float, h: float):
    verts = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    return batch_for_shader(_shader(), "TRI_FAN", {"pos": verts})


def _fill(x: float, y: float, w: float, h: float, color) -> None:
    shader = _shader()
    shader.bind()
    shader.uniform_float("color", color)
    _rect_batch(x, y, w, h).draw(shader)


def _text(font_id: int, x: float, y: float, text: str, size: int, color) -> None:
    if not text:
        return
    blf.size(font_id, size)
    blf.color(font_id, *color)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)


def _draw_overlay() -> None:
    ctx = bpy.context
    pg = ctx.window_manager.mask_luminance_bake_progress
    if not pg.active:
        return

    region = ctx.region
    if region is None:
        return

    w, h = region.width, region.height
    if w < 64 or h < 64:
        return

    gpu.state.blend_set("ALPHA")
    _fill(0, 0, w, h, (0.03, 0.03, 0.07, 0.75))

    card_w = min(640, w - 48)
    card_h = min(220, int(h * 0.3))
    cx = (w - card_w) * 0.5
    cy = (h - card_h) * 0.5

    _fill(cx, cy, card_w, card_h, (0.08, 0.08, 0.12, 0.96))
    _fill(cx, cy + card_h - 3, card_w, 3, (0.28, 0.60, 0.85, 1.0))

    font = 0
    pad = 20
    tx = cx + pad
    ty = cy + card_h - pad

    _text(font, tx, ty, pg.title or "Mask Luminance", 24, (0.95, 0.97, 1.0, 1.0))
    ty -= 34
    _text(font, tx, ty, pg.phase or "Baking…", 18, (0.68, 0.78, 0.92, 1.0))

    if pg.detail:
        ty -= 24
        _text(font, tx, ty, pg.detail, 13, (0.45, 0.50, 0.58, 1.0))

    bar_y = cy + 30
    bar_h = 16
    bar_w = card_w - pad * 2
    frac = pg.step / max(1, pg.total)
    _fill(tx, bar_y, bar_w, bar_h, (0.16, 0.16, 0.22, 1.0))
    if frac > 0:
        _fill(tx, bar_y, bar_w * frac, bar_h, (0.32, 0.55, 0.95, 1.0))

    pct = f"{pg.pct}%"
    blf.size(font, 28)
    tw = blf.dimensions(font, pct)[0]
    _text(font, cx + card_w - pad - tw, cy + card_h - pad - 4, pct, 28, (0.9, 0.94, 1.0, 1.0))

    _text(font, tx, cy + 10, "Esc to cancel", 12, (0.38, 0.42, 0.50, 0.9))
    gpu.state.blend_set("NONE")


def overlay_begin(context: bpy.types.Context, *, title: str, total_steps: int) -> None:
    pg = progress_props(context)
    pg.active = True
    pg.title = title
    pg.phase = "Starting…"
    pg.detail = ""
    pg.step = 0
    pg.total = max(1, total_steps)
    pg.pct = 0
    pg.cancel_requested = False

    _HANDLES.clear()
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            space = area.spaces.active
            if space is None:
                continue
            handle = space.draw_handler_add(_draw_overlay, (), "WINDOW", "POST_PIXEL")
            _HANDLES.append((space, handle))
            area.tag_redraw()

    overlay_refresh(context)


def overlay_end(context: bpy.types.Context) -> None:
    pg = progress_props(context)
    pg.active = False
    for space, handle in _HANDLES:
        try:
            space.draw_handler_remove(handle, "WINDOW")
        except Exception:
            pass
    _HANDLES.clear()
    context.workspace.status_text_set(None)
    overlay_refresh(context)


def overlay_refresh(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def overlay_update(context: bpy.types.Context, *, phase: str, step: int, total: int, detail: str = "") -> bool:
    """Update overlay + WM progress. Returns False if the user cancelled."""
    wm = context.window_manager
    pg = progress_props(context)
    pg.phase = phase
    pg.step = min(step, total)
    pg.total = max(1, total)
    pg.detail = detail
    pg.pct = int(100 * pg.step / pg.total)

    if pg.cancel_requested:
        return False

    context.workspace.status_text_set(f"{pg.title} · {pg.pct}% · {phase}")
    overlay_refresh(context)
    return True
