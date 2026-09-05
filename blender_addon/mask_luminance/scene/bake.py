"""Runs the core pipeline against live Image datablocks.

Thin glue: pull pixels out of ``bpy.types.Image`` datablocks via
``scene.images``, hand plain arrays to ``core.blend`` (the same functions
``core``'s own tests exercise directly), then write the result back into an
Image datablock. No algorithm logic belongs here — see ``core/blend.py`` for
that.

Baking is exposed two ways:

* ``run_bake()`` — runs the whole thing synchronously and returns the result.
  Used by tests and by anything that doesn't need progress feedback.
* ``prepare_bake()`` / ``bake_step()`` / ``finalize_bake()`` (or the
  ``bake_generator()`` wrapper around the middle two) — the same work spread
  across discrete steps, one per enabled channel/blend-group, so a modal
  operator can run one step per timer tick and keep Blender's UI responsive
  on a large texture. ``operators.MASKLUM_OT_bake`` drives this; see that
  class for why a *modal* operator (rather than a background thread) is the
  right tool here — bpy.data must only ever be touched from the main thread.

``run_preview()`` is ``run_bake()`` with one extra step: the source texture
is downsampled (aspect-preserving) before anything else runs, which — since
every mask/diffuse/feature-preserve array is already resized to match the
sample's resolution — is the only change needed to make the whole pipeline
run at proxy resolution instead of full size. Benchmarked on a real
2048x2048 texture with three channels: ~6.6s at full res, ~120ms at 384px,
~37ms at 256px — cheap enough to recompute on (debounced) property changes
for a live-ish preview, still too slow to run on every raw slider-drag tick,
which is why the debounce lives in ``operators.py`` rather than here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import bpy
import numpy as np

from ..core import blend as core_blend
from ..core import color as core_color
from . import uv_bounds
from .images import image_to_rgb, result_image_name, rgb_to_image


@dataclass
class BakeState:
    """Everything ``bake_step`` needs to apply one more work item.

    Built once by ``prepare_bake`` (which does all the up-front, non-steppable
    work: pulling every Image's pixels into arrays and building the diffuse
    target), then mutated in place by repeated ``bake_step`` calls.
    """

    working: np.ndarray
    diffuse_target: np.ndarray
    mask_arrays: Dict[str, np.ndarray]
    work_items: List[Union[core_blend.MaskChannel, List[core_blend.MaskChannel]]]
    palette: Dict[str, Tuple[int, int, int]]
    luminance_only: bool
    feature_preserve: Optional[np.ndarray]
    result_name: str
    flat_target: Optional[np.ndarray] = None
    channel_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    step_index: int = 0
    diffuse_color: Optional[np.ndarray] = None
    out_of_bounds: Optional[np.ndarray] = None
    exposure: float = 0.0
    gamma: float = 1.0
    shadow_bias: float = 0.0

    @property
    def total_steps(self) -> int:
        return len(self.work_items)

    @property
    def done(self) -> bool:
        return self.step_index >= self.total_steps


def _downsample_to_max_dimension(rgb: np.ndarray, max_dimension: int, nearest: bool = False) -> np.ndarray:
    """Aspect-preserving downsample so the longer side is at most ``max_dimension``.

    A no-op (returns ``rgb`` unchanged) if it's already smaller — this never
    upsamples. ``nearest`` must match whatever ``prepare_bake`` would use to
    resize this same array (True for ``blue_paint``/``color_id`` gate masks,
    which encode discrete colors that bilinear filtering would blur into
    wrong ID matches at edges) — see ``_cached_downsampled_rgb``.
    """
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return rgb
    scale = max_dimension / longest
    target_hw = (max(1, round(h * scale)), max(1, round(w * scale)))
    return core_blend.resize_to(rgb, target_hw, nearest=nearest)


# (image.name, max_dimension, nearest) -> downsampled RGB array. See
# _cached_downsampled_rgb for why this exists and its one known limitation.
_preview_array_cache: Dict[Tuple[str, int, bool], np.ndarray] = {}


def _cached_downsampled_rgb(image: bpy.types.Image, max_dimension: int, nearest: bool = False) -> np.ndarray:
    """Downsampled-array cache used only by the live-preview path.

    Extracting a ``bpy.types.Image``'s pixels costs time proportional to its
    *native* resolution — a 4096x4096 mask is ~67M floats through
    ``foreach_get`` — even though a proxy preview only ever needs a few
    hundred pixels per side. Benchmarked impact: ~500-650ms per preview
    recompute without this cache (dominated by re-reading full-resolution
    masks that are immediately thrown away), ~15-100ms with it once the
    source/masks are cached — see ``run_preview``'s docstring numbers for the
    math itself. ``prepare_bake`` never uses this for a real (``max_dimension
    is None``) bake — only ``run_preview`` goes through it.

    Known limitation: the cache key is the Image's *name*, not its pixel
    content, so repainting or reloading a mask in place — without
    reassigning which Image datablock a channel points at — won't be picked
    up until the cache is cleared. ``clear_preview_cache()` /
    ``MASKLUM_OT_clear_preview_cache`` are the escape hatch; every
    source/mask/diffuse/feature-preserve *pointer* change already clears it
    automatically (see operators.py's pointer-property update callback).
    """
    key = (image.name, max_dimension, nearest)
    cached = _preview_array_cache.get(key)
    if cached is not None:
        return cached
    array = _downsample_to_max_dimension(image_to_rgb(image), max_dimension, nearest=nearest)
    _preview_array_cache[key] = array
    return array


def clear_preview_cache(image_name: Optional[str] = None) -> None:
    """Drops cached preview arrays. With no argument, clears every entry.

    A no-argument call also clears ``uv_bounds``'s own rasterization cache —
    every pointer-property change (source/mask/diffuse/feature-preserve
    *and* the UV source object) already routes through this "clear
    everything" path (see operators.py's pointer-change callback), so UV
    rasterizations stay covered by the same single cache-invalidation story
    with no separate wiring needed.
    """
    if image_name is None:
        _preview_array_cache.clear()
        uv_bounds.clear_uv_bounds_cache()
        return
    for key in [k for k in _preview_array_cache if k[0] == image_name]:
        del _preview_array_cache[key]


def prepare_bake(
    source: bpy.types.Image,
    channels: Sequence[core_blend.MaskChannel],
    mask_images: Dict[str, bpy.types.Image],
    diffuse_mode: str = "self",
    diffuse_image: Optional[bpy.types.Image] = None,
    feature_preserve_image: Optional[bpy.types.Image] = None,
    region_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    luminance_only: bool = True,
    self_locality_radius: float = core_blend.DEFAULT_SELF_LOCALITY_RADIUS,
    result_name: Optional[str] = None,
    max_dimension: Optional[int] = None,
    diffuse_color_override: Optional[Sequence[float]] = None,
    uv_source_object: Optional[bpy.types.Object] = None,
    uv_layer_name: str = "",
    exposure: float = 0.0,
    gamma: float = 1.0,
    shadow_bias: float = 0.0,
) -> BakeState:
    """Does every part of a bake that can't be split into per-channel steps.

    Mirrors ``core.blend.process_arrays`` exactly (same diffuse-target
    construction, same ``feature_preserve`` composited-weights handling) but
    stops short of running the channel pipeline, returning a ``BakeState``
    for ``bake_step``/``bake_generator`` to advance instead.

    ``diffuse_color_override``, if given, replaces the diffuse target (and
    flat_fill target) with this flat RGB everywhere, bypassing
    ``diffuse_mode`` — see ``core.blend.process_arrays``. Either way, the
    resulting flat color (auto-detected or overridden) is recorded on
    ``BakeState.diffuse_color`` for the panel to display; it is ``None`` only
    for an un-overridden ``diffuse_mode='uv'``, which has no single color.

    ``exposure``/``gamma``/``shadow_bias``, if not left at their neutral
    defaults, are stashed on the returned ``BakeState`` and applied by
    ``finalize_bake`` as one final global grade after every channel has run
    — see ``core.color.apply_exposure_gamma``.

    ``max_dimension``, if given, downsamples the source texture first (see
    ``_downsample_to_max_dimension``) — every mask/diffuse/feature-preserve
    array below is already resized to match the source's resolution, so
    shrinking just that one array is enough to run the whole pipeline at
    proxy resolution. Extraction itself is routed through
    ``_cached_downsampled_rgb`` in that case too, since re-reading a
    full-resolution mask on every call is the dominant cost otherwise (see
    that function's docstring). This is what ``run_preview`` uses; leave
    ``max_dimension`` as ``None`` for a real (full-resolution, always-fresh,
    never-cached) bake.
    """

    def _extract(image: bpy.types.Image, nearest: bool = False) -> np.ndarray:
        if max_dimension is not None:
            return _cached_downsampled_rgb(image, max_dimension, nearest=nearest)
        return image_to_rgb(image)

    sample = _extract(source)
    sample_hw = sample.shape[:2]

    active = [ch for ch in channels if ch.enabled]
    mask_arrays: Dict[str, np.ndarray] = {}
    for ch in active:
        if ch.name not in mask_images:
            raise KeyError(f"No mask Image provided for enabled channel '{ch.name}'.")
        nearest = ch.gate_mode in ("blue_paint", "color_id")
        mask_arrays[ch.name] = core_blend.resize_to(_extract(mask_images[ch.name], nearest=nearest), sample_hw, nearest=nearest)

    palette = region_palette or core_blend.DEFAULT_REGION_PALETTE

    feature_preserve = None
    if feature_preserve_image is not None:
        fp_arr = core_blend.resize_to(_extract(feature_preserve_image), sample_hw, nearest=False)
        feature_preserve = core_blend.composite_weights(fp_arr)

    exclude = np.zeros(sample_hw, dtype=np.float32)
    for ch in active:
        exclude = np.maximum(exclude, core_blend.compute_channel_gate(mask_arrays[ch.name], ch, palette))
    if feature_preserve is not None:
        exclude = np.maximum(exclude, feature_preserve)

    # Rasterized directly at sample_hw (already the preview-downsampled
    # resolution when max_dimension is set), so no separate resize step is
    # needed — uv_bounds' own cache (keyed on (object, uv layer, resolution))
    # then makes repeated preview ticks at that resolution free after the
    # first. See scene.uv_bounds for what this actually clips.
    out_of_bounds = None
    if uv_source_object is not None:
        out_of_bounds = uv_bounds.out_of_bounds_mask(uv_source_object, sample_hw, uv_layer_name or None)

    diffuse_img = None
    if diffuse_color_override is not None:
        diffuse_target = np.broadcast_to(
            np.asarray(diffuse_color_override, dtype=np.float32), sample.shape
        ).copy()
    elif diffuse_mode == "self":
        diffuse_target = core_blend.build_local_diffuse_target(
            sample, exclude, self_locality_radius, out_of_bounds=out_of_bounds
        )
    else:
        if diffuse_image is None:
            raise ValueError("diffuse_image is required unless diffuse_mode='self'.")
        diffuse_img = _extract(diffuse_image)
        diffuse_target = core_blend.make_diffuse_target(sample, diffuse_img, diffuse_mode)

    flat_target = None
    if any(ch.flat_fill for ch in active):
        flat_target = diffuse_target if diffuse_color_override is not None else core_blend.build_local_diffuse_target(
            sample, exclude, self_locality_radius, flat=True, out_of_bounds=out_of_bounds
        )

    if diffuse_color_override is not None:
        diffuse_color = np.asarray(diffuse_color_override, dtype=np.float32)
    else:
        diffuse_color = core_blend.resolve_diffuse_color(
            sample, diffuse_mode, exclude, diffuse_img, out_of_bounds=out_of_bounds
        )

    return BakeState(
        working=sample.copy(),
        diffuse_target=diffuse_target,
        mask_arrays=mask_arrays,
        work_items=core_blend.group_active_channels(active),
        palette=palette,
        luminance_only=luminance_only,
        feature_preserve=feature_preserve,
        result_name=result_name or result_image_name(source),
        flat_target=flat_target,
        diffuse_color=diffuse_color,
        out_of_bounds=out_of_bounds,
        exposure=exposure,
        gamma=gamma,
        shadow_bias=shadow_bias,
    )


def bake_step(state: BakeState) -> bool:
    """Applies the next work item (one channel or one blend group) in place.

    Returns False (without doing anything) once every item has been applied.
    """
    if state.done:
        return False

    item = state.work_items[state.step_index]
    if isinstance(item, list):
        state.working, group_soft = core_blend.apply_blend_group(
            state.working, state.diffuse_target, state.mask_arrays, item, state.palette,
            state.luminance_only, state.feature_preserve, state.flat_target, state.out_of_bounds,
        )
        state.channel_masks.update(group_soft)
    else:
        state.working, soft = core_blend.apply_mask_channel(
            state.working, state.diffuse_target, state.mask_arrays[item.name], item, state.palette,
            state.luminance_only, state.feature_preserve, state.flat_target, state.out_of_bounds,
        )
        state.channel_masks[item.name] = soft

    state.step_index += 1
    return True


def bake_generator(state: BakeState) -> Generator[Tuple[int, int], None, None]:
    """Yields ``(completed_steps, total_steps)`` after each work item.

    A modal operator calls ``next()`` on this once per timer tick — see
    ``operators.MASKLUM_OT_bake.modal``.
    """
    while bake_step(state):
        yield state.step_index, state.total_steps


def finalize_bake(state: BakeState) -> bpy.types.Image:
    """Writes the (fully-stepped) result into an Image datablock.

    Applies the global exposure/gamma grade (if not left at its neutral
    defaults) here, after every channel's own step has run and before the
    result ever reaches an Image datablock — the same point a fresh
    ``run_preview`` recomputes it from, so the live preview and a real bake
    always agree.
    """
    working = state.working
    if state.exposure != 0.0 or state.gamma != 1.0 or state.shadow_bias != 0.0:
        working = core_color.apply_exposure_gamma(
            working, exposure=state.exposure, gamma=state.gamma, shadow_bias=state.shadow_bias
        )
    existing = bpy.data.images.get(state.result_name)
    return rgb_to_image(working, state.result_name, existing=existing, non_color=False)


def run_bake(
    source: bpy.types.Image,
    channels: Sequence[core_blend.MaskChannel],
    mask_images: Dict[str, bpy.types.Image],
    diffuse_mode: str = "self",
    diffuse_image: Optional[bpy.types.Image] = None,
    feature_preserve_image: Optional[bpy.types.Image] = None,
    region_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    luminance_only: bool = True,
    self_locality_radius: float = core_blend.DEFAULT_SELF_LOCALITY_RADIUS,
    result_name: Optional[str] = None,
    max_dimension: Optional[int] = None,
    diffuse_color_override: Optional[Sequence[float]] = None,
    uv_source_object: Optional[bpy.types.Object] = None,
    uv_layer_name: str = "",
    exposure: float = 0.0,
    gamma: float = 1.0,
    shadow_bias: float = 0.0,
) -> Tuple[bpy.types.Image, Dict[str, np.ndarray]]:
    """Bakes every enabled channel onto ``source`` synchronously and writes the result to an Image.

    Runs every step of ``prepare_bake``/``bake_step``/``finalize_bake`` back
    to back with no progress feedback — for tests, and for anything that
    doesn't need the modal/progress-overlay path. ``max_dimension`` is
    forwarded to ``prepare_bake``; see ``run_preview`` for the intended use.
    See ``run_bake_with_color`` for a variant that also returns the resolved
    flat diffuse color, used by the live-preview path to feed the panel's
    diffuse-color swatch.
    """
    image, masks, _ = run_bake_with_color(
        source, channels, mask_images,
        diffuse_mode=diffuse_mode,
        diffuse_image=diffuse_image,
        feature_preserve_image=feature_preserve_image,
        region_palette=region_palette,
        luminance_only=luminance_only,
        self_locality_radius=self_locality_radius,
        result_name=result_name,
        max_dimension=max_dimension,
        diffuse_color_override=diffuse_color_override,
        uv_source_object=uv_source_object,
        uv_layer_name=uv_layer_name,
        exposure=exposure,
        gamma=gamma,
        shadow_bias=shadow_bias,
    )
    return image, masks


def run_bake_with_color(
    source: bpy.types.Image,
    channels: Sequence[core_blend.MaskChannel],
    mask_images: Dict[str, bpy.types.Image],
    diffuse_mode: str = "self",
    diffuse_image: Optional[bpy.types.Image] = None,
    feature_preserve_image: Optional[bpy.types.Image] = None,
    region_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    luminance_only: bool = True,
    self_locality_radius: float = core_blend.DEFAULT_SELF_LOCALITY_RADIUS,
    result_name: Optional[str] = None,
    max_dimension: Optional[int] = None,
    diffuse_color_override: Optional[Sequence[float]] = None,
    uv_source_object: Optional[bpy.types.Object] = None,
    uv_layer_name: str = "",
    exposure: float = 0.0,
    gamma: float = 1.0,
    shadow_bias: float = 0.0,
) -> Tuple[bpy.types.Image, Dict[str, np.ndarray], Optional[np.ndarray]]:
    """``run_bake``, plus the resolved flat diffuse color (see ``BakeState.diffuse_color``).

    A separate function (rather than changing ``run_bake``'s return value)
    so existing callers that only want the Image/masks pair keep working
    unchanged.
    """
    state = prepare_bake(
        source, channels, mask_images,
        diffuse_mode=diffuse_mode,
        diffuse_image=diffuse_image,
        feature_preserve_image=feature_preserve_image,
        region_palette=region_palette,
        luminance_only=luminance_only,
        self_locality_radius=self_locality_radius,
        result_name=result_name,
        max_dimension=max_dimension,
        diffuse_color_override=diffuse_color_override,
        uv_source_object=uv_source_object,
        uv_layer_name=uv_layer_name,
        exposure=exposure,
        gamma=gamma,
        shadow_bias=shadow_bias,
    )
    while bake_step(state):
        pass
    return finalize_bake(state), state.channel_masks, state.diffuse_color


DEFAULT_PREVIEW_MAX_DIMENSION = 384
PREVIEW_SUFFIX = "_preview"


def run_preview(
    source: bpy.types.Image,
    channels: Sequence[core_blend.MaskChannel],
    mask_images: Dict[str, bpy.types.Image],
    diffuse_mode: str = "self",
    diffuse_image: Optional[bpy.types.Image] = None,
    feature_preserve_image: Optional[bpy.types.Image] = None,
    region_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    luminance_only: bool = True,
    self_locality_radius: float = core_blend.DEFAULT_SELF_LOCALITY_RADIUS,
    max_dimension: int = DEFAULT_PREVIEW_MAX_DIMENSION,
    diffuse_color_override: Optional[Sequence[float]] = None,
    uv_source_object: Optional[bpy.types.Object] = None,
    uv_layer_name: str = "",
    exposure: float = 0.0,
    gamma: float = 1.0,
    shadow_bias: float = 0.0,
) -> bpy.types.Image:
    """``run_bake`` at proxy resolution, into a dedicated ``<source>_preview`` Image.

    Same algorithm as a real bake (no approximation beyond the resolution
    itself), so what you see is what a full bake converges toward — just
    computed cheaply enough (see the module docstring's benchmark numbers)
    to redo on every settled property change. Callers that want to skip
    incomplete channels (e.g. one added but not yet given a mask) rather
    than raise should filter ``channels``/``mask_images`` themselves before
    calling this — it has the same "every enabled channel needs a mask"
    contract as ``run_bake``. See ``run_preview_with_color`` for a variant
    that also returns the resolved flat diffuse color.
    """
    image, _ = run_preview_with_color(
        source, channels, mask_images,
        diffuse_mode=diffuse_mode,
        diffuse_image=diffuse_image,
        feature_preserve_image=feature_preserve_image,
        region_palette=region_palette,
        luminance_only=luminance_only,
        self_locality_radius=self_locality_radius,
        max_dimension=max_dimension,
        diffuse_color_override=diffuse_color_override,
        uv_source_object=uv_source_object,
        uv_layer_name=uv_layer_name,
        exposure=exposure,
        gamma=gamma,
        shadow_bias=shadow_bias,
    )
    return image


def run_preview_with_color(
    source: bpy.types.Image,
    channels: Sequence[core_blend.MaskChannel],
    mask_images: Dict[str, bpy.types.Image],
    diffuse_mode: str = "self",
    diffuse_image: Optional[bpy.types.Image] = None,
    feature_preserve_image: Optional[bpy.types.Image] = None,
    region_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    luminance_only: bool = True,
    self_locality_radius: float = core_blend.DEFAULT_SELF_LOCALITY_RADIUS,
    max_dimension: int = DEFAULT_PREVIEW_MAX_DIMENSION,
    diffuse_color_override: Optional[Sequence[float]] = None,
    uv_source_object: Optional[bpy.types.Object] = None,
    uv_layer_name: str = "",
    exposure: float = 0.0,
    gamma: float = 1.0,
    shadow_bias: float = 0.0,
) -> Tuple[bpy.types.Image, Optional[np.ndarray]]:
    """``run_preview``, plus the resolved flat diffuse color for the panel's swatch.

    Cheap to call on every settled property change (same proxy-resolution
    budget as ``run_preview``) since it reuses ``prepare_bake``'s downsampled
    array cache rather than re-extracting full-resolution pixels.
    """
    result_image, _, diffuse_color = run_bake_with_color(
        source, channels, mask_images,
        diffuse_mode=diffuse_mode,
        diffuse_image=diffuse_image,
        feature_preserve_image=feature_preserve_image,
        region_palette=region_palette,
        luminance_only=luminance_only,
        self_locality_radius=self_locality_radius,
        result_name=result_image_name(source, suffix=PREVIEW_SUFFIX),
        max_dimension=max_dimension,
        diffuse_color_override=diffuse_color_override,
        uv_source_object=uv_source_object,
        uv_layer_name=uv_layer_name,
        exposure=exposure,
        gamma=gamma,
        shadow_bias=shadow_bias,
    )
    return result_image, diffuse_color
