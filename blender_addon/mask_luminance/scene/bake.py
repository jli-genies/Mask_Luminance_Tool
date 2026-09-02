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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import bpy
import numpy as np

from ..core import blend as core_blend
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
    channel_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    step_index: int = 0

    @property
    def total_steps(self) -> int:
        return len(self.work_items)

    @property
    def done(self) -> bool:
        return self.step_index >= self.total_steps


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
) -> BakeState:
    """Does every part of a bake that can't be split into per-channel steps.

    Mirrors ``core.blend.process_arrays`` exactly (same diffuse-target
    construction, same ``feature_preserve`` composited-weights handling) but
    stops short of running the channel pipeline, returning a ``BakeState``
    for ``bake_step``/``bake_generator`` to advance instead.
    """
    sample = image_to_rgb(source)
    sample_hw = sample.shape[:2]

    active = [ch for ch in channels if ch.enabled]
    mask_arrays: Dict[str, np.ndarray] = {}
    for ch in active:
        if ch.name not in mask_images:
            raise KeyError(f"No mask Image provided for enabled channel '{ch.name}'.")
        nearest = ch.gate_mode in ("blue_paint", "color_id")
        mask_arrays[ch.name] = core_blend.resize_to(
            image_to_rgb(mask_images[ch.name]), sample_hw, nearest=nearest
        )

    palette = region_palette or core_blend.DEFAULT_REGION_PALETTE

    feature_preserve = None
    if feature_preserve_image is not None:
        fp_arr = core_blend.resize_to(image_to_rgb(feature_preserve_image), sample_hw, nearest=False)
        feature_preserve = core_blend.composite_weights(fp_arr)

    if diffuse_mode == "self":
        exclude = np.zeros(sample_hw, dtype=np.float32)
        for ch in active:
            exclude = np.maximum(exclude, core_blend.compute_channel_gate(mask_arrays[ch.name], ch, palette))
        if feature_preserve is not None:
            exclude = np.maximum(exclude, feature_preserve)
        diffuse_target = core_blend.build_local_diffuse_target(sample, exclude, self_locality_radius)
    else:
        if diffuse_image is None:
            raise ValueError("diffuse_image is required unless diffuse_mode='self'.")
        diffuse_target = core_blend.make_diffuse_target(sample, image_to_rgb(diffuse_image), diffuse_mode)

    return BakeState(
        working=sample.copy(),
        diffuse_target=diffuse_target,
        mask_arrays=mask_arrays,
        work_items=core_blend.group_active_channels(active),
        palette=palette,
        luminance_only=luminance_only,
        feature_preserve=feature_preserve,
        result_name=result_name or result_image_name(source),
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
            state.luminance_only, state.feature_preserve,
        )
        state.channel_masks.update(group_soft)
    else:
        state.working, soft = core_blend.apply_mask_channel(
            state.working, state.diffuse_target, state.mask_arrays[item.name], item, state.palette,
            state.luminance_only, state.feature_preserve,
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
    """Writes the (fully-stepped) result into an Image datablock."""
    existing = bpy.data.images.get(state.result_name)
    return rgb_to_image(state.working, state.result_name, existing=existing, non_color=False)


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
) -> Tuple[bpy.types.Image, Dict[str, np.ndarray]]:
    """Bakes every enabled channel onto ``source`` synchronously and writes the result to an Image.

    Runs every step of ``prepare_bake``/``bake_step``/``finalize_bake`` back
    to back with no progress feedback — for tests, and for anything that
    doesn't need the modal/progress-overlay path.
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
    )
    while bake_step(state):
        pass
    return finalize_bake(state), state.channel_masks
