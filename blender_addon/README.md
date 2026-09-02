# Mask Luminance — Blender addon

Port of the repo-root `matte_luminance_blend.py` CLI tool into a Blender
addon. Follows the same two-environment / `core` + `scene` + `operators`
conventions as Genies' `HeadGen` addon.

## Layout

```
blender_addon/
  mask_luminance/                  <- the addon package (folder name = addon id)
    __init__.py                    <- bl_info (legacy format), register/unregister
    operators.py                   <- CLASSES list; empty until the properties/panels phase
    core/                          <- pure Python/numpy/opencv, NO bpy import anywhere
      blend.py                     <- ported algorithm (from matte_luminance_blend.py)
      infill.py                    <- vendored from genies.meshutils.shading.texture_utils
    tests/                         <- pytest, run against the `bpy` pip package (no Blender needed)
  pyproject.toml                   <- pytest config
  requirements-dev.txt             <- pinned dev venv (numpy, opencv, scipy, pytest, bpy, fake-bpy-module)
  .venv/                           <- not committed
```

## Why `core/infill.py` exists

The original tool depended on the internal `genies.meshutils.shading.texture_utils`
package for two functions (`extend_texture_boundaries`, `apply_extrapolation_blur`).
Blender can't see that package at all, so `core/infill.py` is a verbatim copy of
just those two functions (pure numpy/scipy/opencv, no other genies coupling).
`mask_luminance/tests/test_infill.py` diffs the copy against the real genies
source on every run to catch drift.

## Dev environment setup

Mirrors `HeadGen`'s two-environment strategy: a `.venv` with the `bpy` pip
package for fast headless testing, and real Blender for interactive/visual
work.

```powershell
py -3.13 -m venv .venv          # Blender 5.1 bundles Python 3.13
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -v
```

`requirements-dev.txt` was generated with `bpy==5.1.2` + `fake-bpy-module-5.1`
(matching the installed Blender 5.1.2), plus `numpy`/`opencv-python-headless`/
`scipy`/`pytest`. Re-freeze after adding a package: `pip freeze > requirements-dev.txt`.

### Making the addon actually load in real Blender

Blender's bundled Python ships `numpy` but **not** `cv2` or `scipy`. Blender
also does not add the user site-packages directory to its `sys.path` (by
design, for isolation), so a plain `pip install` into that interpreter is
invisible to Blender even when it succeeds. The supported fix is Blender's
own addon-dependency folder, `<user resources>/scripts/addons/modules`, which
*is* on `sys.path` — no admin rights, no touching the Blender installation
itself:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" -m pip install `
    --target "$env:APPDATA\Blender Foundation\Blender\5.1\scripts\addons\modules" `
    opencv-python-headless scipy
```

(Verified end-to-end: enabling the addon in real Blender 5.1 and running the
full pipeline through it, with these two packages staged there, produced a
byte-identical texture to the same run through the `.venv`/pytest path.)

This is a one-time local dev-machine step. At distribution time, `blender_manifest.toml`
bundles the equivalent wheels via its `wheels = [...]` field instead, so end
users installing the packaged extension won't need to run this manually.

### Symlinking for live reload

Use the "Blender Development" VSCode/Cursor extension exactly as documented
for `HeadGen` — point `BLENDER_USER_RESOURCES` at an isolated folder if you
don't want the dev symlink cluttering your personal Blender profile, and keep
`blender_manifest.toml` out of this folder until packaging for distribution
(its presence switches Blender to the newer extension loading path, which
breaks the legacy `bl_info` symlink/reload workflow).

## Status

- **Done**: `core/blend.py` + `core/infill.py` ported, verified byte-for-byte
  identical to the original CLI tool (real `genies` dependency and all) on the
  repo's checked-in test textures/masks, in both the `.venv` and real Blender.
- **Done**: `scene/images.py` (Image datablock ↔ numpy, top-down/bottom-up flip
  and colorspace-ordering both verified against a real `bpy.types.Image`) and
  `scene/bake.py` (`run_bake()`), byte-for-byte matching `core.blend.process()`
  through both the `.venv` and real Blender.
- **Done**: property groups (`MASKLUM_PG_channel`/`MASKLUM_PG_settings`),
  operators (`mask_luminance.channel_add/remove/move`, `mask_luminance.bake`),
  a `UIList` channel stack and an Image Editor sidebar panel
  (`operators.py`) — driven end-to-end via real `bpy.ops` calls in both
  environments, byte-for-byte matching the CLI tool.
- **Done**: `scene/bake.py`'s `prepare_bake()`/`bake_step()`/`bake_generator()`/
  `finalize_bake()` split the bake into one step per channel/blend-group, and
  `MASKLUM_OT_bake` is now a **modal** operator (`scene/progress_overlay.py`
  draws the progress card) that runs one step per timer tick instead of
  blocking Blender's UI thread for the whole bake. `run_bake()` still exists
  as a synchronous convenience wrapper for tests. Deliberately *not* a
  background `threading` worker — see `MASKLUM_OT_bake`'s docstring for why
  touching `bpy.data` off the main thread isn't an option in Blender.
- **Not yet built**: an operator to add a mask straight from a file browser
  rather than requiring the user to already have it loaded as an Image
  datablock.

## A hard limit on what pytest can prove for the bake operator

A modal operator only advances when Blender's real window/event loop
delivers it `TIMER` events — that loop doesn't exist in `--background` mode
or in the `bpy` pip package pytest runs against. So `test_operators.py` can
only verify `MASKLUM_OT_bake` up through `{'RUNNING_MODAL'}`; the actual
step-by-step correctness is proven separately, by `test_bake_stepped.py`
driving `scene.bake`'s `prepare_bake`/`bake_step`/`bake_generator` directly —
the exact functions `modal()` calls — and confirmed end-to-end (including the
live overlay/timer wiring not crashing) against real headless Blender 5.1.
Live interactive behavior — the progress card actually drawing, Esc
cancelling mid-bake — still needs a manual check in a real windowed Blender
session; that's out of reach for both pytest and headless `--background`
verification.

## Bugs found only by actually running this in real Blender

All three would have shipped silently if development had stopped at the
`.venv`/`bpy`-pip-package level:

1. **`cv2`/`scipy` are not visible to Blender's bundled Python at all** — see
   "Making the addon actually load in real Blender" above.
2. **`Image.colorspace_settings.name` must be assigned *before* the first
   `pixels.foreach_set()` call, never after.** Reassigning it once pixel data
   already exists silently re-interprets (corrupts) that data — confirmed to
   corrupt ~70% of rows on a real 2048x2048 write when the assignment came
   after `foreach_set`. `scene/images.py`'s `rgb_to_image()` sets it first.
3. **A module and a function can't share a name across a package boundary.**
   `scene/bake.py` originally exported a function named `bake`; `from .scene
   import bake as scene_bake` then silently bound `scene_bake` to that
   *function* rather than the module, because `scene/__init__.py`'s own `from
   .bake import bake` had already rebound the package attribute. Renamed the
   function to `run_bake`.
