"""Sanity checks that the addon package itself is well-formed.

Mirrors HeadGen's ``test_bpy_importable`` — these don't test algorithm
correctness (see ``test_blend.py``/``test_infill.py`` for that), they just
catch "the addon can't even load in Blender" mistakes without needing to
open Blender.
"""

from __future__ import annotations


def test_bpy_importable():
    import bpy

    assert hasattr(bpy, "context")


def test_addon_registers_and_unregisters_cleanly():
    import mask_luminance

    mask_luminance.register()
    try:
        assert mask_luminance.operators.CLASSES == mask_luminance.classes
    finally:
        mask_luminance.unregister()


def test_core_has_no_bpy_dependency():
    """core/ must stay pure Python — bpy usage belongs in scene/ (added later)."""
    import ast
    from pathlib import Path

    core_dir = Path(__file__).resolve().parents[1] / "core"
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            assert not any(n and n.split(".")[0] == "bpy" for n in names), (
                f"{path} imports bpy — bpy-dependent code belongs in scene/, not core/"
            )
