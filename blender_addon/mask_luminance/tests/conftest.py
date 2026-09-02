"""Shared fixtures for parity-testing the ported core against the original tool.

The original ``matte_luminance_blend.py`` (repo root, one level up from
``blender_addon/``) is still present and still depends on the real
``genies.meshutils.shading.texture_utils``. Rather than assume our vendored
``core/infill.py`` reimplementation behaves the same, these tests load both
the legacy CLI module and the real genies source file directly (bypassing the
``genies`` package's ``__init__.py`` chain, which pulls in USD/trimesh deps
this venv doesn't need) and diff outputs pixel-for-pixel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GENIES_PACKAGES_ROOT = REPO_ROOT.parent / "genies-shared-python-packages"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses' string-annotation resolution looks itself up via
    # sys.modules[cls.__module__] — without this, decorating MaskChannel
    # blows up with "'NoneType' object has no attribute '__dict__'".
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_genies_importable() -> None:
    """Makes the real ``genies.meshutils.shading.texture_utils`` importable.

    ``genies`` is an implicit namespace package (no ``__init__.py`` at its
    root) so this only needs the repo directory on ``sys.path`` — it does not
    trigger a full ``pip install`` of genies-shared-python-packages (whose
    pyproject.toml pins a ``usd-core`` version incompatible with this venv's
    Python and which texture_utils.py itself never actually imports).
    """
    if not GENIES_PACKAGES_ROOT.exists():
        pytest.skip(f"genies-shared-python-packages not found next to the repo: {GENIES_PACKAGES_ROOT}")
    if str(GENIES_PACKAGES_ROOT) not in sys.path:
        sys.path.insert(0, str(GENIES_PACKAGES_ROOT))


@pytest.fixture(scope="session")
def genies_texture_utils():
    """The real genies module, so vendored infill code can be diffed against it."""
    _ensure_genies_importable()
    from genies.meshutils.shading import texture_utils

    return texture_utils


@pytest.fixture(scope="session")
def legacy_blend():
    """The original standalone matte_luminance_blend.py, genies import and all."""
    legacy_path = REPO_ROOT / "matte_luminance_blend.py"
    if not legacy_path.exists():
        pytest.skip(f"Original tool not found at {legacy_path}")
    _ensure_genies_importable()
    return _load_module_from_path("legacy_matte_luminance_blend", legacy_path)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
