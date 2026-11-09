"""Flat skin color for a texture, sampled from its matching 2x2 diffuse avatar sheet.

The sheets live in a "reproject_textures" tree of generator runs::

    <root>/<ethnicity>/<timestamp>_<project>/<timestamp>_<project>_NNN.png   (1-2 kept sheets)
    <root>/<ethnicity>/<timestamp>_<project>/Payload/payload.json
    <root>/<ethnicity>/<timestamp>_<project>/Payload/reference_04.png        (skintone concept)

Run folders are named by timestamp, not asset, so the asset each run belongs to is read from
``payload.json``: its "Front" reference's ``original_path`` is the source render, e.g.
``...\\african_female_0001_grp_head_view_0.png`` -> asset ``african_female_0001``. Textures are
matched to that by (ethnicity, gender, number), so ``middleEast_male_body_0008_albedo.png``
finds ``middleeast_male_0008``.

Every sheet shares one layout (front / 3-4 / side / back, flat unlit diffuse, 1024x1024), so
skin is sampled at fixed fractional points where no feature (eyes, brows, lips, nostrils,
ears) ever lands: the centre of the back of the head, the side-view cranium and the
front-view cheek. Each point is a small patch median (rejects per-pixel generator noise), and
the sheet's color is the median of its points, so one bad point (a stray beard shadow or
blush) is outvoted. Validated over all 612 sheets: no point ever hit background, and only 4
of 1836 points strayed more than dE 3 from their sheet's median.

Runs that kept two sheets usually disagree on skin tone (40 of 44, median dE 7.8 — they're
separate generations), so averaging them would match neither. Instead the sheet nearest the
run's skintone concept (the tone the generator was asked to keep) wins. The concept is a lit,
shaded portrait with arbitrary backdrop/clothing/hair, so it's sampled only on MediaPipe
FaceMesh cheek patches, and only used to rank sheets — never as the color itself.

Pure logic (no Qt) — used by matte_luminance_ui.py's "sheet" diffuse mode, and runnable as a
QC report over a whole folder of textures::

    python diffuse_sheet.py <sheets_root> --textures <albedo_dir> --csv report.csv
"""
from __future__ import annotations

import argparse
import csv
import functools
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

AssetKey = Tuple[str, str, int]  # (ethnicity, gender, number), e.g. ("middleeast", "male", 8)

# Fractional (x, y) sample points on the 2x2 sheet; see module docstring.
SAMPLE_POINTS: Dict[str, Tuple[float, float]] = {
    "back": (0.75, 0.645),
    "side": (0.195, 0.625),
    "cheek": (0.176, 0.303),
}
PATCH_RADIUS = 4  # 9x9 patch
# A sample point further than this from its sheet's median is reported as an outlier.
OUTLIER_DE = 3.0
# Two sheets whose concept distances are closer than this are a toss-up — worth a human look
# when the sheets themselves differ by more than OUTLIER_DE.
AMBIGUOUS_CHOICE_DE = 2.0

# FaceMesh landmark rings around each cheek (below the eye, beside the nose, above the mouth
# corner) — the concept portrait's most reliably bare, mid-lit skin.
CONCEPT_CHEEK_LANDMARKS = (
    (116, 117, 118, 101, 36, 205, 187, 123),
    (345, 346, 347, 330, 266, 425, 411, 352),
)

_KEY_RE = re.compile(
    r"(african|asian|caucasian|latino|middle[_-]?east)[_-]+(female|male|f|m)(?:[_-]+body)?[_-]+(\d+)",
    re.IGNORECASE,
)
_GENDERS = {"f": "female", "m": "male"}
_RENDER_RE = re.compile(r"_grp_head_view_\d+\.\w+$", re.IGNORECASE)


def parse_asset_key(name: str) -> Optional[AssetKey]:
    """(ethnicity, gender, number) from a texture or render filename, or None if unrecognised.

    Accepts ``african_female_body_0007_albedo.png``, ``middleEast_m_0038_...``,
    ``african_female_0001_grp_head_view_0.png`` and similar.
    """
    m = _KEY_RE.search(os.path.basename(name))
    if not m:
        return None
    ethnicity = re.sub(r"[_-]", "", m[1].lower())
    gender = _GENDERS.get(m[2].lower(), m[2].lower())
    return ethnicity, gender, int(m[3])


def format_asset_key(key: AssetKey) -> str:
    return f"{key[0]}_{key[1]}_{key[2]:04d}"


@dataclass(frozen=True)
class SheetRun:
    """One generator run: its kept sheets plus the skintone concept it was given."""

    sheets: Tuple[str, ...]
    concept: Optional[str]


_INDEX_CACHE: Dict[str, Dict[AssetKey, List[SheetRun]]] = {}


def index_sheets(root: str, refresh: bool = False) -> Dict[AssetKey, List[SheetRun]]:
    """{asset key: [runs]} for every generator run under ``root``.

    Cached per root for the session (reading ~600 payload.json files takes a moment, and the
    live preview asks on every change); pass ``refresh=True`` after adding new sheets.
    """
    root = os.path.normpath(root)
    if not refresh and root in _INDEX_CACHE:
        return _INDEX_CACHE[root]
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Diffuse sheets folder not found: {root}")

    index: Dict[AssetKey, List[SheetRun]] = {}
    for payload_path in glob.glob(os.path.join(root, "**", "Payload", "payload.json"), recursive=True):
        payload_dir = os.path.dirname(payload_path)
        try:
            with open(payload_path, encoding="utf-8") as f:
                refs = json.load(f).get("reference_images") or []
        except (OSError, json.JSONDecodeError):
            continue
        # Reference descriptions are hand-typed ("Front", "FRONT", "Fron", "Skintone(Ceoncept)",
        # blank, ...), so classify by original_path instead: the source renders are all
        # "<asset>_grp_head_view_<angle>.png", and the concept is the one reference that isn't.
        renders = [r for r in refs if _RENDER_RE.search(r.get("original_path", ""))]
        others = [r for r in refs if r not in renders]
        key = parse_asset_key(renders[0]["original_path"].replace("\\", "/")) if renders else None
        sheets = tuple(sorted(glob.glob(os.path.join(os.path.dirname(payload_dir), "*.png"))))
        if not key or not sheets:
            continue
        concept = os.path.join(payload_dir, others[0]["filename"]) if len(others) == 1 else None
        index.setdefault(key, []).append(SheetRun(sheets, concept if concept and os.path.isfile(concept) else None))
    _INDEX_CACHE[root] = index
    return index


def _to_lab(rgb) -> np.ndarray:
    """uint8-range RGB (..., 3) -> CIE Lab (L 0-100), for perceptual dE distances."""
    arr = np.asarray(rgb, dtype=np.float32)
    return cv2.cvtColor(arr.reshape(-1, 1, 3) / 255.0, cv2.COLOR_RGB2Lab).reshape(arr.shape)


def _delta_e(a, b) -> float:
    return float(np.linalg.norm(_to_lab(a) - _to_lab(b)))


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_sheet_points(sheet: np.ndarray) -> Dict[str, np.ndarray]:
    """{point name: patch-median RGB (float32, (3,))} at each of ``SAMPLE_POINTS``."""
    h, w = sheet.shape[:2]
    r = PATCH_RADIUS
    out = {}
    for name, (fx, fy) in SAMPLE_POINTS.items():
        x = int(np.clip(round(fx * w), r, w - r - 1))
        y = int(np.clip(round(fy * h), r, h - r - 1))
        patch = sheet[y - r:y + r + 1, x - r:x + r + 1, :3].reshape(-1, 3)
        out[name] = np.median(patch, axis=0).astype(np.float32)
    return out


@functools.lru_cache(maxsize=2048)
def _sheet_points_cached(path: str) -> Dict[str, np.ndarray]:
    return sample_sheet_points(_read_rgb(path))


@functools.lru_cache(maxsize=512)
def concept_skin_rgb(path: str) -> Optional[np.ndarray]:
    """Approximate skin RGB of a lit concept portrait, or None if no face is found.

    Median of the cheek patches' middle-luminance half (drops both the shadowed and the
    highlighted extremes). Good enough to rank sheets by; not a flat diffuse color.
    """
    from texture_face_segment import detect_landmarks

    img = _read_rgb(path)
    pts = detect_landmarks(img)
    if pts is None:
        return None
    mask = np.zeros(img.shape[:2], np.uint8)
    for ring in CONCEPT_CHEEK_LANDMARKS:
        cv2.fillPoly(mask, [pts[list(ring)].astype(np.int32)], 1)
    px = img[mask.astype(bool)].astype(np.float32)
    if len(px) == 0:
        return None
    lum = _to_lab(px)[:, 0]
    lo, hi = np.percentile(lum, [25, 75])
    return np.median(px[(lum >= lo) & (lum <= hi)], axis=0).astype(np.float32)


@dataclass
class SheetSkin:
    """Resolved skin color for one asset, plus enough detail to QC it."""

    asset: str
    rgb: np.ndarray  # float32 (3,), 0-255 — the chosen sheet's median
    sheet: str  # the sheet the color came from
    candidates: Dict[str, np.ndarray] = field(default_factory=dict)  # every sheet -> its median RGB
    points: Dict[str, np.ndarray] = field(default_factory=dict)  # chosen sheet's per-point RGB
    max_de: float = 0.0  # largest point distance from the chosen sheet's median
    outliers: List[str] = field(default_factory=list)  # point names beyond OUTLIER_DE
    concept_rgb: Optional[np.ndarray] = None
    choice_note: str = ""  # how the sheet was picked when there was more than one

    @property
    def ambiguous_choice(self) -> bool:
        return "toss-up" in self.choice_note

    @property
    def flagged(self) -> bool:
        """True when the color shouldn't be trusted without a human look.

        One outlying point is expected occasionally and harmlessly outvoted; a majority of
        outliers means the sheet itself is inconsistent (e.g. leftover shading).
        """
        return len(self.outliers) * 2 > len(self.points) or self.ambiguous_choice

    def summary(self) -> str:
        r, g, b = (int(round(c)) for c in self.rgb)
        parts = [f"{self.asset} → ({r}, {g}, {b}) from {os.path.basename(self.sheet)}"]
        if self.choice_note:
            parts.append(self.choice_note)
        if self.outliers:
            parts.append(f"outlier points: {', '.join(self.outliers)} (max dE {self.max_de:.1f})")
        return "; ".join(parts)


def _choose_sheet(runs: List[SheetRun]) -> Tuple[str, Optional[np.ndarray], str, Dict[str, np.ndarray]]:
    """(chosen sheet, concept RGB, note, {sheet: median RGB}) — see module docstring."""
    medians = {
        s: np.median(np.stack(list(_sheet_points_cached(s).values())), axis=0).astype(np.float32)
        for run in runs for s in run.sheets
    }
    sheets = list(medians)
    if len(sheets) == 1:
        return sheets[0], None, "", medians

    concept_rgb = None
    for concept in (run.concept for run in runs if run.concept):
        concept_rgb = concept_skin_rgb(concept)
        if concept_rgb is not None:
            break
    if concept_rgb is None:
        return sheets[0], None, f"{len(sheets)} sheets, no concept face found — used the first", medians

    dists = sorted((_delta_e(medians[s], concept_rgb), s) for s in sheets)
    (best_d, best), (second_d, second) = dists[0], dists[1]
    note = f"{len(sheets)} sheets, nearest concept (dE {best_d:.1f} vs {second_d:.1f})"
    if second_d - best_d < AMBIGUOUS_CHOICE_DE and _delta_e(medians[best], medians[second]) > OUTLIER_DE:
        note += " — toss-up between differing sheets"
    return best, concept_rgb, note, medians


def resolve_asset_skin(key: AssetKey, runs: List[SheetRun]) -> SheetSkin:
    if not runs:
        raise LookupError(f"No diffuse sheet for {format_asset_key(key)}.")
    sheet, concept_rgb, note, medians = _choose_sheet(runs)
    points = _sheet_points_cached(sheet)
    rgb = medians[sheet]
    de = {name: _delta_e(p, rgb) for name, p in points.items()}
    return SheetSkin(
        asset=format_asset_key(key),
        rgb=rgb,
        sheet=sheet,
        candidates=medians,
        points=dict(points),
        max_de=max(de.values()),
        outliers=[name for name, d in de.items() if d > OUTLIER_DE],
        concept_rgb=concept_rgb,
        choice_note=note,
    )


def resolve_sheet_skin(texture_path: str, root: str) -> SheetSkin:
    """Skin color for ``texture_path``'s asset. Raises LookupError if it has no sheet."""
    key = parse_asset_key(texture_path)
    if key is None:
        raise LookupError(
            f"Can't tell which asset '{os.path.basename(texture_path)}' is — expected a name like "
            "'<ethnicity>_<gender>_body_<NNNN>_albedo.png'."
        )
    runs = index_sheets(root).get(key)
    if not runs:
        raise LookupError(f"No diffuse sheet for {format_asset_key(key)} under {root}.")
    return resolve_asset_skin(key, runs)


# ---------------------------------------------------------------------------
# QC report CLI
# ---------------------------------------------------------------------------
REPORT_FIELDS = [
    "name", "asset", "status", "r", "g", "b", "hex", "max_de", "outliers",
    "n_sheets", "choice", "sheet", "concept_rgb", "other_sheets_rgb",
]


def _rgb_str(rgb) -> str:
    return " ".join(str(int(round(c))) for c in rgb)


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sheets_root", help="Folder of diffuse sheet runs (e.g. reproject_textures)")
    ap.add_argument("--textures", help="Texture folder; report one row per texture instead of per sheet asset")
    ap.add_argument("--csv", help="Write the report here")
    args = ap.parse_args()

    index = index_sheets(args.sheets_root)
    if args.textures:
        names = sorted(n for n in os.listdir(args.textures) if n.lower().endswith((".png", ".jpg", ".tif")))
        targets = [(n, parse_asset_key(n)) for n in names]
    else:
        targets = [(format_asset_key(k), k) for k in sorted(index)]

    rows, missing, flagged = [], [], []
    for i, (name, key) in enumerate(targets, start=1):
        if key is None or key not in index:
            missing.append(name)
            rows.append({"name": name, "asset": format_asset_key(key) if key else "", "status": "no sheet"})
            continue
        skin = resolve_asset_skin(key, index[key])
        status = "flagged" if skin.flagged else ("outlier rejected" if skin.outliers else "ok")
        if skin.flagged:
            flagged.append(f"{name} ({skin.choice_note or 'inconsistent sheet'})")
        r, g, b = (int(round(c)) for c in skin.rgb)
        rows.append({
            "name": name, "asset": skin.asset, "status": status, "r": r, "g": g, "b": b,
            "hex": f"#{r:02x}{g:02x}{b:02x}", "max_de": round(skin.max_de, 2),
            "outliers": " ".join(skin.outliers), "n_sheets": len(skin.candidates),
            "choice": skin.choice_note, "sheet": skin.sheet,
            "concept_rgb": _rgb_str(skin.concept_rgb) if skin.concept_rgb is not None else "",
            "other_sheets_rgb": "; ".join(
                f"{os.path.basename(s)}={_rgb_str(c)}" for s, c in skin.candidates.items() if s != skin.sheet
            ),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(targets)}…")

    print(f"{len(index)} sheet assets indexed; {len(targets)} targets: "
          f"{len(targets) - len(missing)} matched, {len(missing)} without a sheet, {len(flagged)} flagged.")
    if missing:
        print("No sheet:", ", ".join(missing))
    if flagged:
        print("Flagged for review:\n  " + "\n  ".join(flagged))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Report -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
