"""Rename diffuse sheet run folders to carry their asset, e.g.

    african/2026-05-06_14-06-01_Cleaningtexture -> african/2026-05-06_14-06-01_TextureCleaningAfrican_f_028

The asset is read from the run's ``Payload/payload.json``: the ``original_path`` of
reference_01/02/03 is the source render (``african_female_0028_grp_head_view_0.png``). Whatever
follows the run's timestamp (``TextureCleaning``, ``Cleaningtexture``, nothing) is replaced.

Dry run by default; pass ``--apply`` to rename. Every applied rename is appended to
``rename_log.csv`` in the root so it can be undone with ``--undo``::

    python rename_diffuse_runs.py <reproject_textures> african asian
    python rename_diffuse_runs.py <reproject_textures> african asian --apply
    python rename_diffuse_runs.py <reproject_textures> --undo
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import List, Optional, Tuple

from diffuse_sheet import parse_asset_key

ETHNICITY_LABELS = {
    "african": "African",
    "asian": "Asian",
    "caucasian": "Caucasian",
    "latino": "Latino",
    "middleeast": "MiddleEast",
}
RENDER_REFS = ("reference_01.png", "reference_02.png", "reference_03.png")
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
LOG_NAME = "rename_log.csv"


def run_label(run_dir: str) -> str:
    """``TextureCleaningAfrican_f_028`` for a run folder, from its payload.json.

    Raises ValueError if the payload is missing, or its render references disagree.
    """
    payload_path = os.path.join(run_dir, "Payload", "payload.json")
    with open(payload_path, encoding="utf-8") as f:
        refs = json.load(f).get("reference_images") or []
    keys = {
        parse_asset_key(r.get("original_path", "").replace("\\", "/"))
        for r in refs
        if r.get("filename") in RENDER_REFS
    } - {None}
    if len(keys) != 1:
        raise ValueError(f"expected one asset in reference_01-03, found {sorted(keys) or 'none'}")
    ethnicity, gender, number = keys.pop()
    return f"TextureCleaning{ETHNICITY_LABELS[ethnicity]}_{gender[0]}_{number:03d}"


def plan_renames(root: str, ethnicity_dir: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """([(old_path, new_path)], [problems]) for every run folder under ``root/ethnicity_dir``."""
    parent = os.path.join(root, ethnicity_dir)
    folder_key = re.sub(r"[_-]", "", ethnicity_dir.lower())
    renames: List[Tuple[str, str]] = []
    problems: List[str] = []
    taken = {name: name for name in os.listdir(parent)}  # new name -> folder claiming it

    for name in sorted(os.listdir(parent)):
        run_dir = os.path.join(parent, name)
        if not os.path.isdir(run_dir):
            continue
        timestamp = _TIMESTAMP_RE.match(name)
        if not timestamp:
            problems.append(f"{ethnicity_dir}/{name}: no timestamp prefix, skipped")
            continue
        try:
            label = run_label(run_dir)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            problems.append(f"{ethnicity_dir}/{name}: {e}, skipped")
            continue
        if ETHNICITY_LABELS[folder_key] not in label:
            problems.append(f"{ethnicity_dir}/{name}: payload asset is {label}, wrong ethnicity folder?")
        new_name = f"{timestamp[1]}_{label}"
        if new_name == name:
            continue
        if new_name in taken:
            problems.append(f"{ethnicity_dir}/{name}: {new_name} already claimed by {taken[new_name]}, skipped")
            continue
        taken[new_name] = name
        renames.append((run_dir, os.path.join(parent, new_name)))
    return renames, problems


def undo(root: str) -> None:
    log_path = os.path.join(root, LOG_NAME)
    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for old, new in reversed(rows):
        if os.path.isdir(new) and not os.path.exists(old):
            os.rename(new, old)
            print(f"{os.path.basename(new)} -> {os.path.basename(old)}")
        else:
            print(f"cannot undo {new} -> {old}, skipped", file=sys.stderr)
    os.remove(log_path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="reproject_textures folder")
    ap.add_argument("ethnicities", nargs="*", help="Ethnicity subfolders to rename (e.g. african asian)")
    ap.add_argument("--apply", action="store_true", help="Rename (default is a dry run)")
    ap.add_argument("--undo", action="store_true", help=f"Revert the renames recorded in {LOG_NAME}")
    args = ap.parse_args(argv)

    if args.undo:
        undo(args.root)
        return 0
    if not args.ethnicities:
        ap.error("name at least one ethnicity folder")

    all_renames: List[Tuple[str, str]] = []
    for ethnicity_dir in args.ethnicities:
        renames, problems = plan_renames(args.root, ethnicity_dir)
        for old, new in renames:
            print(f"{ethnicity_dir}/{os.path.basename(old)} -> {os.path.basename(new)}")
        for p in problems:
            print(f"WARNING {p}", file=sys.stderr)
        print(f"{ethnicity_dir}: {len(renames)} to rename, {len(problems)} warnings\n")
        all_renames += renames

    if not args.apply:
        print("Dry run: pass --apply to rename.")
        return 0
    with open(os.path.join(args.root, LOG_NAME), "a", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        for old, new in all_renames:
            os.rename(old, new)
            log.writerow([old, new])
            f.flush()
    print(f"Renamed {len(all_renames)} folders; log in {os.path.join(args.root, LOG_NAME)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
