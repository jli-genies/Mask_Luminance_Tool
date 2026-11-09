"""Export the image textures of every FBX under a root as PNGs, mirroring the folder structure.

Run headless with Blender; everything after ``--`` goes to this script::

    blender -b --factory-startup -P export_fbx_textures.py -- <src_root> <dst_root>
    blender -b --factory-startup -P export_fbx_textures.py -- <src_root> <dst_root> --pattern "african/fbx/*.fbx"

Each FBX matched by ``--pattern`` (relative to ``src_root``, default ``*/fbx/*.fbx``) is imported
into an empty scene, and every image texture node in its materials is saved as PNG to::

    <dst_root>/<fbx dir relative to src_root>/<fbx stem>/<image stem>.png

e.g. ``african/fbx/african_female_loPoly_0003_grp.fbx`` ->
``<dst_root>/african/fbx/african_female_loPoly_0003_grp/african_female_body_0003_albedo.png``.
Existing PNGs are skipped unless ``--overwrite`` is passed.

``--include REGEX`` keeps only images whose stem fully matches, and ``--flat`` writes every PNG
directly into ``dst_root``. Only the body albedos, all in one folder::

    ... -- <src_root> <dst_root> --include "[a-zA-Z]+_(female|male)_body_\\d{4}_albedo" --flat
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="export_fbx_textures.py")
    parser.add_argument("src_root", help="root to search for FBXs")
    parser.add_argument("dst_root", help="root to write PNGs under")
    parser.add_argument("--pattern", default=os.path.join("*", "fbx", "*.fbx"),
                        help="glob relative to src_root (default: */fbx/*.fbx)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing PNGs")
    parser.add_argument("--include", help="regex the image stem must fully match to be saved")
    parser.add_argument("--flat", action="store_true", help="write all PNGs directly into dst_root")
    return parser.parse_args(argv)


def material_images(objects) -> List[bpy.types.Image]:
    """Unique images on image texture nodes in the materials of ``objects``."""
    images = {}
    for obj in objects:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.node_tree is None:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    images[node.image.name] = node.image
    return list(images.values())


def image_stem(image: bpy.types.Image) -> str:
    src = image.filepath or image.name
    return os.path.splitext(os.path.basename(bpy.path.abspath(src)))[0] or image.name


def export_fbx(fbx_path: str, out_dir: str, overwrite: bool, include: Optional[re.Pattern],
               written: Dict[str, str]) -> int:
    """Import ``fbx_path`` into an empty scene and save its textures to ``out_dir``. Returns count.

    ``written`` maps each output path saved this run to its FBX, to catch name collisions.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    images = material_images(bpy.context.scene.objects)
    if include is not None:
        images = [image for image in images if include.fullmatch(image_stem(image))]
    if not images:
        print(f"  no matching textures found")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for image in images:
        out_path = os.path.join(out_dir, image_stem(image) + ".png")
        if out_path in written:
            print(f"  WARNING {out_path} already saved from {written[out_path]}; not overwriting")
            continue
        if os.path.exists(out_path) and not overwrite:
            print(f"  skip (exists) {out_path}")
            continue
        if not image.has_data:
            image.reload()
        if image.size[0] == 0:
            print(f"  WARNING could not load {image.name} ({image.filepath})")
            continue
        # Changing file_format re-encodes on save; byte images (tga/png) are written unchanged.
        image.file_format = "PNG"
        image.save(filepath=out_path)
        written[out_path] = fbx_path
        print(f"  {image.name} -> {out_path}")
        saved += 1
    return saved


def main() -> None:
    args = parse_args()
    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)
    fbx_paths = sorted(glob.glob(os.path.join(src_root, args.pattern)))
    print(f"{len(fbx_paths)} FBX files under {src_root}")

    include = re.compile(args.include) if args.include else None

    total, failed, empty, written = 0, [], [], {}
    for i, fbx_path in enumerate(fbx_paths, 1):
        rel = os.path.relpath(fbx_path, src_root)
        print(f"[{i}/{len(fbx_paths)}] {rel}")
        out_dir = dst_root if args.flat else os.path.join(dst_root, os.path.splitext(rel)[0])
        try:
            n = export_fbx(fbx_path, out_dir, args.overwrite, include, written)
        except Exception as exc:  # keep going; one bad FBX shouldn't stop the batch
            print(f"  ERROR {exc}")
            failed.append(rel)
            continue
        total += n
        if n == 0:
            empty.append(rel)

    print(f"Saved {total} textures to {dst_root}")
    if empty:
        print(f"{len(empty)} FBX files saved nothing (no match, or already exported):\n  "
              + "\n  ".join(empty))
    if failed:
        print(f"{len(failed)} FBX files failed:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
